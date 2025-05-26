from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
import numpy as np
import cv2
import math
import copy

class KeyboardController:
    def __init__(self, robot, leader_config, env_config, render_mode='human', verbose=False, *args, **kwargs):
        self.is_ready = False
        self.require_end = False
        self.shutdown = False

        self.num_limbs = robot.num_limbs
        self.follower_limb_names = robot.limb_names

        self.curr_ts = {limb_name: np.zeros(3) for limb_name in self.follower_limb_names}
        self.curr_Rs = {limb_name: np.eye(3) for limb_name in self.follower_limb_names}
        self.eef_type = robot.robot_config.eef_type
        # keyboard only supports binary open-close
        if self.eef_type == 'parallel_gripper':
            self.gripper_command = {limb_name: 'open' for limb_name in self.follower_limb_names}
        elif self.eef_type == 'power_gripper':
            self.gripper_command = {limb_name: ('open', (0.0, 0.0)) for limb_name in self.follower_limb_names}
        elif self.eef_type is None:
            self.gripper_command = []

        self.init_curr_ts = {limb_name: np.zeros(3) for limb_name in self.follower_limb_names}
        self.init_curr_Rs = {limb_name: np.eye(3) for limb_name in self.follower_limb_names}
        self.init_gripper_command = copy.deepcopy(self.gripper_command)

        self.selected_eef_idx = 0
        self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]

        self.key_info = [
            ["q", "reset simulation"],
            ["spacebar", "toggle gripper (open/close)"],
            [["w","a","s","d"], "move arm horizontally in xy plane"],
            [["r","f"], "move arm vertically"],
            [["z","x"], "rotate arm about x-axis"],
            [["t","g"], "rotate arm about y-axis"],
            [["c","v"], "rotate arm about z-axis"],
            ["1", "change eef"]
        ]
        self.im_patch_list, self.bold_im_patch_list = [], []
        self.im_patch_dict = {}
        self.patch_height, self.patch_width = 40, 700
        for i, (key, info) in enumerate(self.key_info):
            key_str = "-".join(key) if isinstance(key, list) else key
            blank_im_patch = np.zeros([self.patch_height, self.patch_width, 3], dtype=np.uint8)
            im_patch = cv2.putText(blank_im_patch.copy(), f"{key_str.ljust(12)}{info}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            bold_im_patch = cv2.putText(blank_im_patch.copy(), f"{key_str.ljust(12)}{info}", (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            self.im_patch_list.append(im_patch)
            self.bold_im_patch_list.append(bold_im_patch)
            if key == 'spacebar':
                self.im_patch_dict[32] = i
            elif isinstance(key, str):
                self.im_patch_dict[ord(key)] = i
            elif isinstance(key, list):
                for k in key:
                    self.im_patch_dict[ord(k)] = i

        blank_im_patch = np.zeros([self.patch_height, self.patch_width, 3], dtype=np.uint8)
        blank_im_patch = cv2.putText(blank_im_patch, f"current eef: {self.selected_eef_name} pressed_key: {-1}" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        self.im_patch_list.append(blank_im_patch)

        self.view_im = np.concatenate(self.im_patch_list, axis=0)
        self.initialized = True
        self.require_end = False
        self.is_ready = True


        self.Rx = pr.matrix_from_euler([0.1, 0, 0], 0, 1, 2, True)
        self.Ry = pr.matrix_from_euler([0, 0.1, 0], 0, 1, 2, True)
        self.Rz = pr.matrix_from_euler([0, 0, 0.1], 0, 1, 2, True)

        return

    def reset(self, ):
        self.curr_ts = {limb_name: np.zeros(3) for limb_name in self.follower_limb_names}
        self.curr_Rs = {limb_name: np.eye(3) for limb_name in self.follower_limb_names}
        self.gripper_command = self.init_gripper_command.copy()
        return

    def get_curr_pose(self):
        out_pose = {}
        for limb_name in self.follower_limb_names:
            R = self.curr_Rs[limb_name]
            t = self.curr_ts[limb_name]
            Rt = pt.transform_from(R,t)
            out_pose[limb_name] = Rt
        return out_pose

    def get_status(self):
        '''
        q         	reset simulation
        spacebar  	toggle gripper (open/close)
        w-a-s-d   	move arm horizontally in x-y plane
        r-f       	move arm vertically
        z-x       	rotate arm about x-axis
        t-g       	rotate arm about y-axis
        c-v       	rotate arm about z-axis

        '''
        #view_im = self.view_im.
        cv2.imshow('keyboard', self.view_im)
        key = cv2.waitKey(1)

        if key in self.im_patch_dict:
            change_line = self.im_patch_dict[key]
            blank_im_patch = np.zeros([self.patch_height, self.patch_width, 3], dtype=np.uint8)
            blank_im_patch = cv2.putText(blank_im_patch, f"current eef: {self.selected_eef_name} pressed_key: {chr(key)}",
                                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.im_patch_list[-1] = blank_im_patch
            self.view_im = np.concatenate(self.im_patch_list, axis=0)
            start_h, end_h = change_line * self.patch_height, (change_line+1) * self.patch_height
            self.view_im[start_h:end_h, :] = self.bold_im_patch_list[change_line]

        if key == -1:
            return self.get_curr_pose(), self.gripper_command
        elif key == ord("w"): # y
            self.curr_ts[self.selected_eef_name][0] -= 0.01
        elif key == ord("s"): # ys
            self.curr_ts[self.selected_eef_name][0] += 0.01
        elif key == ord("a"): # x
            self.curr_ts[self.selected_eef_name][1] += 0.01
        elif key == ord("d"): # x
            self.curr_ts[self.selected_eef_name][1] -= 0.01
        elif key == ord("r"):
            self.curr_ts[self.selected_eef_name][2] += 0.01
        elif key == ord("f"):
            self.curr_ts[self.selected_eef_name][2] -= 0.01
        elif key == ord("z"):
            self.curr_Rs[self.selected_eef_name] = self.Rx @ self.curr_Rs[self.selected_eef_name]
        elif key == ord("x"):
            self.curr_Rs[self.selected_eef_name] = self.Rx.T @ self.curr_Rs[self.selected_eef_name]
        elif key == ord("t"):
            self.curr_Rs[self.selected_eef_name] = self.Ry @ self.curr_Rs[self.selected_eef_name]
        elif key == ord("g"):
            self.curr_Rs[self.selected_eef_name] = self.Ry.T @ self.curr_Rs[self.selected_eef_name]
        elif key == ord("c"):
            self.curr_Rs[self.selected_eef_name] = self.Rz @ self.curr_Rs[self.selected_eef_name]
        elif key == ord("v"):
            self.curr_Rs[self.selected_eef_name] = self.Rz.T @ self.curr_Rs[self.selected_eef_name]
        elif key == 32: # spacebar
            if self.eef_type == 'parallel_gripper':
                self.gripper_command[self.selected_eef_name] = 'close' if self.gripper_command[self.selected_eef_name] == 'open' else 'open'
            elif self.eef_type == 'power_gripper':
                curr_status = self.gripper_command[self.selected_eef_name][0]
                self.gripper_command[self.selected_eef_name] = ('close', (0.0, 0.0)) if curr_status == 'open' else ('open', (0.0, 0.0))
        elif key == ord('q'):
            self.require_end = True
        elif key == ord('1'):
            self.selected_eef_idx = (self.selected_eef_idx + 1) % self.num_limbs
            self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]
        else:
            print("Invalid key pressed", key)
        #print(self.curr_pose)
        return self.get_curr_pose(), self.gripper_command

    def initialize(self): return

    def close(self): return

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def launch_init(self, *args, **kwargs):
        self.is_ready = True
        self.require_end = False
        self.reset()
        return

    def close_init(self, *args, **kwargs):
        return


if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)
    leader = KeyboardController(robot, leader_config, env_config, render_mode='human')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)
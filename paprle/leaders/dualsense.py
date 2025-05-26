from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
import numpy as np
import cv2
import math
from dualsense_controller import DualSenseController
from threading import Lock
import copy
# https://github.com/yesbotics/dualsense-controller-python
class DualSense:
    def __init__(self, robot, leader_config, *args, **kwargs):
        self.num_limbs = robot.num_limbs
        self.follower_limb_names = robot.limb_names

        self.curr_ts = {limb_name: np.zeros(3) for limb_name in self.follower_limb_names}
        self.curr_Rs = {limb_name: np.eye(3) for limb_name in self.follower_limb_names}
        self.eef_type = robot_config.robot_cfg.eef_type
        # keyboard only supports binary open-close
        if self.eef_type == 'parallel_gripper':
            self.gripper_command = {limb_name: 'open' for limb_name in self.follower_limb_names}
        elif self.eef_type == 'power_gripper':
            self.gripper_command = {limb_name: ('open', (0.0, 0.0)) for limb_name in self.follower_limb_names}
        elif self.eef_type is None:
            self.gripper_command = []

        self.init_curr_ts = {limb_name: np.zeros(3) for limb_name in self.follower_limb_names}
        self.init_curr_Rs = {limb_name: np.eye(3) for limb_name in self.follower_limb_names}
        self.init_gripper_command = self.gripper_command.copy()

        self.selected_eef_idx = 0
        self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]

        self.initialized = True
        self.require_end = False
        self.is_ready = True

        self.Rx = pr.matrix_from_euler([0.01, 0, 0], 0, 1, 2, True)
        self.Ry = pr.matrix_from_euler([0, 0.01, 0], 0, 1, 2, True)
        self.Rz = pr.matrix_from_euler([0, 0, 0.01], 0, 1, 2, True)

        self.controller = DualSenseController()
        self.controller.btn_ps.on_down(self.on_ps_btn_pressed)


        self.controller.activate()

        self.pos_z_pressed = False
        self.pos_lock = Lock()
        self.ori_yaw_pressed = False
        self.ori_lock = Lock()

        self.R_lock = Lock()
        self.T_lock = Lock()
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

    def on_ps_btn_pressed(self):
        print('PS button pressed -> stop')
        self.require_end = True
        return

    def on_btn_r3_pressed(self,):
        with self.T_lock:
            if self.pos_z_pressed:
                self.curr_ts[self.selected_eef_name][2] += 0.001
            else:
                self.curr_ts[self.selected_eef_name][2] -= 0.001

    def on_right_stick_changed(self, right_stick):
        x, y = right_stick.x, right_stick.y
        with self.T_lock:
            self.curr_ts[self.selected_eef_name][1] += y * 1e-4
            self.curr_ts[self.selected_eef_name][0] += x * 1e-4
        return

    def on_right_trigger_changed(self, value):
        with self.pos_lock:
            self.pos_z_pressed = value > 0.5

    def on_left_trigger_changed(self, value):
        with self.ori_lock:
            self.ori_yaw_pressed = value > 0.5




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

        if self.controller.left_stick.value.x > 0.7:
            self.curr_Rs[self.selected_eef_name] = self.Rx @ self.curr_Rs[self.selected_eef_name]
        elif self.controller.left_stick.value.x < -0.7:
            self.curr_Rs[self.selected_eef_name] = self.Rx.T @ self.curr_Rs[self.selected_eef_name]
        if self.controller.left_stick.value.y > 0.7:
            self.curr_Rs[self.selected_eef_name] = self.Ry @ self.curr_Rs[self.selected_eef_name]
        elif self.controller.left_stick.value.y < -0.7:
            self.curr_Rs[self.selected_eef_name] = self.Ry.T @ self.curr_Rs[self.selected_eef_name]
        if self.controller.btn_l3.pressed:
            with self.ori_lock:
                if self.controller.btn_l1.pressed:
                    self.curr_Rs[self.selected_eef_name] = self.Rz @ self.curr_Rs[self.selected_eef_name]
                else:
                    self.curr_Rs[self.selected_eef_name] = self.Rz.T @ self.curr_Rs[self.selected_eef_name]

        self.curr_ts[self.selected_eef_name][1] += self.controller.right_stick.value.x * 1e-3
        self.curr_ts[self.selected_eef_name][0] += self.controller.right_stick.value.y * 1e-3
        if self.controller.btn_r3.pressed:
            with self.T_lock:
                if self.controller.btn_r1.pressed:
                    self.curr_ts[self.selected_eef_name][2] += 0.001
                else:
                    self.curr_ts[self.selected_eef_name][2] -= 0.001

        self.gripper_command[self.selected_eef_name] = np.array([self.controller.right_trigger.value])

        if self.controller.btn_cross.pressed:
            self.selected_eef_idx = (self.selected_eef_idx + 1) % self.num_limbs
            self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]
            print("selected eef: ", self.selected_eef_name)

        return self.get_curr_pose(), self.gripper_command

    def initialize(self): return

    def close(self): return

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def launch_init(self, *args, **kwargs):
        self.reset()
        self.is_ready = True
        self.require_end = False
        return

    def close_init(self, *args, **kwargs):
        return



if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)
    leader = DualSense(robot, leader_config, env_config, render_mode='human')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)
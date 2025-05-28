from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
import numpy as np
import cv2
import math
import threading
from threading import Lock
import copy
from pyjoycon import JoyCon
from pyjoycon.constants import JOYCON_VENDOR_ID, JOYCON_PRODUCT_IDS
from pyjoycon.constants import JOYCON_L_PRODUCT_ID, JOYCON_R_PRODUCT_ID
from dualsense_controller.core.hidapi.hidapi import Device, enumerate
# https://github.com/tocoteron/joycon-python
class JoyConFixed(JoyCon):
    def __init__(self, vendor_id: int, product_id: int, serial: str = None, simple_mode=False, device_info=None):
        self.leader_config = leader_config
        if vendor_id != JOYCON_VENDOR_ID:
            raise ValueError(f'vendor_id is invalid: {vendor_id!r}')

        if product_id not in JOYCON_PRODUCT_IDS:
            raise ValueError(f'product_id is invalid: {product_id!r}')

        self.vendor_id   = vendor_id
        self.product_id  = product_id
        self.serial      = serial
        self.simple_mode = simple_mode  # TODO: It's for reporting mode 0x3f

        # setup internal state
        self._input_hooks = []
        self._input_report = bytes(self._INPUT_REPORT_SIZE)
        self._packet_number = 0
        self.set_accel_calibration((0, 0, 0), (1, 1, 1))
        self.set_gyro_calibration((0, 0, 0), (1, 1, 1))

        # connect to joycon
        self._joycon_device = Device(device_info)#self._open(vendor_id, product_id, serial=None)
        self._read_joycon_data()
        self._setup_sensors()

        # start talking with the joycon in a daemon thread
        self._update_input_report_thread = threading.Thread(target=self._update_input_report)
        self._update_input_report_thread.setDaemon(True)
        self._update_input_report_thread.start()

class JoyConController:
    def __init__(self, robot, leader_config, *args, **kwargs):
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
        self.init_gripper_command = self.gripper_command.copy()

        self.selected_eef_idx = 0
        self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]

        self.initialized = True
        self.require_end = False
        self.is_ready = True

        self.Rx = pr.matrix_from_euler([0.01, 0, 0], 0, 1, 2, True)
        self.Ry = pr.matrix_from_euler([0, 0.01, 0], 0, 1, 2, True)
        self.Rz = pr.matrix_from_euler([0, 0, 0.01], 0, 1, 2, True)

        output = enumerate(vendor_id=JOYCON_VENDOR_ID, product_id=JOYCON_L_PRODUCT_ID)
        if len(output) == 0:
            raise ValueError("No right joycon found")
        else: device_info = output[0]
        self.left_joycon = JoyConFixed(vendor_id=device_info.vendor_id, product_id=device_info.product_id,
                             serial=device_info.serial_number,
                             device_info=device_info)
        self.left_joycon_x_offset, self.left_joycon_x_scale = 2000, 1500
        self.left_joycon_y_offset, self.left_joycon_y_scale = 2200, 1000

        output = enumerate(vendor_id=JOYCON_VENDOR_ID, product_id=JOYCON_R_PRODUCT_ID)
        if len(output) == 0:
            raise ValueError("No right joycon found")
        else: device_info = output[0]
        self.right_joycon = JoyConFixed(vendor_id=device_info.vendor_id, product_id=device_info.product_id,
                              serial=device_info.serial_number, device_info=device_info)
        self.right_joycon_x_offset, self.right_joycon_x_scale = 2200, 1300
        self.right_joycon_y_offset, self.right_joycon_y_scale = 2000, 1000

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
        self.reset()
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
        left_status = self.left_joycon.get_status()
        right_status = self.right_joycon.get_status()

        if right_status['buttons']['shared']['home']:
            self.require_end = True

        left_stick_value_x = (left_status['analog-sticks']['left']['horizontal'] - self.left_joycon_x_offset) / self.left_joycon_x_scale
        left_stick_value_y = (left_status['analog-sticks']['left']['vertical'] - self.left_joycon_y_offset) / self.left_joycon_y_scale
        right_stick_value_x = (right_status['analog-sticks']['right']['horizontal'] - self.right_joycon_x_offset) / self.right_joycon_x_scale
        right_stick_value_y = (right_status['analog-sticks']['right']['vertical'] - self.right_joycon_y_offset) / self.right_joycon_y_scale
        if left_stick_value_x > 0.7:
            self.curr_Rs[self.selected_eef_name] = self.Rx @ self.curr_Rs[self.selected_eef_name]
        elif left_stick_value_x < -0.7:
            self.curr_Rs[self.selected_eef_name] = self.Rx.T @ self.curr_Rs[self.selected_eef_name]
        if left_stick_value_y > 0.7:
            self.curr_Rs[self.selected_eef_name] = self.Ry @ self.curr_Rs[self.selected_eef_name]
        elif left_stick_value_y < -0.7:
            self.curr_Rs[self.selected_eef_name] = self.Ry.T @ self.curr_Rs[self.selected_eef_name]
        if left_status['buttons']['shared']['l-stick']:
            with self.ori_lock:
                if left_status['buttons']['left']['l']:
                    self.curr_Rs[self.selected_eef_name] = self.Rz @ self.curr_Rs[self.selected_eef_name]
                else:
                    self.curr_Rs[self.selected_eef_name] = self.Rz.T @ self.curr_Rs[self.selected_eef_name]

        self.curr_ts[self.selected_eef_name][1] += right_stick_value_x * 1e-3
        self.curr_ts[self.selected_eef_name][0] += right_stick_value_y * 1e-3
        if right_status['buttons']['shared']['r-stick']:
            with self.T_lock:
                if right_status['buttons']['right']['r']:
                    self.curr_ts[self.selected_eef_name][2] += 0.001
                else:
                    self.curr_ts[self.selected_eef_name][2] -= 0.001

        self.gripper_command[self.selected_eef_name] = np.array([right_status['buttons']['right']['zr']])

        if right_status['buttons']['shared']['plus']:
            self.selected_eef_idx = (self.selected_eef_idx + 1) % self.num_limbs
            self.selected_eef_name = self.follower_limb_names[self.selected_eef_idx]
            print("selected eef: ", self.selected_eef_name)

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
    leader = JoyConController(robot, leader_config, env_config, render_mode='human')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)
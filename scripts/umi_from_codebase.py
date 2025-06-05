import zarr
import matplotlib.pyplot as plt
from configs import BaseConfig
from paprle.utils.config_utils import change_working_directory
change_working_directory()
import numpy as np
from paprle.visualizer.mujoco import MujocoViz
from paprle.follower import Robot
import cv2
import glob
import os
import pandas as pd
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import matplotlib.pyplot as plt
from threading import Thread, Lock
from scipy.spatial.transform import Rotation as R

robot_config, device_config, env_config = BaseConfig().parse()
robot = Robot(robot_config)

lp, lR = None, None
rp, rR = None, None
zero_qpos = np.zeros(robot.num_joints)
def viz_thread():
    viz = MujocoViz(robot)
    viz.init_viewer()

    while True:
        viz.env.plot_T(p=[0, 0, 0], R=np.eye(3), PLOT_AXIS=True, axis_len=0.5, axis_width=0.005,
                       label='Tick:[%d]' % (viz.env.tick))
        if lp is not None and lR is not None:
            viz.env.plot_T(p=lp, R=lR, PLOT_AXIS=True, label='Left', axis_len=0.2)
        if rp is not None and rR is not None:
            viz.env.plot_T(p=rp, R=rR, PLOT_AXIS=True, label='Right', axis_len=0.2)

        viz.set_qpos(zero_qpos)
        viz.render()

thread = Thread(target=viz_thread)
thread.start()
pos_lock = Lock()


zarr_path = '/media/obin/36724ed6-bcb5-4555-abd1-5a15b9d076bd/40clean/sort_toys_organized/result'
with zarr.ZipStore(zarr_path) as zip_store:
    root = zarr.group(zip_store)
    data_group = root['/data']

    episode_ends = root['/meta/episode_ends']
    num_ep = episode_ends.shape[0]

    robot0_demo_end_pose = data_group['robot0_demo_end_pose']
    robot0_demo_start_pose = data_group['robot0_demo_start_pose']
    robot0_eef_pos = data_group['robot0_eef_pos']
    robot0_eef_rot = data_group['robot0_eef_rot_axis_angle']
    robot0_gripper_width = data_group['robot0_gripper_width']

    robot1_demo_end_pose = data_group['robot1_demo_end_pose']
    robot1_demo_start_pose = data_group['robot1_demo_start_pose']
    robot1_eef_pos = data_group['robot1_eef_pos']
    robot1_eef_rot = data_group['robot1_eef_rot_axis_angle']
    robot1_gripper_width = data_group['robot1_gripper_width']

    UMI2PAPRLE = np.array([[0, -1, 0],
                           [0, 0, -1],
                           [1, 0, 0]], dtype=np.float64)

    offset_Rt = np.eye(4, dtype=np.float64)
    offset_Rt[:3, :3] = R.from_rotvec(np.array([0, 0, -np.pi/2])).as_matrix()
    offset_Rt[:3, 3] = np.array([0.8, 0.2, 0])
    for i in range(1, num_ep):
        ep_start = 0 if i == 0 else episode_ends[i-1]
        ep_end = episode_ends[i]
        print(f"Episode {i+1}/{num_ep}: Start at {ep_start}, End at {ep_end}")


        t = ep_start
        while True:
            left_t = t
            right_t = t #+ 30

            lRt = np.eye(4)
            rRt = np.eye(4)

            lRt[:3,3] = robot1_eef_pos[left_t]
            lRt[:3,:3] = R.from_rotvec(robot1_eef_rot[left_t]).as_matrix()
            lRt[:3,:3] = lRt[:3,:3] @ UMI2PAPRLE

            rRt[:3,3] = robot0_eef_pos[right_t]
            rRt[:3,:3] = R.from_rotvec(robot0_eef_rot[right_t]).as_matrix()
            rRt[:3,:3] = rRt[:3,:3] @ UMI2PAPRLE

            # apply offset
            lRt = offset_Rt @ lRt
            rRt = offset_Rt @ rRt


            with pos_lock:
                lp = lRt[:3,3]
                lR = lRt[:3,:3]
                rp = rRt[:3,3]
                rR = rRt[:3,:3]

            im0 = data_group['camera1_rgb'][left_t]
            im1 = data_group['camera0_rgb'][right_t]
            im = np.concatenate((im0, im1), axis=1)
            cv2.imshow('Camera View', im)
            key = cv2.waitKey(0)


            if key == 81:
                t = max(0, t - 1)
            elif key == 83:
                t = min(ep_end - 1, t + 1)
            elif key == ord("q"):
                break
            else:
                t = min(ep_end - 1, t + 1)










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

from paprle.ik.pinocchio import PinocchioIKSolver
r3_ik_solver = PinocchioIKSolver(robot, robot.ik_config['robot3'])
r4_ik_solver = PinocchioIKSolver(robot, robot.ik_config['robot4'])
r3_ik_solver.max_iter = 200
r4_ik_solver.max_iter = 200
r3_ik_solver.reset()
r4_ik_solver.reset()
qpos = r3_ik_solver.get_current_qpos()
qpos[:8] = [0.384, -1.057, 0.0, 0.485, 0.0, 1.125, 0.0, 0.0]
r3_ik_solver.set_current_qpos(qpos)
qpos[:8] = [-0.384, -1.057, 0.0, 0.485, 0.0, 1.125, 0.0, 0.0]
r4_ik_solver.set_current_qpos(qpos)



lp, lR = None, None
rp, rR = None, None
lqpos, rqpos = None, None
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
        if lqpos is not None and rqpos is not None:
            viz.set_qpos(np.concatenate((rqpos, lqpos)))
        else:
            viz.set_qpos(zero_qpos)
        viz.render()

thread = Thread(target=viz_thread)
thread.start()
pos_lock = Lock()



zarr_path = 'data/flip_result'
offset_dict = {
    'bill_collection': [0.4, -0.3, 0.2, 0, 0, -np.pi/2],
    'flip': [0.35, -0.15, 0.12, np.deg2rad(-14.21), np.deg2rad(11.05), -np.pi/2+np.deg2rad(13.68)],
    'sort_toys': [0.8, 0.2, 0, 0, 0, -np.pi/2],
}
if 'bill_collection' in zarr_path: offset = offset_dict['bill_collection']
elif 'flip' in zarr_path: offset = offset_dict['flip']
elif 'sort_toys' in zarr_path: offset = offset_dict['sort_toys']
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
    offset_Rt[:3, 3] = np.array(offset[:3])
    offset_Rt[:3, :3] = R.from_rotvec(np.array(offset[3:])).as_matrix()

    r3_base_Rt = robot.urdf.get_transform('robot3/link1','world')
    r4_base_Rt = robot.urdf.get_transform('robot4/link1','world')
    for i in range(3, num_ep):
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

            im0 = data_group['camera1_rgb'][left_t]
            im1 = data_group['camera0_rgb'][right_t]
            im = np.concatenate((im0, im1), axis=1)

            local_lRt = np.linalg.inv(r4_base_Rt) @ lRt
            local_rRt = np.linalg.inv(r3_base_Rt) @ rRt

            solved_lq = r4_ik_solver.solve(local_lRt[:3, 3], pr.quaternion_from_matrix(local_lRt[:3, :3]))
            solved_rq = r3_ik_solver.solve(local_rRt[:3, 3], pr.quaternion_from_matrix(local_rRt[:3, :3]))

            l_gripper_width = 1 - robot1_gripper_width[left_t] / 0.08
            r_gripper_width = 1 - robot0_gripper_width[right_t] / 0.08

            with pos_lock:
                lp = lRt[:3,3]
                lR = lRt[:3,:3]
                rp = rRt[:3,3]
                rR = rRt[:3,:3]
                lqpos = np.concatenate((solved_lq, l_gripper_width))
                rqpos = np.concatenate((solved_rq, r_gripper_width))



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










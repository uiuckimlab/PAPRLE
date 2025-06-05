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

def kalman_filter_3d(trajectory, dts):
    n = len(trajectory)
    smoothed = []

    x = np.zeros((6, 1))
    x[0:3, 0] = trajectory[0]
    P = np.eye(6)

    H = np.zeros((3, 6))
    H[0, 0] = H[1, 1] = H[2, 2] = 1

    Q_b = np.eye(6) * 0.005
    R = np.eye(3) * 0.01

    for i, z in enumerate(trajectory):
        dt = dts[i]
        # z = z.reshape((3, 1))
        A = np.eye(6)
        for j in range(3):
            A[j, j + 3] = dt

        Q = Q_b * dt
        z = z.reshape((3, 1))
        x = A @ x
        P = A @ P @ A.T + Q

        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(6) - K @ H) @ P

        smoothed.append(x[:3].flatten())

    return np.array(smoothed)

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
            viz.env.plot_T(p=rp, R=rR, PLOT_AXIS=True, label='Right', axis_len=0.2)

        viz.set_qpos(zero_qpos)
        viz.render()

thread = Thread(target=viz_thread)
thread.start()
pos_lock = Lock()

data_dir = '/media/obin/36724ed6-bcb5-4555-abd1-5a15b9d076bd/40clean'
task_list = glob.glob(data_dir + '/*')
demo_dict = {}
for task_name in task_list:
    task_base_name = os.path.basename(task_name)
    demo_dict[task_base_name] =  os.listdir(task_name + '/Left/')




for task_name, demo_list in demo_dict.items():
    #if task_name != 'Flip': continue
    for demo_name in demo_list:
        left_cam_trajectory = os.path.join(data_dir, task_name, 'Left', demo_name, 'camera_trajectory.csv')
        right_cam_trajectory = os.path.join(data_dir, task_name, 'Right', demo_name, 'camera_trajectory.csv')
        left_cam_trajectory = pd.read_csv(left_cam_trajectory)
        right_cam_trajectory = pd.read_csv(right_cam_trajectory)
        left_video_path = os.path.join(data_dir, task_name, 'Left', demo_name, 'raw_video.mp4')
        right_video_path = os.path.join(data_dir, task_name, 'Right', demo_name, 'raw_video.mp4')
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)

        lT, rT = len(left_cam_trajectory), len(right_cam_trajectory)

        l_trajectory = left_cam_trajectory[["x", "y", "z"]].to_numpy()
        l_timestamp = left_cam_trajectory["timestamp"].to_numpy()

        r_trajectory = right_cam_trajectory[["x", "y", "z"]].to_numpy()
        r_timestamp = right_cam_trajectory["timestamp"].to_numpy()

        ldts = np.diff(l_timestamp, prepend=l_timestamp[0])
        rdts = np.diff(r_timestamp, prepend=r_timestamp[0])

        smoothed_l_trajectory = kalman_filter_3d(l_trajectory, ldts)
        smoothed_r_trajectory = kalman_filter_3d(r_trajectory, rdts)

        # I think they are synchronized, so we can use the same T
        T = min(lT, rT)
        left_q = left_cam_trajectory[["q_w", "q_x", "q_y", "q_z"]].to_numpy()
        right_q = right_cam_trajectory[["q_w", "q_x", "q_y", "q_z"]].to_numpy()


        # smoothed_l_trajectory[:,1] = - smoothed_l_trajectory[:,1]
        # smoothed_r_trajectory[:,1] = - smoothed_r_trajectory[:,1]
        # smoothed_l_trajectory[:,2] = - smoothed_l_trajectory[:,2]  + 0.5
        # smoothed_r_trajectory[:,2] = - smoothed_r_trajectory[:,2]  + 0.5
        smoothed_l_trajectory = smoothed_l_trajectory * [1, -1, -1] + np.array([0, 0., 0.5])
        smoothed_r_trajectory = smoothed_r_trajectory * [1, -1, -1] + np.array([0, 0., 0.5])
        UMI2PAPRLE = np.array([[0, -1, 0],
                               [0, 0, -1],
                               [1, 0, 0]], dtype=np.float64)

        # rotate R about x-axis by 40 degrees
        dR = pr.matrix_from_euler([np.deg2rad(5), np.deg2rad(-45), 0], i=0, j=1, k=2, extrinsic=False)
        t = 0
        l_im_array = []
        r_im_array = []
        while True:
            if t >= len(l_im_array):
                ret, lframe = left_cap.read()
                ret, rframe = right_cap.read()
                l_im_array.append(lframe)
                r_im_array.append(rframe)
            else:
                lframe = l_im_array[t]
                rframe = r_im_array[t]


            #if t < 1676: continue

            im = np.concatenate((lframe, rframe), axis=1)

            lq, rq = left_q[t], right_q[t]

            with pos_lock:
                lp = smoothed_l_trajectory[t]
                rp = smoothed_r_trajectory[t]

                lR = pr.matrix_from_quaternion(lq)
                rR = pr.matrix_from_quaternion(rq)
                lR = lR @ UMI2PAPRLE# @ dR
                rR = rR @ UMI2PAPRLE# @ dR

            im = cv2.resize(im, dsize=None, fx=0.25, fy=0.25)
            cv2.imshow("frame", im)
            key = cv2.waitKey(0)
            if key == 81:
                t = max(0, t - 1)
            elif key == 83:
                t = min(T - 1, t + 1)
            elif key == ord("q"):
                break
            else:
                t = min(T - 1, t + 1)
            print("")
    #         render_im = viz.env.grab_image()
    #         plt.imshow(render_im)
    #         plt.title(UMI2PAPRLE)
    #         plt.show()
    #         break
    #     break
    #
    # break





import matplotlib.pyplot as plt
from configs import BaseConfig
from paprle.utils.config_utils import change_working_directory
change_working_directory()
import numpy as np
from paprle.visualizer.mujoco import MujocoViz
from paprle.follower import Robot
import cv2

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


data_dir = '/home/obin/ws_public/src/PAPRLE/data/40clean/Bill_Collection/Left/demo1/camera_trajectory.csv'
import pandas as pd
data = pd.read_csv(data_dir)

video_path = data_dir.replace("camera_trajectory.csv", "raw_video.mp4")
cap = cv2.VideoCapture(video_path)


T = len(data['frame_idx'])

trajectory = data[["x", "y", "z"]].to_numpy()
timestamp = data["timestamp"].to_numpy()
dts = np.diff(timestamp, prepend=timestamp[0])
smoothed_trajectory = kalman_filter_3d(trajectory, dts)
x, y, z = smoothed_trajectory.T
qw = data["q_w"]
qx = data["q_x"]
qy = data["q_y"]
qz = data["q_z"]


robot_config, device_config, env_config = BaseConfig().parse()
robot = Robot(robot_config)
viz = MujocoViz(robot)
viz.init_viewer()

zero_qpos = np.zeros(robot.num_joints)
for t in range(T):
    ret, frame = cap.read()
    xt, yt, zt = smoothed_trajectory[t]
    viz.env.plot_T(p=[xt,yt,zt], R=np.eye(3), PLOT_AXIS=True)
    viz.set_qpos(zero_qpos)
    viz.render()

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("frame", frame[..., ::-1])
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
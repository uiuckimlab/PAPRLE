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
from tqdm import tqdm
robot_config, device_config, env_config = BaseConfig().parse()
robot = Robot(robot_config)

from paprle.ik.pinocchio import PinocchioIKSolver
WRIST_MODE = False
if WRIST_MODE:
    robot.ik_config['robot3'].ee_link = 'robot1/hand_link'
    robot.ik_config['robot4'].ee_link = 'robot1/hand_link'
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
lerr, rerr = None, None

zero_qpos = np.zeros(robot.num_joints)

viz = MujocoViz(robot)
viz.init_viewer(viewer_width=224, viewer_height=224,FONTSCALE_VALUE=50)
viz.update_viewer(
    azimuth=-175,
    distance=2.0,
    elevation=-20,
)


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

    offset = np.array([0.35, -0.15, 0.12, np.deg2rad(-14.21), np.deg2rad(11.05), -np.pi/2+np.deg2rad(13.68)])

    x_ziggle = np.linspace(-0.1, 0.1, 20)
    y_ziggle = np.linspace(-0.1, 0.1, 20)
    z_ziggle = np.linspace(-0.1, 0.1, 20)

    ziggle = np.meshgrid(x_ziggle, y_ziggle, z_ziggle)
    ziggle = np.stack(ziggle, axis=-1).reshape(-1, 3)

    inds = np.arange(ziggle.shape[0])
    np.random.shuffle(inds)
    RENDER = True
    for zig_idx in tqdm(inds):
        zig = ziggle[zig_idx]


        new_offset = offset.copy()
        new_offset[:3] += zig

        offset_Rt = np.eye(4, dtype=np.float64)
        offset_Rt[:3, 3] = np.array(new_offset[:3])
        offset_Rt[:3, :3] = R.from_rotvec(np.array(new_offset[3:])).as_matrix()

        r3_base_Rt = robot.urdf.get_transform('robot3/link1','world')
        r4_base_Rt = robot.urdf.get_transform('robot4/link1','world')

        zig_dir = os.path.join('data/ziggle/', f"{zig[0]:.03f}_{zig[1]:.03f}_{zig[2]:.03f}")
        if WRIST_MODE:
            zig_dir = zig_dir.replace('ziggle', 'ziggle_wrist')
        os.makedirs(zig_dir, exist_ok=True)

        total_errs = []
        for i in range(0, num_ep):
            # result_dir = glob.glob(os.path.join(zig_dir, f"ep_{i:02d}_results") + '_*')
            # if len(result_dir) == 0: continue
            # result_dir = result_dir[0]
            # if 'c0' not in result_dir: continue


            ep_start = 0 if i == 0 else episode_ends[i-1]
            ep_end = episode_ends[i]
            print(f"Episode {i+1}/{num_ep}: Start at {ep_start}, End at {ep_end}")

            if RENDER:
                writer = cv2.VideoWriter(os.path.join(zig_dir, f"ep_{i+1:02d}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (672, 224))

            t = ep_start
            prev_lqpos = None
            prev_rqpos = None
            errors = []
            large_qchanges = []

            qpos = r3_ik_solver.get_current_qpos()
            qpos[:8] = [0.384, -1.057, 0.0, 0.485, 0.0, 1.125, 0.0, 0.0]
            r3_ik_solver.set_current_qpos(qpos)
            qpos[:8] = [-0.384, -1.057, 0.0, 0.485, 0.0, 1.125, 0.0, 0.0]
            r4_ik_solver.set_current_qpos(qpos)


            for t in range(ep_start, ep_end):
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
                #solved_lq = np.clip(solved_lq, robot.joint_limits[:7, 0], robot.joint_limits[:7, 1])
                #solved_rq = np.clip(solved_rq, robot.joint_limits[:7, 0], robot.joint_limits[:7, 1])

                l_gripper_width = 1 - robot1_gripper_width[left_t] / 0.08
                r_gripper_width = 1 - robot0_gripper_width[right_t] / 0.08

                lp = lRt[:3,3]
                lR = lRt[:3,:3]
                rp = rRt[:3,3]
                rR = rRt[:3,:3]
                lqpos = np.concatenate((solved_lq, l_gripper_width))
                rqpos = np.concatenate((solved_rq, r_gripper_width))


                viz.set_qpos(np.concatenate((rqpos, lqpos)))

                # v
                if WRIST_MODE:
                    r3_p = viz.env.get_p_body('robot3/hand_link')
                    r4_p = viz.env.get_p_body('robot4/hand_link')
                else:
                    r3_p = viz.env.get_p_body('robot3/end_effector_link')
                    r4_p = viz.env.get_p_body('robot4/end_effector_link')
                target_r3p = rRt[:3, 3]
                target_r4p = lRt[:3, 3]

                r3_err = np.linalg.norm(r3_p - target_r3p)
                r4_err = np.linalg.norm(r4_p - target_r4p)
                if RENDER:
                    viz.env.plot_T(p=[0, 0, 0], R=np.eye(3), PLOT_AXIS=True, axis_len=0.5, axis_width=0.005)
                    viz.env.plot_T(p=lp, R=lR, PLOT_AXIS=True, axis_len=0.2)
                    viz.env.plot_T(p=rp, R=rR, PLOT_AXIS=True, axis_len=0.2)
                    viz.env.plot_line_fr2to(target_r3p + [0, 0, 0.3], r3_p, label=f"r3:{r3_err:.3f}")
                    viz.env.plot_line_fr2to(target_r4p + [0, 0, 0.3], r4_p, label=f"r4:{r4_err:.3f}")
                    viz.env.plot_contact_info(h_arrow=0.3, rgba_arrow=[1, 0, 0, 1], PRINT_CONTACT_BODY=True)  # contact

                errors.append((r3_err, r4_err))
                if prev_lqpos is not None:
                    lq_diff = np.linalg.norm(lqpos - prev_lqpos)
                    rq_diff = np.linalg.norm(rqpos - prev_rqpos)
                    large_qchanges.append((lq_diff > 0.1, rq_diff > 0.1))
                prev_lqpos = lqpos
                prev_rqpos = rqpos

                if RENDER:
                    viz.render()
                    viz_im = viz.env.grab_image()
                    viz_im = cv2.resize(viz_im, (int(viz_im.shape[1] / viz_im.shape[0] * im.shape[0]), im.shape[0]))
                    total_im = np.concatenate((im, viz_im),1)
                    writer.write(total_im[..., ::-1])  # BGR to RGB

            large_qchanges = np.array(large_qchanges)
            large_qchange_l = np.sum(large_qchanges[:, 0])
            large_qchange_r = np.sum(large_qchanges[:, 1])
            errors = np.array(errors)
            l_err = np.mean(errors[:, 0])
            r_err = np.mean(errors[:, 1])
            results = {
                'l_large_qchange': large_qchanges[:,0],
                'r_large_qchange': large_qchanges[:,1],
                'l_err': errors[1:,0],
                'r_err': errors[1:,1],
            }
            with open(os.path.join(zig_dir, f"ep_{i:02d}_results_lq{large_qchange_l}_rq{large_qchange_r}_lerr{l_err:.03f}_rerr{r_err:.03f}.csv"), 'w') as f:
                pd.DataFrame(results).to_csv(f, index=False)
            if RENDER:
                writer.release()
            total_errs.append((l_err, r_err, large_qchange_l, large_qchange_r))
            print(f"Episode {i+1}/{num_ep} done. l_err: {l_err:.03f}, r_err: {r_err:.03f}, large_qchange_l: {large_qchange_l}, large_qchange_r: {large_qchange_r}")

        if len(total_errs) == 0: continue
        total_errors = np.mean(np.array(total_errs), 0)
        total_error_str = f"l_err_{total_errors[0]:.03f}_r_err_{total_errors[1]:.03f}_lqchange_{total_errors[2]}_rqchange_{total_errors[3]}"
        #os.rename(zig_dir, zig_dir + '_' + total_error_str)
        print(f"Total errors for ziggle {zig}: {total_error_str}")
        print("==============================================")







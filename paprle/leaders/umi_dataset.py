import zarr
import numpy as np
import matplotlib.pyplot as plt
from configs import BaseConfig
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from threading import Thread, Lock
import cv2

#[0, 1, 2, 3, 4, 6, 8, 10, 12, 15]
gripper_close = {
    0: [315 , 1268],
    1: [1732-1507, 3104-1507],
    2: [3522-3266, 4555-3266], # 3266
    3: [4973-4743, 5967-4743], #4743
    4: [6400-6214, 7500-6214], # 6214
    5: [9532-9281, 10520-9281], # 9281
    6: [12524-12196, 13451-12196], # 12196
    7: [15442-15225, 16516-15225], # 15225
    8: [18439-18172, 19478-18172], # 18172
    9: [22501-22315, 23545-22315] # 22315
}

from paprle.ik.pinocchio import PinocchioIKSolver

class UMIDataset:
    def __init__(self, robot, leader_config, env_config, render_mode='human', verbose=False, *args, **kwargs):
        self.is_ready = False
        self.require_end = False
        self.shutdown = False
        self.limb_names = leader_config.limb_names
        self.follower_robot = robot
        self.leader_config = leader_config

        file_path = leader_config.file_path
        self.data_dict = {}
        self.ik_solvers = {} # for better initialization pose
        with zarr.ZipStore(file_path) as zip_store:
            root = zarr.group(zip_store)
            data_group = root['/data']

            self.episode_ends = np.array(root['/meta/episode_ends'])
            self.num_ep = self.episode_ends.shape[0]

            for limb_id, limb_name in enumerate(self.limb_names):
                follower_limb_name = leader_config.limb_mapping[limb_name]
                follower_eef_name = robot.robot_config.end_effector_link[follower_limb_name]
                self.data_dict[limb_name] = {
                    'demo_end_pose': data_group[f'robot{limb_id}_demo_end_pose'][:],
                    'demo_start_pose': data_group[f'robot{limb_id}_demo_start_pose'][:],
                    'eef_pos': data_group[f'robot{limb_id}_eef_pos'][:],
                    'eef_rot': data_group[f'robot{limb_id}_eef_rot_axis_angle'][:],
                    'gripper_width': data_group[f'robot{limb_id}_gripper_width'][:],
                    'base_Rt': robot.urdf.get_transform(robot.ik_config[follower_limb_name].base_link, 'world')
                }
                #if render_mode:
                #    print("Loading RGB data for limb:", limb_name)
                #    self.data_dict[limb_name]['camera'] = data_group[f'camera{limb_id}_rgb'][:]

                self.ik_solvers[limb_name] = PinocchioIKSolver(robot, robot.ik_config[follower_limb_name])
                self.ik_solvers[limb_name].max_iter = 200
                self.ik_solvers[limb_name].reset()
                qpos = self.ik_solvers[limb_name].get_current_qpos()
                inds = robot.ctrl_joint_idx_mapping[follower_limb_name]
                qpos[:len(inds)] = robot.init_qpos[inds]
                self.ik_solvers[limb_name].set_current_qpos(qpos)

        self.UMI2PAPRLE = np.array([[0, -1, 0],
                                           [0, 0, -1],
                                           [1, 0, 0]], dtype=np.float64)
        self.offset = np.array([0.439,-0.155,0.073,-0.248,0.1928,-1.3320], dtype=np.float64)
        self.offset_Rt = np.eye(4, dtype=np.float64)
        self.offset_Rt[:3, :3] = R.from_rotvec(self.offset[3:]).as_matrix()
        self.offset_Rt[:3, 3] = self.offset[:3]

        self.curr_ep_id = 0
        self.curr_timestep = self.curr_ep_start

        self.render_mode = render_mode
        if self.render_mode:
            self.viz_lock = Lock()
            self.viz_thread = Thread(target=self.render)
            self.viz_thread.start()


        return

    @property
    def curr_timestep_l(self):
        return min(self.curr_timestep+13, self.curr_ep_end)

    def render(self):
        file_path = self.leader_config.file_path
        with zarr.ZipStore(file_path) as zip_store:
            root = zarr.group(zip_store)
            data_group = root['/data']

            while not self.shutdown:

                ims = []
                for limb_id, limb_name in enumerate(self.limb_names):
                    if limb_id == 1: t = self.curr_timestep
                    else: t = self.curr_timestep_l
                    im = data_group[f'camera{limb_id}_rgb'][t]
                    ims.append(im)
                ims = np.concatenate(ims[::-1], axis=1)
                ims = cv2.resize(ims, dsize=None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
                with self.viz_lock:
                    cv2.imshow("UMI Dataset Visualization", ims[...,[2,1,0]])
                    key = cv2.waitKey(1)

        return

    @property
    def curr_ep_start(self):
        if self.curr_ep_id == 0: return 0
        return self.episode_ends[self.curr_ep_id - 1]

    @property
    def curr_ep_end(self):
        return self.episode_ends[self.curr_ep_id]

    def reset(self, ):
        self.curr_timestep = self.curr_ep_start
        for limb_id, limb_name in enumerate(self.limb_names):
            follower_limb_name = leader_config.limb_mapping[limb_name]
            ik_solver = self.ik_solvers[limb_name]
            ik_solver.reset()
            qpos = ik_solver.get_current_qpos()
            inds = self.follower_robot.ctrl_joint_idx_mapping[follower_limb_name]
            qpos[:len(inds)] = self.follower_robot.init_qpos[inds]
            ik_solver.set_current_qpos(qpos)

        return

    def get_Rt(self, limb_name, t):
        Rt = np.eye(4, dtype=np.float64)
        p = self.data_dict[limb_name]['eef_pos'][t]
        r = self.data_dict[limb_name]['eef_rot'][t]
        Rt[:3, 3] = p
        Rt[:3, :3] = R.from_rotvec(r).as_matrix()
        Rt[:3, :3] = Rt[:3, :3] @ self.UMI2PAPRLE
        Rt = self.offset_Rt @ Rt
        if limb_name == 'right':
           Rt[:3, 3] += np.array([0.1, -0.1, 0.0])
        if limb_name == 'left':
           Rt[:3, 3] += np.array([0.0, 0.0, -0.03])
        return Rt

    def launch_init(self, init_env_qpos):

        user_input = input("Enter episode ID to start (0 to {}): ".format(self.num_ep - 1))
        try:
            self.curr_ep_id = int(user_input)
            if self.curr_ep_id < 0 or self.curr_ep_id >= self.num_ep:
                raise ValueError("Episode ID out of range.")
        except ValueError as e:
            print(f"Invalid input: {e}. Defaulting to episode 0.")
            self.curr_ep_id = 0

        self.curr_timestep = self.curr_ep_start

        # get first pose, solve ik
        total_qpos = []
        self.init_eef_poses = []
        self.last_command = [{}, {}]
        for limb_id, limb_name in enumerate(self.limb_names):
            follower_limb_name = self.leader_config.limb_mapping[limb_name]
            if limb_id == 1:
                t = self.curr_timestep
            else:
                t = self.curr_timestep_l

            Rt = self.get_Rt(limb_name,t)
            ik_solver = self.ik_solvers[limb_name]
            ik_solver.max_iter = 200
            qpos = ik_solver.get_current_qpos()
            inds = self.follower_robot.ctrl_joint_idx_mapping[follower_limb_name]
            qpos[:len(inds)] = self.follower_robot.init_qpos[inds]
            ik_solver.set_current_qpos(qpos)

            local_Rt = np.linalg.inv(self.data_dict[limb_name]['base_Rt']) @ Rt
            solved_q = ik_solver.solve(local_Rt[:3,3], pr.quaternion_from_matrix(local_Rt[:3, :3]))

            gripper_width = 1 - self.data_dict[limb_name]['gripper_width'][t] / 0.08
            total_qpos.append(np.concatenate((solved_q, gripper_width)))
            ee_pose = ik_solver.compute_ee_pose(ik_solver.get_current_qpos())
            self.init_eef_poses.append(pt.transform_from_pq(ee_pose))
            self.last_command[0][follower_limb_name] = np.eye(4)
            self.last_command[1][follower_limb_name] = np.array(gripper_width)

        total_qpos = np.concatenate(total_qpos)
        self.last_qpos = total_qpos
        self.initialized_with_qpos = False
        self.is_ready = True
        self.require_end = False
        print("[UMIDataset] Initialized with first pose:", total_qpos)
        return

    def close_init(self):
        return

    def initialize(self):
        return

    def get_status(self):
        if not self.initialized_with_qpos:
            self.initialized_with_qpos = True
            return {'command_type': 'joint_pos', 'command': self.last_qpos.copy()}
        else:
            local_poses, hand_poses = {}, {}
            for limb_id, limb_name in enumerate(self.limb_names):

                if limb_id == 0:
                    t = self.curr_timestep
                else:
                    t = self.curr_timestep_l
                Rt = self.get_Rt(limb_name,t)
                Rt = np.linalg.inv(self.data_dict[limb_name]['base_Rt']) @ Rt
                init_Rt = self.init_eef_poses[limb_id]

                new_pos = init_Rt[:3, :3].T @ Rt[:3, 3] - init_Rt[:3, :3].T @ init_Rt[:3, 3]
                new_R = init_Rt[:3, :3].T @ Rt[:3, :3]
                new_Rt = pt.transform_from(R=new_R, p=new_pos)
                follower_limb_name = self.leader_config.limb_mapping[limb_name]
                local_poses[follower_limb_name] = new_Rt

                gripper_width = 1 - self.data_dict[limb_name]['gripper_width'][t][0] / 0.08
                if (t - self.curr_ep_start) < gripper_close[self.curr_ep_id][0]:
                    pass
                elif (t - self.curr_ep_start) < gripper_close[self.curr_ep_id][1]:
                    gripper_width = max(gripper_width, self.last_command[1][follower_limb_name][0])
                else:
                    gripper_width = min(gripper_width, self.last_command[1][follower_limb_name][0])

                hand_poses[follower_limb_name] = np.array([gripper_width])

            self.curr_timestep += 1
            if self.curr_timestep_l >= self.curr_ep_end:
                self.require_end = True
                self.is_ready = False
            self.last_command = (local_poses, hand_poses)
            return {'command_type': 'delta_eef_pose', 'command': (local_poses, hand_poses)}
    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def close(self):
        self.shutdown = True
        if self.render_mode:
            self.viz_thread.join()
        return

if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)
    leader = UMIDataset(robot, leader_config, env_config, render_mode='')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)
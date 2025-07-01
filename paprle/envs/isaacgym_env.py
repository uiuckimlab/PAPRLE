import isaacgym
import torch
import os
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from pathlib import Path
import numpy as np
from paprle.envs.isaacgym_env_utils.isaac_utils import default_sim_params, refresh_tensors, setup_viewer_camera
from typing import Dict
from pytransform3d.rotations import matrix_from_quaternion
from yourdfpy.urdf import URDF


class IsaacGymEnv:
    def __init__(self, robot, device_config, env_config, verbose=False, render_mode='', **kwargs):

        self.robot = robot
        self.robot_config = robot.robot_config
        self.urdf_file = robot.urdf_file
        self.gym = gymapi.acquire_gym()
        self.device = 'cpu'
        self.initialized = True

        self.sim_params = default_sim_params(use_gpu=True if self.device == 'cuda:0' else False)
        self.sim_params.substeps = 2
        self.sim_params.dt = 1.0 / 125.0
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.use_gpu = False
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        self.num_envs = 1

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        self.create_env()

        cam_pos = gymapi.Vec3(*self.robot_config.viewer_args.isaacgym.cam_pos)
        cam_target = gymapi.Vec3(*self.robot_config.viewer_args.isaacgym.cam_target)
        self.gym.viewer_camera_look_at(self.viewer,self.env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)
        self.initialize_tensors()

        self.render_mode = render_mode
        self.vis_info = None


    def create_env(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        asset_path = os.path.join(*self.urdf_file.split("/")[1:])
        asset_root = self.robot_config.asset_cfg.asset_dir
        asset_name = self.robot_config.name

        # Load asset with default control type of position for all joints
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = getattr(self.robot_config.asset_cfg, 'flip_visual_attachments', False)
        if 'kitchen' in self.urdf_file:
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.resolution = 1000000
            asset_options.vhacd_params.max_convex_hulls = 60
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.convex_decomposition_from_submeshes = True
        asset_options.disable_gravity = False
        asset = self.gym.load_asset(self.sim, asset_root, asset_path, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(asset)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 2)
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(0,0,1.0)
        initial_pose.r = gymapi.Quat(0, 0, 0, 1.)
        self.actor = self.gym.create_actor(self.env, asset, initial_pose, asset_name, 0, 0)

        self.dof_prop = self.gym.get_actor_dof_properties(self.env, self.actor)
        self.dof_prop["driveMode"] = (gymapi.DOF_MODE_POS,) * self.num_dof
        self.dof_prop["stiffness"] = (1000,) * self.num_dof
        self.dof_prop["damping"] = (300.0,) * self.num_dof
        self.gym.set_actor_dof_properties(self.env, self.actor, self.dof_prop)
        self.joint_limits = np.stack([self.dof_prop["lower"], self.dof_prop["upper"]], axis=1)
        self.joint_names = self.gym.get_actor_dof_names(self.env, self.actor)

        self.isaacgym_control_names = self.joint_names
        self.input_control_names = self.robot_config.ctrl_joint_names
        self.idx_mappings = [self.isaacgym_control_names.index(name) for name in self.input_control_names]

        # Isaacgym does not support mimic joints, so we need to mimic them manually
        self.urdf = URDF.load(self.urdf_file)
        self.mimic_joints_info = []
        for joint_name in self.joint_names:
            if self.urdf.joint_map[joint_name].mimic:
                this_idx = self.isaacgym_control_names.index(joint_name)
                mimic_idx = self.isaacgym_control_names.index(self.urdf.joint_map[joint_name].mimic.joint)
                mimic_info = [this_idx, mimic_idx, self.urdf.joint_map[joint_name].mimic.multiplier, self.urdf.joint_map[joint_name].mimic.offset]
                self.mimic_joints_info.append(mimic_info)
        self.mimic_joints_info = np.array(self.mimic_joints_info)

        all_qpos = np.zeros(self.num_dof)
        #all_qpos[self.idx_mappings] = self.init_qpos
        if len(self.mimic_joints_info):
            all_qpos[self.mimic_joints_info[:, 0].astype(np.int32)] = all_qpos[self.mimic_joints_info[:, 1].astype(np.int32)] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]
        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        dof_states['pos'] = all_qpos
        self.gym.set_actor_dof_states(self.env, self.actor, dof_states, gymapi.STATE_ALL)

        return

    def reset(self, ):
        all_qpos = np.zeros(self.num_dof)
        all_qpos[self.idx_mappings] = np.array(self.robot.init_qpos)
        if len(self.mimic_joints_info):
            all_qpos[self.mimic_joints_info[:, 0].astype(np.int32)] = all_qpos[self.mimic_joints_info[:, 1].astype(np.int32)] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]
        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        dof_states['pos'] = all_qpos
        self.gym.set_actor_dof_states(self.env, self.actor, dof_states, gymapi.STATE_ALL)

        curr_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        return curr_states['pos'][self.idx_mappings]


    def initialize(self, initialize_qpos=None):
        self.init_qpos  = initialize_qpos
        all_qpos = np.zeros(self.num_dof)
        all_qpos[self.idx_mappings] = self.init_qpos
        if len(self.mimic_joints_info):
            all_qpos[self.mimic_joints_info[:, 0].astype(np.int32)] = all_qpos[self.mimic_joints_info[:, 1].astype(np.int32)] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]
        dof_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        dof_states['pos'] = all_qpos
        self.gym.set_actor_dof_states(self.env, self.actor, dof_states, gymapi.STATE_ALL)
        curr_states = self.gym.get_actor_dof_states(self.env, self.actor, gymapi.STATE_ALL)
        return curr_states['pos'][self.idx_mappings]


    def initialize_tensors(self):

        refresh_tensors(self.gym, self.sim)
        # get actor root state tensor
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(_root_states).view(self.num_envs, -1, 13)
        self.root_state = root_states

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        self.gym.sync_frame_time(self.sim)


        return

    def render(self, sync_frame_time=True):

        # update viewer
        #if self.args.follow:
        #    self.move_camera()
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, False)
        if sync_frame_time:
            self.gym.sync_frame_time(self.sim)

        return

    def simulate(self):
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)


    def step(self, qpos=None, sync_frame_time=False):
        all_qpos = np.zeros(self.num_dof)
        if qpos is not None:
            all_qpos[self.idx_mappings] = qpos
            if len(self.mimic_joints_info):
                all_qpos[self.mimic_joints_info[:,0].astype(np.int32)] = all_qpos[self.mimic_joints_info[:,1].astype(np.int32)] * self.mimic_joints_info[:,2] + self.mimic_joints_info[:,3]
        self.gym.set_actor_dof_position_targets(self.env, self.actor, all_qpos.astype(np.float32))
        #TODO: If you are considering RL, you might want to use the following line for proper GPU usage
        # qpos_tensor = gymtorch.unwrap_tensor(all_qpos) # assuming all_qpos is a gpu torch tensor
        # self.gym.set_dof_position_target_tensor(self.sim, qpos_tensor)

        self.simulate()
        if self.render_mode:
            self.render(sync_frame_time=sync_frame_time)
        self.curr_qpos = all_qpos

    def get_current_qpos(self):
        return np.array(self.curr_qpos)

    def get_curr_viewer_camera_setting(self):
        transform = self.gym.get_viewer_camera_transform(self.viewer, self.env)

        curr_pos = transform.p
        target_pos = transform.transform_vector(gymapi.Vec3(1, 0, 0)) + curr_pos

        self.gym.viewer_camera_look_at(self.viewer, self.env, curr_pos, target_pos)

        print(f"cam_pos: [{curr_pos.x:.2f}, {curr_pos.y:.2f}, {curr_pos.z:.2f}]")
        print(f"cam_target: [{target_pos.x:.2f}, {target_pos.y:.2f}, {target_pos.z:.2f}]")

        for i in range(1000):
            self.render()

        curr_pos = transform.p
        target_pos = transform.transform_vector(gymapi.Vec3(0, 1, 0)) + curr_pos

        self.gym.viewer_camera_look_at(self.viewer, self.env, curr_pos, target_pos)

        print(f"cam_pos: [{curr_pos.x:.2f}, {curr_pos.y:.2f}, {curr_pos.z:.2f}]")
        print(f"cam_target: [{target_pos.x:.2f}, {target_pos.y:.2f}, {target_pos.z:.2f}]")

        for i in range(1000):
            self.render()

        curr_pos = transform.p
        target_pos = transform.transform_vector(gymapi.Vec3(0, 0, 1)) + curr_pos

        self.gym.viewer_camera_look_at(self.viewer, self.env, curr_pos, target_pos)

        print(f"cam_pos: [{curr_pos.x:.2f}, {curr_pos.y:.2f}, {curr_pos.z:.2f}]")
        print(f"cam_target: [{target_pos.x:.2f}, {target_pos.y:.2f}, {target_pos.z:.2f}]")

        for i in range(1000):
            self.render()

        return

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return

if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)

    env = IsaacGymEnv(robot, leader_config, env_config, render_mode='human')
    while True:
        env.step()



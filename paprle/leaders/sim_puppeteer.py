import numpy as np
import tkinter as tk
import copy

from omegaconf import OmegaConf

from paprle.envs.mujoco_env_utils.util import MultiSliderClass
from threading import Thread
from pytransform3d import transformations as pt
from paprle.follower import Robot
from paprle.utils.config_utils import add_info_robot_config

import pinocchio as pin
import time

class SimPuppeteer:
    def __init__(self, follower_robot, leader_config, env_config, render_mode='none', verbose=False, *args, **kwargs):
        self.follower_robot = follower_robot
        leader_config.robot_cfg = add_info_robot_config(leader_config)
        self.leader_robot = Robot(leader_config)
        self.leader_config = leader_config
        self.env_config = env_config
        self.is_ready = False
        self.require_end = False
        self.shutdown = False

        self.joint_names = self.leader_robot.joint_names
        self.sliders = MultiSliderClass(
            n_slider = len(self.leader_robot.joint_names),
            title=f'[LEADER] {leader_config.name} - {self.leader_robot.name}',
            window_width=600,
            window_height=800,
            x_offset=50,
            y_offset=100,
            slider_width=300,
            label_texts=self.leader_robot.joint_names,
            slider_mins=self.leader_robot.joint_limits[:, 0],
            slider_maxs=self.leader_robot.joint_limits[:, 1],
            slider_vals=self.leader_robot.init_qpos,
            resolution=0.001,
            VERBOSE=verbose,
        )
        # add reset button in the slider
        self.reset_button = tk.Button(self.sliders.gui, text="RESET", command=self.reset)
        self.reset_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.render_mode = render_mode
        render_base = leader_config.render_base
        if render_mode != 'none' and render_base == 'trimesh':
            self.render_thread = Thread(target=self.__render_trimesh, args=())
            self.render_thread.start()
        elif render_mode != 'none' and render_base == 'mujoco':
            self.render_thread = Thread(target=self.__render_mujoco, args=())
            self.render_thread.start()

        self.last_qpos = self.leader_robot.init_qpos
        self.output_type = leader_config.output_type

        self.direct_joint_mapping = leader_config.output_type == 'joint_pos'
        if not self.direct_joint_mapping:
            self.motion_mapping_method = leader_config.motion_mapping
            self.motion_scale = leader_config.motion_scale
            # Load Leader Pin Model
            self.pin_model, self.pin_data, self.pin_model_joint_names, self.eef_frame_ids = self.load_pin_model(
                leader_config.robot_cfg.asset_cfg.urdf_path,
                leader_config.robot_cfg.asset_cfg.asset_dir,
                leader_config.robot_cfg.end_effector_link
            )
            self.idx_pin2state = [[id, self.leader_robot.joint_names.index(name)] for id, name in enumerate(self.pin_model_joint_names) if name in self.leader_robot.joint_names]
            self.idx_pin2state = np.array(self.idx_pin2state)

            self.hand_mapping = {}
            for leader_limb_name, leader_hand_joint_idxs in self.leader_robot.ctrl_hand_joint_idx_mapping.items():
                if len(leader_hand_joint_idxs) > 0:
                    follower_limb_name = self.leader_config.limb_mapping[leader_limb_name]
                    follower_hand_joint_idxs = self.follower_robot.ctrl_hand_joint_idx_mapping[follower_limb_name]
                    if len(follower_hand_joint_idxs) > 0:
                        self.hand_mapping[leader_limb_name] = leader_hand_joint_idxs
                    else:
                        self.hand_mapping[leader_limb_name] = [] # does not output hand joint if follower does not have hand joint

            if self.motion_mapping_method == 'follower_reprojection':
                # Leader Δq → Virtual Follower Δq → Virtual Follower ΔEEF → Target Follower ΔEEF
                # Load virtual follower model
                virtual_follower_config_file = f'configs/follower/{leader_config.direct_mapping_available_robots[0]}.yaml'
                self.virtual_follower_config = OmegaConf.load(virtual_follower_config_file)
                self.virtual_follower_config.robot_cfg = add_info_robot_config(self.virtual_follower_config.robot_cfg)
                urdf_path = self.virtual_follower_config.robot_cfg.asset_cfg.urdf_path
                asset_dir = self.virtual_follower_config.robot_cfg.asset_cfg.asset_dir
                self.vfollower_model, self.vfollower_data, self.vfollower_joint_names, self.vfollower_eef_frame_ids = self.load_pin_model(
                    urdf_path, asset_dir, self.virtual_follower_config.robot_cfg.end_effector_link
                )
                self.idx_vfpin2state = []
                vfollower_ctrl_joint_names = self.virtual_follower_config.robot_cfg.ctrl_joint_names
                for idx, name in enumerate(self.vfollower_joint_names):
                    if name in vfollower_ctrl_joint_names:
                        ctrl_id = vfollower_ctrl_joint_names.index(name)
                        self.idx_vfpin2state.append([idx, ctrl_id])
                self.idx_vfpin2state = np.array(self.idx_vfpin2state)

                # make mapping leader_robot.joint_names -> vfollower_model_joint_names
            elif  self.motion_mapping_method == 'leader_reprojection':
                # Leader Δq → Leader ΔEEF → Virtual Target-Leader ΔEEF → Virtual Target-Leader Δq → Target Follower Δq
                # Load target leader model
                virtual_leader_config_file = f'configs/leader/sim_puppeteer_{follower_robot.name}.yaml'
                self.virtual_leader_config = OmegaConf.load(virtual_leader_config_file)
                urdf_path = self.virtual_leader_config.robot_cfg.asset_cfg.urdf_path
                asset_dir = self.virtual_leader_config.robot_cfg.asset_cfg.asset_dir
                self.vleader_model, self.vleader_data, self.vleader_joint_names, self.vleader_eef_frame_ids = self.load_pin_model(
                    urdf_path, asset_dir, self.virtual_leader_config.robot_cfg.end_effector_link
                )

        return
    def load_pin_model(self, urdf_path, asset_dir, end_effector_link_dict, ):
        pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_dirs=[asset_dir])
        pin_data = pin_model.createData()
        pin_model_joint_names = [name for name in pin_model.names]
        if 'universe' in pin_model_joint_names:
            pin_model_joint_names.remove('universe')
        frame_mapping = {}
        for i, frame in enumerate(pin_model.frames):
            frame_mapping[frame.name] = i
        end_effector_frame_ids = [frame_mapping[eef_name] for limb_name, eef_name in end_effector_link_dict.items()]
        return pin_model, pin_data, pin_model_joint_names, end_effector_frame_ids

    def __render_trimesh(self):
        from yourdfpy.urdf import URDF
        self.leader_model = URDF.load(self.leader_config.asset_cfg.urdf_path)

        def callback(scene,  **kwargs ):
            self.leader_model.update_cfg(self.last_qpos)
            # # To get current camera transform
            # print(self.leader_model._scene.camera_transform)
            # print(pt.pq_from_transform(self.leader_model._scene.camera_transform))

        if 'trimesh' in self.leader_robot.robot_config.viewer_args:
            pq = self.leader_robot.robot_config.viewer_args.trimesh.pq
            self.leader_model._scene.camera_transform = pt.transform_from_pq(pq)
            from trimesh.creation import axis as create_axis
            for limb_name, eef_name in self.leader_robot.eef_names.items():
                axis = create_axis(origin_size=self.leader_model._scene.scale/30)
                self.leader_model._scene.add_geometry(axis, 'axis', parent_node_name=eef_name)
        self.leader_model._scene.show(
            callback=callback,
            flags={'grid': True}
        )

    def __render_mujoco(self):
        from paprle.visualizer.mujoco import MujocoViz
        self.leader_model = MujocoViz(self.leader_robot)
        self.leader_model.init_viewer(
            viewer_title=self.leader_robot.robot_config.name, viewer_width=1200, viewer_height=800,
            viewer_hide_menus=True,
        )
        if 'mujoco' in self.leader_config.viewer_args:
            viewer_args = self.leader_config.viewer_args.mujoco
            self.leader_model.update_viewer(
                azimuth=viewer_args.azimuth, distance=viewer_args.distance,
                elevation=viewer_args.elevation, lookat=viewer_args.lookat,
                VIS_TRANSPARENT=False,
            )
        while not self.shutdown:
            self.leader_model.set_qpos(self.last_qpos)
            for limb_name, eef_name in self.leader_robot.eef_names.items():
                p, R = self.leader_model.env.get_pR_body(eef_name)
                self.leader_model.env.plot_T(p, R, PLOT_AXIS=True, PLOT_SPHERE=True, sphere_r=0.02, axis_len=0.12, axis_width=0.005)
            self.leader_model.render()
            time.sleep(0.01)



    def reset(self, ):
        self.require_end = True
        return

    def launch_init(self, init_env_qpos):
        self.initialize(init_env_qpos)

        self.is_ready = True
        self.require_end = False
        return

    def initialize(self, init_env_qpos):
        if not self.direct_joint_mapping:
            init_env_qpos = self.leader_robot.init_qpos.copy()
            if self.motion_mapping_method in ['direct_scaling', 'leader_reprojection']:
                pin_qpos = pin.neutral(self.pin_model)
                pin_qpos[self.idx_pin2state[:, 0]] = init_env_qpos[self.idx_pin2state[:, 1]]
                self.init_eef_poses = self.get_eef_poses(pin_qpos, self.pin_model, self.pin_data, self.eef_frame_ids)
            elif self.motion_mapping_method == 'follower_reprojection':
                pin_qpos = pin.neutral(self.vfollower_model)
                pin_qpos[self.idx_vfpin2state[:, 0]] = init_env_qpos[self.idx_vfpin2state[:, 1]]
                self.init_eef_poses = self.get_eef_poses(pin_qpos, self.vfollower_model, self.vfollower_data, self.vfollower_eef_frame_ids)
            self.init_ts, self.init_Rs = [], []
            for Rt in self.init_eef_poses:
                self.init_ts.append(Rt[:3])
                self.init_Rs.append(Rt[:3, :3])
        elif init_env_qpos is None:
            init_env_qpos = self.leader_robot.init_qpos.copy()
        self.sliders.set_slider_values(init_env_qpos)
        self.last_qpos = init_env_qpos
        return

    def close_init(self):
        return

    def get_status(self):
        self.sliders.update()
        q_from_sliders = self.sliders.get_slider_values()
        self.last_qpos = q_from_sliders
        if self.output_type == 'delta_eef_pose':
            if self.motion_mapping_method == 'direct_scaling':
                pin_qpos = pin.neutral(self.pin_model)
                pin_qpos[self.idx_pin2state[:, 0]] = q_from_sliders[self.idx_pin2state[:, 1]]
                new_eef_poses = self.get_eef_poses(pin_qpos, self.pin_model, self.pin_data, self.eef_frame_ids)
            elif self.motion_mapping_method == 'follower_reprojection':
                pin_qpos = pin.neutral(self.vfollower_model)
                pin_qpos[self.idx_vfpin2state[:, 0]] = q_from_sliders[self.idx_vfpin2state[:, 1]]
                new_eef_poses = self.get_eef_poses(pin_qpos, self.vfollower_model, self.vfollower_data, self.vfollower_eef_frame_ids)
            else:
                raise

            local_poses, hand_poses = {}, {}
            for limb_id, (Rt, init_Rt) in enumerate(zip(new_eef_poses, self.init_eef_poses)):
                new_pos = init_Rt[:3, :3].T @ Rt[:3, 3] - init_Rt[:3, :3].T @ init_Rt[:3, 3]
                new_R = init_Rt[:3, :3].T @ Rt[:3, :3]
                new_pos = new_pos * self.motion_scale
                new_Rt = pt.transform_from(R=new_R, p=new_pos)
                leader_limb_name = self.leader_robot.limb_names[limb_id]
                follower_limb_name = self.leader_config.limb_mapping[leader_limb_name]
                local_poses[follower_limb_name] = new_Rt
                hand_poses[follower_limb_name] = q_from_sliders[self.hand_mapping[leader_limb_name]]

            return (local_poses, hand_poses)
        else:
            return q_from_sliders

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def close(self):
        return

    def get_eef_poses(self, pin_qpos, model, data, eef_frame_ids):
        pin.forwardKinematics(model, data, pin_qpos)
        eef_poses = []
        for frame_id in eef_frame_ids:
            oMf: pin.SE3 = pin.updateFramePlacement(model, data, frame_id)
            eef_poses.append(np.array(oMf))
        return eef_poses

if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)
    leader = SimPuppeteer(robot, leader_config, env_config, render_mode='human')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)


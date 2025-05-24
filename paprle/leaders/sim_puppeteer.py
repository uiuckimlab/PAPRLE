import numpy as np
import tkinter as tk
import copy
from paprle.envs.mujoco_env_utils.util import MultiSliderClass
from threading import Thread
from pytransform3d import transformations as pt
from paprle.follower import Robot
from paprle.utils.config_utils import add_info_robot_config

import pinocchio as pin


class SimPuppeteer:
    def __init__(self, robot, leader_config, env_config, render_mode='none', verbose=False, *args, **kwargs):

        leader_config.robot_cfg = add_info_robot_config(leader_config)
        self.leader_robot = Robot(leader_config)
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
            from yourdfpy.urdf import URDF
            self.leader_model = URDF.load(leader_config.asset_cfg.urdf_path)
            self.render_thread = Thread(target=self.__render_trimesh, args=())
            self.render_thread.start()
        elif render_mode != 'none' and render_base == 'mujoco':
            from paprle.visualizer.mujoco import MujocoViz
            self.leader_model = MujocoViz(self.leader_robot.robot_config, env_config)

        self.last_qpos = self.leader_robot.init_qpos
        self.output_type = leader_config.output_type

        self.direct_joint_mapping = leader_config.output_type == 'joint_pos'
        if not self.direct_joint_mapping:
            self.motion_mapping_method = leader_config.motion_mapping
            # Load Leader Pin Model
            self.pin_model, collision_model, visual_model = pin.buildModelsFromUrdf(
                leader_config.asset_cfg.urdf_path,
                package_dirs=[leader_config.asset_cfg.asset_dir])
            data = self.pin_model.createData()
            self.pin_model_joint_names = [name for name in self.pin_model.names]
            if 'universe' in self.pin_model_joint_names:
                self.pin_model_joint_names.remove('universe')
            frame_mapping = {}
            for i, frame in enumerate(self.pin_model.frames):
                frame_mapping[frame.name] = i
            self.end_effector_frame_ids = [frame_mapping[eef_name] for limb_name, eef_name in leader_config.end_effector_link.items()]
            self.motion_scale = getattr(leader_config, 'motion_scale', 1.0)
            self.neutral_pos = getattr(leader_config, 'neutral_pos', pin.neutral(self.model))
            self.neutral_pos = np.array(self.neutral_pos, dtype=np.float32)
            self.idx_state2pin = [self.leader_robot.joint_names.index(name) for name in self.pin_model_joint_names]




        return

    def __render_trimesh(self):
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

    def reset(self, ):
        self.require_end = True
        return

    def launch_init(self, init_env_qpos):
        self.initialize(init_env_qpos)
        self.last_qpos[:] = init_env_qpos
        self.is_ready = True
        self.require_end = False
        return

    def initialize(self, init_env_qpos):
        if init_env_qpos is None:
            init_env_qpos = self.leader_robot.init_qpos.copy()
        self.sliders.set_slider_values(init_env_qpos)
        return

    def close_init(self):
        return

    def get_status(self):
        self.sliders.update()
        q_from_sliders = self.sliders.get_slider_values()
        self.last_qpos = q_from_sliders
        if self.output_type == 'delta_eef_pose':
            arm_poses, hand_poses = {}, {}
            for limb_name in self.leader_robot.limb_names:
                arm_poses[limb_name] = q_from_sliders[self.leader_robot.ctrl_joint_idx_mapping[limb_name]]
                hand_poses[limb_name] = q_from_sliders[self.leader_robot.ctrl_hand_joint_idx_mapping[limb_name]]
            return (arm_poses, hand_poses)
        else:
            return q_from_sliders

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def close(self):
        return

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


import numpy as np
import tkinter as tk
import copy
from paprle.envs.mujoco_env_utils.util import MultiSliderClass
from threading import Thread
from pytransform3d import transformations as pt


# Only supports direct joint mapping
class Sliders:
    def __init__(self, robot, leader_config, env_config, render_mode='none', verbose=False, *args, **kwargs):
        self.leader_config = leader_config
        self.robot = robot
        self.is_ready = False
        self.require_end = False
        self.shutdown = False

        self.joint_names = robot.joint_names
        self.sliders = MultiSliderClass(
            n_slider = len(robot.joint_names),
            title=f'[LEADER] {leader_config.name} - {robot.name}',
            window_width=600,
            window_height=800,
            x_offset=50,
            y_offset=100,
            slider_width=300,
            label_texts=robot.joint_names,
            slider_mins=robot.joint_limits[:, 0],
            slider_maxs=robot.joint_limits[:, 1],
            slider_vals=robot.init_qpos,
            resolution=0.001,
            VERBOSE=verbose,
        )
        # add reset button in the slider
        self.reset_button = tk.Button(self.sliders.gui, text="RESET", command=self.reset)
        self.reset_button.pack(side=tk.BOTTOM, fill=tk.X)

        self.render_mode = render_mode
        if render_mode != 'none':
            from yourdfpy.urdf import URDF
            self.leader_model = URDF.load(self.robot.urdf_file)
            self.render_thread = Thread(target=self.__render, args=())
            self.render_thread.start()

        self.last_qpos = robot.init_qpos
        self.output_type = leader_config.output_type

        return

    def __render(self):
        joint_mapping_inds = [self.leader_model.actuated_joint_names.index(name) for name in self.robot.joint_names]
        def callback(scene,  **kwargs ):
            leader_model_pose =  np.zeros(len(self.leader_model.actuated_joint_names))
            leader_model_pose[joint_mapping_inds] = self.last_qpos
            self.leader_model.update_cfg(leader_model_pose)
            # # To get current camera transform
            # print(self.leader_model._scene.camera_transform)
            # print("pq: ", pt.pq_from_transform(self.leader_model._scene.camera_transform).tolist())

        if 'trimesh' in self.robot.robot_config.viewer_args:
            pq = self.robot.robot_config.viewer_args.trimesh.pq
            self.leader_model._scene.camera_transform = pt.transform_from_pq(pq)
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
            init_env_qpos = self.robot.init_qpos
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
            for limb_name in self.robot.limb_names:
                arm_poses[limb_name] = q_from_sliders[self.robot.ctrl_joint_idx_mapping[limb_name]]
                hand_poses[limb_name] = q_from_sliders[self.robot.ctrl_hand_joint_idx_mapping[limb_name]]
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
    leader = Sliders(robot, leader_config, env_config, render_mode='human')

    for ep in range(100):
        leader.launch_init(None)
        while not leader.is_ready:
            time.sleep(0.01)

        while True:
            if leader.require_end: break
            qpos = leader.get_status()
            #print(qpos)
            time.sleep(0.01)


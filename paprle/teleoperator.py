from paprle.ik.pinocchio import PinocchioIKSolver
from paprle.hands import BaseHand, ParallelGripper, PowerGripper
from paprle.collision import MujocoCollisionChecker
from paprle.visualizer.mujoco import MujocoViz
import numpy as np
from pytransform3d import transformations as pt
from threading import Thread, Lock
import time

class Teleoperator:
    def __init__(self, robot, leader_config, env_config, render_mode='mujoco'):
        self.shutdown, self.viz = False, None
        self.threads = []

        self.robot = robot
        self.collision_checker = MujocoCollisionChecker(robot)


        self.joint_limits = self.robot.joint_limits
        self.max_joint_diff = robot.robot_config.max_joint_vel * robot.robot_config.teleop_dt

        self.leader_config = leader_config
        self.env_config = env_config
        self.command_type = leader_config.output_type  # joint_pos or eef_pose

        self.init_qpos = robot.init_qpos

        self.ik_idx_mappings, self.curr_ee_poses = {}, {}
        self.ik_solvers, self.init_world2ees, self.world2bases, self.base2ees = {}, {}, {}, {}
        if self.command_type == 'delta_eef_pose':
            for limb_id, (limb_name, eef_ik_info) in enumerate(self.robot.ik_config.items()):
                ik_solver = PinocchioIKSolver(limb_name, eef_ik_info)
                self.ik_solvers[limb_name] = ik_solver

                joint_inds = self.robot.ctrl_joint_idx_mapping[limb_name]
                idx_mapping = [[i, eef_ik_info.joint_names.index(name)] for i, name in enumerate(ik_solver.get_joint_names()) if name in eef_ik_info.joint_names]
                idx_mapping = np.array(idx_mapping)
                self.ik_idx_mappings[limb_name] = idx_mapping

                transform = self.collision_checker.get_transform(eef_ik_info.base_link, getattr(self.robot.robot_config, 'world_link', 'world'))
                self.world2bases[limb_name] = transform

                arm_start_qpos = self.init_qpos[joint_inds]
                qpos = ik_solver.get_current_qpos()
                qpos[idx_mapping[:, 0]] = arm_start_qpos[idx_mapping[:, 1]]
                ik_solver.set_current_qpos(qpos)
                curr_ee_pose = pt.pq_from_transform(np.array(ik_solver.ee_pose))
                self.curr_ee_poses[limb_name] = curr_ee_pose
                self.base2ees[limb_name] = pt.transform_from_pq(curr_ee_pose)

                Rt = self.world2bases[limb_name] @ self.base2ees[limb_name]
                self.init_world2ees[limb_name] = Rt
            self.robot.robot_config.base_pose = {limb_name: pt.pq_from_transform(self.world2bases[limb_name]).tolist() for
                                           limb_name in self.robot.limb_names}

        self.pos_lock = Lock()
        self.last_target_qpos = self.init_qpos
        self.target_ee_poses = None

        self.hand_solvers = {}
        self.eef_type = self.robot.robot_config.eef_type # hand or gripper
        for i, limb_name in enumerate(self.robot.limb_names):
            if self.eef_type == 'parallel_gripper':
                hand_solver = ParallelGripper(self.robot.robot_config, self.leader_config, self.env_config)
            elif self.eef_type == 'power_gripper':
                name = list(self.robot.robot_config.retargeting)[i]
                retargeting_config = self.robot.robot_config.retargeting[name]
                hand_solver = PowerGripper(self.robot.robot_config, self.leader_config, self.env_config, retargeting_config, joint_limits=self.joint_limits)
            elif self.eef_type == None:
                hand_solver = BaseHand(self.robot.robot_config, self.leader_config, self.env_config)
            else:
                raise ValueError('Unknown end effector type: %s' % self.eef_type)
            self.hand_solvers[limb_name] = hand_solver

        self.render_mode = render_mode
        if self.render_mode:
            self.viz_thread = Thread(target=self.render_thread)
            self.viz_thread.start()
            self.threads.append(self.viz_thread)
        return

    def render_thread(self):
        while True:
            if self.shutdown:
                if self.viz is not None:
                    self.viz.env.close_viewer()
                return
            if self.render_mode:
                if self.viz is None:
                    # if self.env_config.sim == 'mujoco': # TODO: Add other visualizer, other than mujoco
                    #     self.viz = None # MeshcatViz(self.config)
                    # else:
                    if self.render_mode == 'mujoco':
                        self.viz = MujocoViz(self.robot)
                        viewer_args = self.robot.robot_config.viewer_args.mujoco
                        self.viz.init_viewer(viewer_title='Teleoperator',
                                             viewer_width=viewer_args.viewer_width,
                                             viewer_height=viewer_args.viewer_height,
                                             viewer_hide_menus=True)
                        self.viz.update_viewer(**viewer_args)
                # print(
                #     f"azimuth: {self.viz.env.viewer.cam.azimuth}\n"
                #     f"distance: {self.viz.env.viewer.cam.distance}\n"
                #     f"elevation: {self.viz.env.viewer.cam.elevation}\n"
                #     f"lookat: {self.viz.env.viewer.cam.lookat.tolist()}")
                if self.viz is not None:
                    if self.target_ee_poses is not None:
                        target_ee_poses = self.target_ee_poses.copy()
                        self.viz.set_ee_target(target_ee_poses)
                    if self.last_target_qpos is not None:
                        target_qpos = self.last_target_qpos.copy()
                        self.viz.set_qpos(target_qpos)
                    #self.viz.log = self.vis_log
                    self.viz.render()
                    self.last_image = self.viz.env.grab_image()
            time.sleep(0.001)

    def step(self, command, initial=False):
        target_qpos = self.last_target_qpos.copy()
        target_ee_poses = {}
        if self.command_type == 'joint_pos':
            target_qpos = command
        elif self.command_type == 'delta_eef_pose':
            pass

        target_qpos = self.process_joint_pos(target_qpos, initial=initial)
        with self.pos_lock:
            self.last_target_qpos = target_qpos
            if len(target_ee_poses):
                self.target_ee_poses = target_ee_poses
        return target_qpos

    def process_joint_pos(self, input_qpos, initial=False):
        qpos = np.clip(input_qpos, self.joint_limits[:,0], self.joint_limits[:,1])
        new_qpos = self.collision_checker.get_collision_free_pose(qpos)

        if new_qpos is not None:
            qpos = new_qpos
        else:
            qpos = self.last_target_qpos

        # if it deviates too much from the current joint angles, we need to move slowly
        if not initial:
            qpos_diff = qpos - self.last_target_qpos
            qpos_diff = np.clip(qpos_diff, -self.max_joint_diff, self.max_joint_diff)
            qpos = self.last_target_qpos + qpos_diff

        return qpos

    def reset(self):
        for eef_idx, limb_name in enumerate(self.robot.limb_names):
            self.hand_solvers[limb_name].reset()
            if len(self.ik_solvers) > 0:
                self.ik_solvers[limb_name].reset()
                qpos = self.ik_solvers[limb_name].get_current_qpos()
                ready_pose = self.robot.init_qpos
                curr_limb_ready_pose = ready_pose[self.robot.ctrl_joint_idx_mapping[limb_name]]
                qpos[self.ik_idx_mappings[limb_name][:, 0]] = curr_limb_ready_pose[self.ik_idx_mappings[limb_name][:, 1]]
                self.ik_solvers[limb_name].set_current_qpos(qpos)

                self.curr_ee_poses[limb_name] = pt.pq_from_transform(np.array(self.ik_solvers[limb_name].ee_pose))
                self.base2ees[limb_name] = pt.transform_from_pq(self.curr_ee_poses[limb_name])
                self.init_world2ees[limb_name] = self.world2bases[limb_name] @ self.base2ees[limb_name]
        self.last_target_qpos = self.init_qpos
        return

    def close(self):
        self.shutdown = True
        # If any threads are running, join them here
        for t in self.threads:
            t.join()
        return


if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.utils.config_utils import change_working_directory
    change_working_directory()
    from paprle.follower import Robot

    robot_config, device_config, env_config = BaseConfig().parse()
    robot = Robot(robot_config)

    teleoperator = Teleoperator(robot, device_config, env_config)


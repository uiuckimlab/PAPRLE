
from paprle.utils.misc import detect_ros_version
ROS_VERSION = detect_ros_version()
if ROS_VERSION == 'ROS1':
    import rospy
    import pinocchio as pin
elif ROS_VERSION == 'ROS2':
    import rclpy
    import pinocchio as pin
    from rclpy.node import Node
else:
    raise ImportError("Unknown ROS version. Please check your environment.")

import numpy as np
from omegaconf import OmegaConf
from threading import Thread, Lock
from pytransform3d import transformations as pt
from paprle.follower import Robot
from paprle.utils.config_utils import add_info_robot_config
import time
from sensor_msgs.msg import JointState


class Puppeteer:
    def __init__(self, follower_robot, leader_config, env_config, render_mode='none', verbose=False, *args, **kwargs):
        self.follower_robot = follower_robot
        leader_config.robot_cfg = add_info_robot_config(leader_config)
        self.leader_robot = Robot(leader_config)
        self.leader_config = leader_config
        self.env_config = env_config
        self.is_ready = False
        self.require_end = False
        self.shutdown = False

        self.ROS_VERSION = ROS_VERSION
        if self.ROS_VERSION == 'ROS1':
            try:
                rospy.init_node("Puppeteer")
            except rospy.exceptions.ROSException as e:
                print("Node has already been initialized, do nothing")
        elif self.ROS_VERSION == 'ROS2':
            rclpy.init(args=None)
            self.node = Node("Puppeteer")

        self.render_mode = render_mode
        render_base = leader_config.render_base
        if render_mode != 'none' and render_base == 'trimesh':
            self.render_thread = Thread(target=self.__render_trimesh, args=())
            self.render_thread.start()
        elif render_mode != 'none' and render_base == 'mujoco':
            self.render_thread = Thread(target=self.__render_mujoco, args=())
            self.render_thread.start()

        self.last_qpos = self.leader_robot.init_qpos
        self.last_command, self.command_lock = None, Lock()
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


        self.leader_states = None
        self.pos_lock = Lock()
        self.hand_offsets, self.hand_scales, self.hand_ids = [], [], []
        for leader_limb_name, leader_hand_joint_idxs in self.leader_robot.ctrl_hand_joint_idx_mapping.items():
            follower_min_val, follower_max_val = 0.0, 1.0
            gripper_min_val, gripper_max_val = self.leader_config.hand_limits[leader_limb_name]
            scale = (follower_max_val - follower_min_val) / (gripper_max_val - gripper_min_val)
            offset = follower_min_val - gripper_min_val * scale
            self.hand_offsets.append(offset)
            self.hand_scales.append(scale)
            self.hand_ids.extend(leader_hand_joint_idxs)

        self.topic_joint_names, self.topic_joint_mapping = [], []
        if self.ROS_VERSION == 'ROS1':
            self.subscriber = rospy.Subscriber(leader_config.leader_subscribe_topic, JointState, self.joint_state_callback)
            self.log_info = self.log_info
            self.ros_shutdown = rospy.is_shutdown

        elif self.ROS_VERSION == 'ROS2':
            self.subscriber = self.node.create_subscription(JointState, leader_config.leader_subscribe_topic, self.joint_state_callback, 10)
            self.log_info = self.node.get_logger().info

            def spin_executor(arm):
                from rclpy.executors import SingleThreadedExecutor
                executor = SingleThreadedExecutor()
                executor.add_node(arm)
                executor.spin()
                return
            self.sub_thread = Thread(target=spin_executor, args=(self.node,))
            self.sub_thread.start()
            self.ros_shutdown = lambda : (self.node.executor is not None and self.node.executor._is_shutdown)

        while self.leader_states is None:
            print("Waiting for the leader joint states...")
            time.sleep(0.01)


        self.leader_viz_info = {'color': 'blue',  'log': "Puppeteer is ready!"}
        self.end_detection_thread = Thread(target=self.detect_end_signal)
        self.end_detection_thread.start()
        return

    def joint_state_callback(self, msg):
        if len(self.topic_joint_names) == 0:
            self.topic_joint_names = msg.name
            self.topic_joint_mapping = [[id, self.topic_joint_names.index(name)] for id, name in enumerate(self.leader_robot.joint_names) if name in self.topic_joint_names]
            self.topic_joint_mapping = np.array(self.topic_joint_mapping)

        leader_states = np.zeros(len(self.leader_robot.joint_names))
        leader_states[self.topic_joint_mapping[:, 0]] = np.array(msg.position)[self.topic_joint_mapping[:, 1]]
        leader_states[self.hand_ids] = leader_states[self.hand_ids] * np.array(self.hand_scales) + np.array(self.hand_offsets)
        with self.pos_lock:
            self.leader_states = leader_states
        return

    def detect_end_signal(self):
        dt, gripper_th = 0.03, 0.8
        iteration = 0
        enough_to_end, threshold_time = 0.0, 3.0
        while not self.shutdown and not self.ros_shutdown():
            if self.leader_states is None or not self.is_ready:
                time.sleep(dt)
                continue
            start_time = time.time()
            self.leader_viz_info['color'] = 'green'
            with self.pos_lock:
                positions = self.leader_states[self.hand_ids].copy()
                
            hand_closed = (positions[self.hand_ids] > gripper_th).all()
            
            pin_qpos = pin.neutral(self.pin_model)
            pin_qpos[self.idx_pin2state[:, 0]] = positions[self.idx_pin2state[:, 1]]
            new_eef_poses = self.get_eef_poses(pin_qpos, self.pin_model, self.pin_data, self.eef_frame_ids)
            diffs = []
            for i in range(len(new_eef_poses)):
                diff = abs(new_eef_poses[i][:3, 3] - np.array(self.leader_config.reset_pose[i])) < np.array(self.leader_config.reset_cube)
                diffs.append(diff.all())
            wrist_up = np.array(diffs).all()
            
            if not self.require_end and hand_closed and wrist_up:
                enough_to_end += dt
                self.leader_viz_info['color'] = (enough_to_end * 255 / threshold_time, 255 - enough_to_end * 255 / threshold_time, 0)
                self.leader_viz_info['log'] = f"Running... End signal detected! {enough_to_end:.2f}/{threshold_time:d}"
                print(f"Running... End signal detected! {enough_to_end:.2f}/{threshold_time:d}", end="\r")
            else:
                if enough_to_end > 0.0:
                    print(f"Running... End signal detected! 0.0/{threshold_time:d}", end="\r")
                enough_to_end = 0.0
            
            if enough_to_end >= threshold_time:
                self.require_end = True
                self.leader_viz_info['color'] = 'red'
                self.leader_viz_info['log'] = "End signal detected! Resetting the leader and follower"
                self.is_ready = False
                enough_to_end = 0.0
                print("[Puppeteer] End signal detected!, resetting the controller...")
        
            left_time = max(dt -  (time.time() - start_time), 0.0)
            time.sleep(left_time)
            iteration += 1
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
        gripper_th, iteration = 0.8, 0
        print("[Leader] Launching Puppeteer initialization...")

        while not ((self.leader_states[self.hand_ids] < gripper_th).all()):
            #print("Curr gripper values:", self.leader_states[self.hand_ids])
            print("Open the gripper to initialize the controller....", end="\r")
            time.sleep(0.1)

        self.init_thread = Thread(target=self.initialize, args=(init_env_qpos,))
        self.init_thread.start()
        return

    def initialize(self, init_env_qpos):
        gripper_th, iteration, initialize_progress = 0.8, 0, 0
        dt, threshold_time = 0.01, 3
        prev_pose = None

        arm_inds = [i for i in range(len(self.leader_robot.joint_names)) if i not in self.hand_ids]
        while not initialize_progress >= threshold_time:
            start_time = time.time()
            if self.shutdown or self.ros_shutdown(): return

            hand_closed = (self.leader_states[self.hand_ids] > gripper_th).all()
            if hand_closed:
                initialize_progress += dt
            else:
                initialize_progress = 0.0

            curr_status = ''

            diff = np.linalg.norm(self.leader_states[arm_inds] - prev_pose) if prev_pose is not None else 0.0
            if diff > 0.01:
                initialize_progress = 0
                curr_status += "Don't move!"

            print(
                f"Close the grippers to initialize the controller.... {initialize_progress:.2f}/{threshold_time}   gripper: " + curr_status,
                end="\r")

            self.leader_viz_info['color'] = 'blue'
            self.leader_viz_info['log'] = f"Close the grippers to initialize the controller.... {initialize_progress:.2f}/{threshold_time}   gripper: " + curr_status

            left_time = max(dt - (time.time() - start_time), 0.0)
            time.sleep(left_time)
            iteration += 1
            prev_pose = self.leader_states[arm_inds].copy()

        print("[Puppeteer] Gripper closed. Initializing the leader...")
        self.leader_viz_info['color'] = 'red'
        self.leader_viz_info['log'] = 'Initializing the controller...'

        curr_pose = self.leader_states[arm_inds].copy()
        if not self.direct_joint_mapping:
            init_env_qpos = curr_pose
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

        self.last_qpos = init_env_qpos
        self.leader_viz_info['color'] = 'green'
        self.leader_viz_info['log'] = 'Puppeteer initialized successfully!'
        self.is_ready = True
        self.require_end = False
        print("[Puppeteer] Puppeteer initialized successfully!")
        return

    def close_init(self):
        self.init_thread.join()
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
            with self.command_lock:
                self.last_command = (local_poses, hand_poses)
            return (local_poses, hand_poses)
        else:
            with self.command_lock:
                self.last_command = q_from_sliders.copy()
            return q_from_sliders

    def update_vis_info(self, env_vis_info):
        if env_vis_info is not None:
            env_vis_info['leader'] = self.leader_viz_info
        return env_vis_info

    def close(self):
        self.shutdown = True
        self.end_detection_thread.joint()
        if self.init_thread is not None:
            self.init_thread.join()
        print("[Puppeteer] Closed Successfully!")
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


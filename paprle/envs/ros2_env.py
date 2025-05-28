import rclpy
import numpy as np
import time
import copy
from threading import Thread, Lock
from rclpy.node import Node
from rclpy.parameter import Parameter, ParameterType
from rcl_interfaces.srv import GetParameters
from paprle.envs.base_env import BaseEnv
from pymoveit2 import MoveIt2
import xml.etree.ElementTree as ET
from paprle.envs.ros2_env_utils.subscribe_and_publish import JointStateSubscriber, ControllerPublisher

def lp_filter(new_value, prev_value, alpha=0.5):
    if prev_value is None: return new_value
    if not isinstance(prev_value, np.ndarray): prev_value = np.array(prev_value)
    if not isinstance(new_value, np.ndarray): new_value = np.array(new_value)
    if prev_value.shape != new_value.shape:
        prev_value = prev_value.mean(0)
    y = alpha * new_value + (1 - alpha) * prev_value
    return y

def calculate_vel(last_qpos, qpos, dt):
    if last_qpos.shape != qpos.shape:
        last_qpos = last_qpos.mean(0)
    vel = (qpos - last_qpos) / dt
    return vel

# TODO: change use_sim_time param to True if you want to use gazebo
class ROS2Env(BaseEnv):
    def __init__(self, robot, device_config, env_config, verbose=False, render_mode=False, **kwargs):
        super().__init__(robot, device_config, env_config, verbose=verbose, render_mode=render_mode, **kwargs)
        try:
            rclpy.init()
        except:
            print("rclpy already initialized")

        self.motion_planning_method = robot.ros2_config.motion_planning
        if self.motion_planning_method == 'moveit':
            self.moveit_node = Node('teleop_env_moveit')
            self.moveit_config = robot.ros2_config.moveit
            self.get_param_cli = None
            self.setup_moveit()

            use_sim_time_param = self.get_move_group_params('use_sim_time')
            self.use_sim_time = use_sim_time_param if isinstance(use_sim_time_param,  bool) else use_sim_time_param.bool_value
        else:
            self.use_sim_time = True
        self.use_sim_time = False
        # Setup Subscribers and Publishers
        topics_to_sub, topics_to_pub = {}, {}
        for limb_name, limb_info in robot.ros2_config.robots.items():
            arm_state_sub_topic = limb_info['arm_state_sub_topic']
            joint_names = robot.robot_config.limb_joint_names[limb_name]
            if arm_state_sub_topic not in topics_to_sub:
                topics_to_sub[arm_state_sub_topic] = {'type': limb_info['arm_state_sub_msg_type'], 'joint_names': []}
            topics_to_sub[arm_state_sub_topic]['joint_names'].extend(joint_names)

            arm_control_pub_topic = limb_info['arm_control_topic']
            if arm_control_pub_topic not in topics_to_pub:
                topics_to_pub[arm_control_pub_topic] = {'type': limb_info['arm_control_msg_type'], 'joint_names': []}
            topics_to_pub[arm_control_pub_topic]['joint_names'].extend(joint_names)

            hand_state_sub_topic = limb_info['hand_state_sub_topic']
            hand_joint_names = robot.robot_config.hand_joint_names[limb_name]
            if hand_state_sub_topic not in topics_to_sub:
                topics_to_sub[hand_state_sub_topic] = {'type': limb_info['hand_state_msg_type'], 'joint_names': []}
            topics_to_sub[hand_state_sub_topic]['joint_names'].extend(hand_joint_names)

            hand_control_pub_topic = limb_info['hand_control_topic']
            if hand_control_pub_topic not in topics_to_pub:
                topics_to_pub[hand_control_pub_topic] = {'type': limb_info['hand_control_msg_type'], 'joint_names': []}
            topics_to_pub[hand_control_pub_topic]['joint_names'].extend(hand_joint_names)

        def spin_executor(node, spin_name='', mode='single'):
            from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
            if mode == 'single':
                executor = SingleThreadedExecutor()
            else:
                executor = MultiThreadedExecutor()
            executor.add_node(node)
            executor.spin()
            print(f"Shutting down {spin_name}")
            return
        if robot.name == 'g1':
            from paprle.envs.ros2_env_utils.g1_subscribe_and_publish import JointStateSubscriber as G1JointStateSubscriber
            self.state_subscriber = G1JointStateSubscriber(topics_to_sub, robot.joint_names)
        else:
            self.state_subscriber = JointStateSubscriber(topics_to_sub, robot.joint_names)
        self.state_subscriber.set_parameters([Parameter('use_sim_time', value=self.use_sim_time)])
        self.sub_thread = Thread(target=spin_executor, args=(self.state_subscriber,"Subscriber Thread"))
        self.sub_thread.start()

        # TODO: Maybe wait here until the first state is updated
        iter = 0
        while not all(self.state_subscriber.flag_first_state_updated.values()):
            ss = '.' * (iter % 5 + 1)
            print("[Env] Waiting for first state update.." + ss, end='\r')
            time.sleep(0.1)
            iter += 1

        self.command_lock = Lock()
        self.last_command, self.lp_filter_alpha = None, self.robot.ros2_config.lp_filter_alpha
        self.last_vel, self.dt = None, self.robot.control_dt
        if robot.name == 'g1':
            from paprle.envs.ros2_env_utils.g1_subscribe_and_publish import ControllerPublisher as G1ControllerPublisher
            self.controller_publisher = G1ControllerPublisher(topics_to_pub, robot.joint_names, self.state_subscriber.states)
        else:
            self.controller_publisher = ControllerPublisher(topics_to_pub, robot.joint_names, self.state_subscriber.states)
        self.controller_publisher.set_parameters([Parameter('use_sim_time', value=self.use_sim_time)])
        self.pub_thread = Thread(target=spin_executor, args=(self.controller_publisher,"Publisher Thread",'multi'))
        self.pub_thread.start()

    def setup_get_move_group_params(self):
        self.get_param_cli = self.moveit_node.create_client(GetParameters, '/move_group/get_parameters')
        while not self.get_param_cli.wait_for_service(timeout_sec=1.0):
            self.moveit_node.get_logger().info('[Env] [Moveit] service not available, waiting again...')
        self.req = GetParameters.Request()
        return

    def get_move_group_params(self, param):
        if self.get_param_cli is None: self.setup_get_move_group_params()
        self.req.names = [param]
        itreation = 0
        while True:
            itreation += 1
            self.future = self.get_param_cli.call_async(self.req)
            rclpy.spin_until_future_complete(self.moveit_node, self.future, timeout_sec=1.0)
            if self.future.result() is not None:
                return self.future.result().values[0]
            elif itreation > 10: return False
            else:
                print("Service call failed %r" % (self.future.exception(),))
                time.sleep(1)
                continue

    def moveit_extract_joint_info(self):
        semantic_description = self.get_move_group_params('robot_description_semantic')
        self.moveit_info = ET.fromstring(semantic_description.string_value)
        self.moveit_group_poses, self.moveit_arms_joint_names, self.moveit_hand_joint_names = {}, [], []
        for child in self.moveit_info :
            if child.tag == 'group_state' and child.attrib['group'] == self.moveit_config.arm_group_name:
                if self.moveit_config.arm_group_name not in self.moveit_group_poses:
                    self.moveit_group_poses[self.moveit_config.arm_group_name] = {}
                self.moveit_group_poses[self.moveit_config.arm_group_name][child.attrib['name']] = []
                for joint in child:
                    self.moveit_group_poses[self.moveit_config.arm_group_name][child.attrib['name']].append(float(joint.attrib['value']))
                if len(self.moveit_arms_joint_names) == 0:
                    self.moveit_arms_joint_names = [joint.attrib['name'] for joint in child]
            elif child.tag == 'group_state' and child.attrib['group'] == self.moveit_config.hand_group_name:
                if self.moveit_config.hand_group_name not in self.moveit_group_poses:
                    self.moveit_group_poses[self.moveit_config.hand_group_name] = {}
                self.moveit_group_poses[self.moveit_config.hand_group_name][child.attrib['name']] = []
                for joint in child:
                    self.moveit_group_poses[self.moveit_config.hand_group_name][child.attrib['name']].append(float(joint.attrib['value']))
                if len(self.moveit_hand_joint_names) == 0:
                    self.moveit_hand_joint_names = [joint.attrib['name'] for joint in child]
        self.ctrl2moveit_arm_mapping = np.array([
            [id, self.moveit_arms_joint_names.index(name)]
            for id, name in enumerate(self.robot.joint_names)
            if name in self.moveit_arms_joint_names
        ])
        self.ctrl2moveit_hand_mapping = np.array([
            [id, self.moveit_hand_joint_names.index(name)]
            for id, name in enumerate(self.robot.joint_names)
            if name in self.moveit_hand_joint_names
        ])
        return

    def setup_moveit(self):
        self.moveit_extract_joint_info()
        self.arms_group = MoveIt2(node=self.moveit_node, joint_names=self.moveit_arms_joint_names,
                                  base_link_name='', end_effector_name='',
                                  group_name=self.moveit_config.arm_group_name, use_move_group_action=self.moveit_config.use_move_group_action)
        self.arms_group.num_planning_attempts = self.moveit_config.num_planning_attempts
        self.arms_group.allowed_planning_time = self.moveit_config.planning_time
        self.arms_group.max_velocity = self.moveit_config.max_velocity_scaling_factor

        self.arms_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.arm_group_name]['init'])
        self.arms_group.wait_until_executed()

        if self.moveit_hand_joint_names:
            self.hand_group = MoveIt2(node=self.moveit_node, joint_names=self.moveit_hand_joint_names,
                                      base_link_name='', end_effector_name='',
                                      group_name=self.moveit_config.hand_group_name, use_move_group_action=True)
            self.hand_group.num_planning_attempts = self.moveit_config.num_planning_attempts
            self.hand_group.allowed_planning_time = self.moveit_config.planning_time
            self.hand_group.max_velocity = self.moveit_config.max_velocity_scaling_factor
            self.hand_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.hand_group_name]['init'])
            self.hand_group.wait_until_executed()
        return

    def initialize(self, initial_qpos: np.ndarray) -> None:
        if self.motion_planning_method == 'moveit':
            moveit_pose = np.zeros(len(self.moveit_arms_joint_names))
            moveit_pose[self.ctrl2moveit_arm_mapping[:,1]] = initial_qpos[self.ctrl2moveit_arm_mapping[:,0]]
            self.arms_group.move_to_configuration(moveit_pose, joint_names=self.moveit_arms_joint_names)
            self.arms_group.wait_until_executed()
        else:
            self.controller_publisher.interpolate_duration = 2.0
            self.controller_publisher.interpolate_time_ = {
                k: 0.0 for k in self.controller_publisher.interpolate_time_.keys()
            }
            self.controller_publisher.mode = 'interpolate'
            with self.command_lock:
                self.controller_publisher.command_pos = initial_qpos
            while not (np.array(list(self.controller_publisher.interpolate_time_.values())) > self.controller_publisher.interpolate_duration).all():
                time.sleep(0.1)

        self.controller_publisher.mode = 'direct_publish'
        self.controller_publisher.interpolate_time_ = {
            k: 0.0 for k in self.controller_publisher.interpolate_time_.keys()
        }
        curr_pose = copy.deepcopy(self.state_subscriber.states['pos'])
        with self.command_lock:
            self.controller_publisher.command_pos = curr_pose
            self.controller_publisher.command_vel = np.zeros_like(self.state_subscriber.states['vel'])
        self.last_command = curr_pose
        self.initialized = True
        return

    def close(self):
        self.rest_position()
        with self.command_lock:
            self.controller_publisher.command_pos = None
        self.controller_publisher.destroy_node()
        self.state_subscriber.destroy_node()
        self.moveit_node.destroy_node()
        return

    def rest_position(self):
        self.arms_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.arm_group_name]['rest'])
        self.arms_group.wait_until_executed()
        if self.moveit_hand_joint_names:
            self.hand_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.hand_group_name]['init'])
            self.hand_group.wait_until_executed()
        return

    def reset(self):
        if self.motion_planning_method == 'moveit':
            # when we expect moveit to move the robot,
            # we need to clear out the current target command to not interfere with moveit
            with self.command_lock:
                self.controller_publisher.command_pos = None
                self.controller_publisher.command_vel = None
                self.controller_publisher.command_acc = None

            self.arms_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.arm_group_name]['init'])
            self.arms_group.wait_until_executed()

            if self.moveit_hand_joint_names:
                self.hand_group.move_to_configuration(self.moveit_group_poses[self.moveit_config.hand_group_name]['init'])
                self.hand_group.wait_until_executed()
        else:
            # else, we will just interpolate the poses to the init pose, so we don't need to clear the target qpos
            # actually, considering g1, it is not good to clear the target qpos, because it will make g1 to damping mode - dangerous!
            self.controller_publisher.interpolate_duration = 2.0
            self.controller_publisher.interpolate_time_ = {
                k: 0.0 for k in self.controller_publisher.interpolate_time_.keys()
            }
            self.controller_publisher.mode = 'interpolate'
            init_qpos = self.robot.init_qpos.copy()
            with self.command_lock:
                self.controller_publisher.command_pos = init_qpos
            while not (np.array(list(self.controller_publisher.interpolate_time_.values())) > self.controller_publisher.interpolate_duration).all():
                time.sleep(0.1)
        self.controller_publisher.mode = 'direct_publish'
        self.controller_publisher.interpolate_time_ = {
            k: 0.0 for k in self.controller_publisher.interpolate_time_.keys()
        }
        with self.command_lock:
            self.controller_publisher.command_pos = copy.deepcopy(self.state_subscriber.states['pos'])
            self.controller_publisher.command_vel = np.zeros_like(self.state_subscriber.states['vel'])
    
        return np.array(self.state_subscriber.states['pos'].copy())
    
    def step(self, command):
        pos = lp_filter(command, self.last_command, self.lp_filter_alpha)
        vel = calculate_vel(self.last_command, pos, self.dt)
        if self.last_vel is not None:
            acc = calculate_vel(self.last_vel, vel, self.dt)
        else:
            acc = None

        with self.command_lock:
            self.controller_publisher.command_pos = pos
            self.controller_publisher.command_vel = vel
            self.controller_publisher.command_acc = acc
        self.last_command = pos
        return

    def get_current_qpos(self):
        return np.array(self.state_subscriber.states['pos'].copy())










if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.utils.config_utils import change_working_directory
    change_working_directory()
    from paprle.follower import Robot

    robot_config, device_config, env_config = BaseConfig().parse()
    robot = Robot(robot_config)
    env = ROS2Env(robot, device_config, env_config, verbose=False, render_mode='mujoco')
    obs = env.reset()
    env.initialize(robot.init_qpos)

    min_qpos = robot.joint_limits[:, 0] + 0.3
    max_qpos = robot.joint_limits[:, 1] - 0.3
    interpolate_trajectory = np.linspace(min_qpos, max_qpos, num=1000)
    while True:
        env.initialize(interpolate_trajectory[0])
        for i in range(100):
            action = interpolate_trajectory[i]
            obs, rew, done, info = env.step(action)
            if done:
                break
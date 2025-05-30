import sys
import os
import time

sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append("/opt/ros/noetic/lib/python3.8/site-packages/")
import rospy
import rospy
import numpy as np
from omegaconf import OmegaConf
import moveit_commander
import copy
import cv2
from threading import Thread, Lock
import logging
from paprle.envs.base_env import BaseEnv
import xml.etree.ElementTree as ET
from paprle.envs.ros_env_utils.subscribe_and_publish import JointStateSubscriber, ControllerPublisher
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

class ROSEnv(BaseEnv):
    def __init__(self, robot, device_config, env_config, verbose=False, render_mode=False, **kwargs):
        super().__init__(robot, device_config, env_config, verbose=verbose, render_mode=render_mode, **kwargs)
        try:
            rospy.init_node('ros_env_node')
        except:
            print("rclpy already initialized")


        self.motion_planning_method = robot.ros2_config.motion_planning
        if self.motion_planning_method == 'moveit':
            self.moveit_config = robot.ros2_config.moveit
            self.setup_moveit()

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

        if robot.name == 'g1':
            from paprle.envs.ros2_env_utils.g1_subscribe_and_publish import \
                JointStateSubscriber as G1JointStateSubscriber
            self.state_subscriber = G1JointStateSubscriber(topics_to_sub, robot.joint_names)
        else:
            self.state_subscriber = JointStateSubscriber(topics_to_sub, robot.joint_names)

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
            self.controller_publisher = G1ControllerPublisher(topics_to_pub, robot.joint_names,
                                                              self.state_subscriber.states)
        else:
            self.controller_publisher = ControllerPublisher(topics_to_pub, robot.joint_names,
                                                            self.state_subscriber.states)

    def setup_moveit(self):
        self.moveit_extract_joint_info()
        self.arms_group = moveit_commander.MoveGroupCommander(
            robot_description=f"{self.moveit_config.namespace}/robot_description",
            name=self.moveit_config.arm_group_name,
            ns=self.moveit_config.namespace)
        if self.moveit_config.hand_group_name != '':  # If the hand can be controlled by moveit
            self.hands_group = moveit_commander.MoveGroupCommander(
                robot_description=f"{self.moveit_config.namespace}/robot_description",
                name=self.moveit_config.hand_group_name,
                ns=self.moveit_config.namespace)
        self.planning_scene = moveit_commander.PlanningSceneInterface(ns=self.moveit_config.namespace)

        self.arms_group.set_num_planning_attempts(self.moveit_config.num_planning_attempts)
        self.arms_group.set_planning_time(self.moveit_config.planning_time)
        self.arms_group.set_max_velocity_scaling_factor(self.moveit_config.max_velocity_scaling_factor)
        self.arms_group.set_max_acceleration_scaling_factor(self.moveit_config.max_acceleration_scaling_factor)
        if self.moveit_config.hand_group_name != '':
            self.hands_group.set_max_velocity_scaling_factor(self.moveit_config.max_velocity_scaling_factor)
            self.hands_group.set_max_acceleration_scaling_factor(self.moveit_config.max_acceleration_scaling_factor)

    def moveit_extract_joint_info(self):
        semantic_description = rospy.get_param(f'{self.moveit_config.namespace}/robot_description_semantic')
        self.moveit_info = ET.fromstring(semantic_description)
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


    def initialize(self, initial_qpos: np.ndarray) -> None:
        if self.motion_planning_method == 'moveit':
            moveit_pose = np.zeros(len(self.moveit_arms_joint_names))
            moveit_pose[self.ctrl2moveit_arm_mapping[:,1]] = initial_qpos[self.ctrl2moveit_arm_mapping[:,0]]
            self.arms_group.set_start_state_to_current_state()
            self.arms_group.set_joint_value_target(moveit_pose)
            self.arms_group.go(wait=True)
            self.arms_group.stop()
            self.arms_group.clear_pose_targets()

            if self.moveit_config.hand_group_name != '':
                self.hands_group.set_named_target("init")
                self.hands_group.go(wait=True)
                self.hands_group.stop()
                self.hands_group.clear_pose_targets()
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
        self.shutdown = True
        self.rest_position()
        with self.command_lock:
            self.controller_publisher.command_pos = None
        for timer in self.controller_publisher.timers:
            timer.stop()

        return

    def rest_position(self):
        self.arms_group.set_named_target("rest")
        self.arms_group.go(wait=True)
        self.arms_group.stop()
        self.arms_group.clear_pose_targets()

        # TODO: Find some general way to initialize the hand group
        if self.moveit_config.hand_group_name != '':
            # open with moveit
            try:
                self.hands_group.set_named_target("rest")
            except:
                self.hands_group.set_named_target("open")
            self.hands_group.go(wait=True)
            self.hands_group.stop()
            self.hands_group.clear_pose_targets()
        return

    def reset(self):
        if self.motion_planning_method == 'moveit':
            # when we expect moveit to move the robot,
            # we need to clear out the current target command to not interfere with moveit
            with self.command_lock:
                self.controller_publisher.command_pos = None
                self.controller_publisher.command_vel = None
                self.controller_publisher.command_acc = None

            self.arms_group.set_named_target("init")
            self.arms_group.go(wait=True)
            self.arms_group.stop()
            self.arms_group.clear_pose_targets()

            # TODO: Find some general way to initialize the hand group
            if self.moveit_config.hand_group_name != '':
                # open with moveit
                try:
                    self.hands_group.set_named_target("init")
                except:
                    self.hands_group.set_named_target("open")
                self.hands_group.go(wait=True)
                self.hands_group.stop()
                self.hands_group.clear_pose_targets()
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
            while not (np.array(
                    list(self.controller_publisher.interpolate_time_.values())) > self.controller_publisher.interpolate_duration).all():
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




import rclpy
from rclpy.node import Node
from pymoveit2 import MoveIt2
from rcl_interfaces.srv import GetParameters
import xml.etree.ElementTree as ET
import time
import numpy as np

ARM_GROUP_NAME = 'arm34'
GRIPPER_GROUP_NAME = 'gripper34'
POSE_NAME = 'rest'
VELOCITY_FACTOR = 0.1
ACCELERATION_FACTOR = 0.1
PLANNING_TIME = 5.0
PLANNING_ATTEMPTS = 10


class Moveit:
    def __init__(self):
        self.moveit_node = Node('reset_moveit_node')
        self.get_param_cli = None

    def setup_get_move_group_params(self, get_parameters='/move_group/get_parameters'):
        self.get_param_cli = self.moveit_node.create_client(GetParameters, get_parameters)
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
            elif itreation > 10:
                return False
            else:
                print("Service call failed %r" % (self.future.exception(),))
                time.sleep(1)
                continue

    def moveit_extract_joint_info(self):
        semantic_description = self.get_move_group_params('robot_description_semantic')
        self.moveit_info = ET.fromstring(semantic_description.string_value)
        self.moveit_group_poses, self.moveit_arms_joint_names, self.moveit_hand_joint_names = {}, [], []
        for child in self.moveit_info:
            if child.tag == 'group_state' and child.attrib['group'] == ARM_GROUP_NAME:
                if ARM_GROUP_NAME not in self.moveit_group_poses:
                    self.moveit_group_poses[ARM_GROUP_NAME] = {}
                self.moveit_group_poses[ARM_GROUP_NAME][child.attrib['name']] = []
                for joint in child:
                    self.moveit_group_poses[ARM_GROUP_NAME][child.attrib['name']].append(
                        float(joint.attrib['value']))
                if len(self.moveit_arms_joint_names) == 0:
                    self.moveit_arms_joint_names = [joint.attrib['name'] for joint in child]
            elif child.tag == 'group_state' and child.attrib['group'] == GRIPPER_GROUP_NAME:
                if GRIPPER_GROUP_NAME not in self.moveit_group_poses:
                    self.moveit_group_poses[GRIPPER_GROUP_NAME] = {}
                self.moveit_group_poses[GRIPPER_GROUP_NAME][child.attrib['name']] = []
                for joint in child:
                    self.moveit_group_poses[GRIPPER_GROUP_NAME][child.attrib['name']].append(
                        float(joint.attrib['value']))
                if len(self.moveit_hand_joint_names) == 0:
                    self.moveit_hand_joint_names = [joint.attrib['name'] for joint in child]
        return

    def setup_moveit(self):
        self.moveit_extract_joint_info()
        self.arms_group = MoveIt2(node=self.moveit_node, joint_names=self.moveit_arms_joint_names,
                                  base_link_name='', end_effector_name='',
                                  group_name=ARM_GROUP_NAME,
                                  use_move_group_action=True)
        self.arms_group.num_planning_attempts = PLANNING_ATTEMPTS
        self.arms_group.allowed_planning_time = PLANNING_TIME
        self.arms_group.max_velocity = VELOCITY_FACTOR
        self.arms_group.max_acceleration = ACCELERATION_FACTOR

        self.arms_group.move_to_configuration(self.moveit_group_poses[ARM_GROUP_NAME][POSE_NAME])
        self.arms_group.wait_until_executed()

        if self.moveit_hand_joint_names:
            self.hand_group = MoveIt2(node=self.moveit_node, joint_names=self.moveit_hand_joint_names,
                                      base_link_name='', end_effector_name='',
                                      group_name=GRIPPER_GROUP_NAME, use_move_group_action=True)
            self.hand_group.num_planning_attempts = PLANNING_ATTEMPTS
            self.hand_group.allowed_planning_time = PLANNING_TIME
            self.hand_group.max_velocity = VELOCITY_FACTOR
            self.hand_group.max_acceleration = ACCELERATION_FACTOR
            if POSE_NAME in self.moveit_group_poses[GRIPPER_GROUP_NAME]:
                self.hand_group.move_to_configuration(
                    self.moveit_group_poses[GRIPPER_GROUP_NAME][POSE_NAME])
            else:
                self.hand_group.move_to_configuration(self.moveit_group_poses[GRIPPER_GROUP_NAME]['init'])
            self.hand_group.wait_until_executed()
        return

if __name__ == '__main__':
    rclpy.init()
    moveit = Moveit()
    moveit.setup_moveit()
    print("MoveIt setup complete.")
    rclpy.shutdown()


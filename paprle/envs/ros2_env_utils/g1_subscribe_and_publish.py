from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from functools import partial
from threading import Thread, Lock
import numpy as np
import rclpy

from unitree_hg.msg._low_cmd import LowCmd
from unitree_hg.msg._low_state import LowState
from unitree_sdk2py.utils.crc import CRC
import cyclonedds.idl as idl
class Custom_CRC(CRC):
    def Crc(self, msg: idl.IdlStruct):
        return self._CRC__Crc32(self._CRC__PackHGLowCmd(msg))
C = Custom_CRC()
class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

g1_joint_mapping = {
    'left_shoulder_pitch_joint':  G1JointIndex.LeftShoulderPitch,
    'left_shoulder_roll_joint': G1JointIndex.LeftShoulderRoll,
    'left_shoulder_yaw_joint': G1JointIndex.LeftShoulderYaw,
    'left_elbow_joint': G1JointIndex.LeftElbow,
    'left_wrist_roll_joint': G1JointIndex.LeftWristRoll,
    'left_wrist_pitch_joint': G1JointIndex.LeftWristPitch,
    'left_wrist_yaw_joint': G1JointIndex.LeftWristYaw,
    'right_shoulder_pitch_joint': G1JointIndex.RightShoulderPitch,
    'right_shoulder_roll_joint': G1JointIndex.RightShoulderRoll,
    'right_shoulder_yaw_joint': G1JointIndex.RightShoulderYaw,
    'right_elbow_joint': G1JointIndex.RightElbow,
    'right_wrist_roll_joint': G1JointIndex.RightWristRoll,
    'right_wrist_pitch_joint': G1JointIndex.RightWristPitch,
    'right_wrist_yaw_joint': G1JointIndex.RightWristYaw,
    'left_hip_pitch_joint': G1JointIndex.LeftHipPitch,
    'left_hip_roll_joint': G1JointIndex.LeftHipRoll,
    'left_hip_yaw_joint': G1JointIndex.LeftHipYaw,
    'left_knee_joint': G1JointIndex.LeftKnee,
    'left_ankle_pitch_joint': G1JointIndex.LeftAnklePitch,
    'left_ankle_roll_joint': G1JointIndex.LeftAnkleRoll,
    'right_hip_pitch_joint': G1JointIndex.RightHipPitch,
    'right_hip_roll_joint': G1JointIndex.RightHipRoll,
    'right_hip_yaw_joint': G1JointIndex.RightHipYaw,
    'right_knee_joint': G1JointIndex.RightKnee,
    'right_ankle_pitch_joint': G1JointIndex.RightAnklePitch,
    'right_ankle_roll_joint': G1JointIndex.RightAnkleRoll,
    'waist_yaw_joint': G1JointIndex.WaistYaw,
    'waist_roll_joint': G1JointIndex.WaistRoll,
    'waist_pitch_joint': G1JointIndex.WaistPitch,
}
g1_joint_names = list(g1_joint_mapping.keys())
sub_to_pose_mapping = [[g1_joint_names.index(name), g1_joint_mapping[name]] for name in g1_joint_names]
mode_machine = 0

class JointStateSubscriber(Node):
    def __init__(self, sub_info, output_joint_names):
        super().__init__('joint_state_subscriber')
        self.output_joint_names = output_joint_names
        self.joint_mapping = {}

        self.sub_info = sub_info
        self.flag_first_state_updated = {}
        for sub_topic, sub_info in sub_info.items():
            sub_msg_type = sub_info['type']
            self.joint_mapping[sub_topic] = None
            self.create_subscription(
                JointState,
                sub_topic,
                partial(self.listener_callback, topic=sub_topic),
                10
            )
            self.flag_first_state_updated[sub_topic] = False
        self.lock = Lock()
        self.states = {
            'pos': np.zeros(len(output_joint_names)),
            'vel': np.zeros(len(output_joint_names)),
            'eff': np.zeros(len(output_joint_names))
        }

    def listener_callback(self, msg, topic):
        if self.joint_mapping[topic] is None:
            global mode_machine  # kind of not good implementation, but it works
            mode_machine = msg.mode_machine
            self.joint_mapping[topic] = []
            for id, name in enumerate(self.output_joint_names):
                if name in g1_joint_names:
                    self.joint_mapping[topic].append((g1_joint_mapping, id))
            self.joint_mapping[topic] = np.array(self.joint_mapping[topic])
        id1, id2 = self.joint_mapping[topic][:, 0], self.joint_mapping[topic][:, 1]
        pos, vel, effort = [], [], []
        for idx in range(len(id1)):
            joint_position = msg.motor_state[idx].q
            joint_velocity = msg.motor_state[idx].dq
            joint_effort = msg.motor_state[idx].tau_est
            pos.append(joint_position)
            vel.append(joint_velocity)
            effort.append(joint_effort)
        with self.lock:
            self.states['pos'][id2] = np.array(pos)
            self.states['vel'][id2] = np.array(vel)
            self.states['eff'][id2] = np.array(effort)
        self.flag_first_state_updated[topic] = True

class ControllerPublisher(Node):
    def __init__(self, pub_info, command_joint_names, joint_states, timer_period=0.02):
        super().__init__('controller_publisher')
        self.timer_period = timer_period
        self.pub_info = pub_info
        self.pubs = {}
        self.command_pos, self.command_vel, self.command_acc = None, None, None
        self.command_joint_names = command_joint_names
        self.joint_mapping = {}

        self.kp = 60
        self.kp_dict = {joint_idx: self.kp for joint_idx in g1_joint_mapping.values()}
        for waist_joint in [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]:
            self.kp_dict[waist_joint] = 200

        self.kd = 1.5
        self.kd_dict = {joint_idx: self.kd for joint_idx in g1_joint_mapping.values()}
        for waist_joint in [G1JointIndex.WaistYaw, G1JointIndex.WaistRoll, G1JointIndex.WaistPitch]:
            self.kd_dict[waist_joint] = 5

        # Preventing overheating
        wrist_joints = [
            G1JointIndex.LeftWristRoll, G1JointIndex.LeftWristYaw, G1JointIndex.LeftWristPitch,
            G1JointIndex.RightWristRoll, G1JointIndex.RightWristYaw, G1JointIndex.RightWristPitch
        ]
        for wrist_joint in wrist_joints:
            self.kp_dict[wrist_joint] = 10
            self.kd_dict[wrist_joint] = 0.5

        if len(self.command_joint_names) < (7 + 7 + 4 + 4):
            self.arm_sdk_on = 1 # arm sdk on
        else:
            self.arm_sdk_on = 0

        self.mode = 'direct_publish' # 'interpolate'
        self.interpolate_time_ = {}
        self.interpolate_duration = 3.0
        self.joint_states = joint_states
        for pub_topic, pub_info in pub_info.items():
            pub_msg_type = eval(pub_info['type'])
            self.joint_mapping[pub_topic] = []
            for id, name in enumerate(self.command_joint_names):
                if name in g1_joint_names:
                    self.joint_mapping[pub_topic].append((g1_joint_mapping[name], id))
            self.joint_mapping[pub_topic] = np.array(self.joint_mapping[pub_topic])
            self.pubs[pub_topic] = self.create_publisher(pub_msg_type, pub_topic, 10)
            self.interpolate_time_[pub_topic] = 0.0
            self.create_timer(timer_period, partial(self.publish, pub_topic))

    def publish(self, topic):
        if self.command_pos is not None:
            msg = LowCmd()
            msg.mode_machine = mode_machine
            if self.arm_sdk_on:
                msg.motor_cmd[G1JointIndex.kNotUsedJoint].q = self.arm_sdk_on
            ids1, ids2 = self.joint_mapping[topic][:, 0], self.joint_mapping[topic][:, 1]
            pos = self.command_pos.copy()
            if self.mode == 'direct_publish':
                for id1, id2 in zip(ids1, ids2):
                    msg.motor_cmd[id1].q = pos[id2]
                    msg.motor_cmd[id1].kp = self.kp_dict[id1]
                    msg.motor_cmd[id1].kd = self.kd_dict[id1]
                    msg.motor_cmd[id1].tau = 0
                    msg.motor_cmd[id1].dq = 0.0
            else:
                ratio = min(self.interpolate_time_[topic]/self.interpolate_duration, 1.0)
                for id1, id2 in zip(ids1, ids2):
                    msg.motor_cmd[id1].q = (1-ratio) * self.joint_states['pos'][id2] + ratio * pos[id2]
                    msg.motor_cmd[id1].kp = self.kp_dict[id1]
                    msg.motor_cmd[id1].kd = self.kd_dict[id1]
                    msg.motor_cmd[id1].tau = 0
                    msg.motor_cmd[id1].dq = 0.0
            msg.crc = C.Crc(msg)
            self.pubs[topic].publish(msg)
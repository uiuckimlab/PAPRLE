from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rclpy.parameter import Parameter, ParameterType
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from functools import partial
from threading import Thread, Lock
import numpy as np
import rclpy

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
            self.joint_mapping[topic] = []
            for id, name in enumerate(msg.name):
                if name in self.output_joint_names:
                    self.joint_mapping[topic].append((id, self.output_joint_names.index(name)))
            self.joint_mapping[topic] = np.array(self.joint_mapping[topic])
        with self.lock:
            id1, id2 = self.joint_mapping[topic][:, 0], self.joint_mapping[topic][:, 1]
            self.states['pos'][id2] = np.array(msg.position)[id1]
            self.states['vel'][id2] = np.array(msg.velocity)[id1]
            self.states['eff'][id2] = np.array(msg.effort)[id1]
        self.flag_first_state_updated[topic] = True


class ControllerPublisher(Node):
    def __init__(self, pub_info, command_joint_names, joint_states, timer_period=0.02):
        super().__init__('controller_publisher')

        self.cli = self.create_client(GetParameters, '/controller_manager/get_parameters')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().debug('Waiting for parameter service... controller_manager_get_parameters')
        self.req = GetParameters.Request()
        self.req.names = ['use_sim_time']
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            values = future.result().values
            for v in values:
                self.get_logger().info(f'Value: {v.string_value}')
        else:
            self.get_logger().error('Failed to call service')
        self.set_parameters([Parameter('use_sim_time', value=values[0].bool_value)])

        self.timer_period = timer_period
        self.pub_info = pub_info
        self.pubs = {}
        self.command_pos, self.command_vel, self.command_acc = None, None, None
        self.command_joint_names = command_joint_names
        self.joint_mapping = {}
        self.duration_msg = rclpy.duration.Duration(seconds=timer_period).to_msg()
        self.mode = 'direct_publish' # 'interpolate'
        self.interpolate_time_ = {}
        self.interpolate_duration = 3.0
        self.joint_states = joint_states
        for pub_topic, pub_info in pub_info.items():
            pub_msg_type = eval(pub_info['type'])
            self.joint_mapping[pub_topic] = [command_joint_names.index(name) for name in pub_info['joint_names']]
            self.pubs[pub_topic] = self.create_publisher(pub_msg_type, pub_topic, 10)
            self.interpolate_time_[pub_topic] = 0.0
            self.create_timer(timer_period, partial(self.publish, pub_topic))

    def publish(self, topic):
        if self.command_pos is not None:
            msg = JointTrajectory()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = self.pub_info[topic]['joint_names']
            mapping_inds = self.joint_mapping[topic]
            if self.mode == 'direct_publish':
                pos = self.command_pos[mapping_inds]
                vel = [] if self.command_vel is None else self.command_vel[mapping_inds]
                acc = [] if self.command_acc is None else self.command_acc[mapping_inds]
            else:
                ratio = min(self.interpolate_time_[topic]/self.interpolate_duration, 1.0)
                pos = (1-ratio) * self.joint_states['pos'][mapping_inds] + ratio * self.command_pos[mapping_inds]
                vel = [0.0] * len(mapping_inds)
                acc = [0.0] * len(mapping_inds)
                self.interpolate_time_[topic] += self.timer_period
            msg.points = [JointTrajectoryPoint(positions=pos, velocities=vel, accelerations=acc,
                                               time_from_start=self.duration_msg)]
            self.pubs[topic].publish(msg)
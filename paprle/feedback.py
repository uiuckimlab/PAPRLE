import sys
from paprle.utils.misc import detect_ros_version, import_pinocchio
ROS_VERSION = detect_ros_version()
if ROS_VERSION == "ROS1":
    import rospy
elif ROS_VERSION == "ROS2":
    import rclpy
    from rclpy.node import Node
else:
    raise ImportError("Unknown ROS version. Please check your environment.")
pin = import_pinocchio()
from sensor_msgs.msg import JointState
from teleop_msgs.srv import SetBasePose
from teleop_msgs.msg import EEFTransforms
from std_msgs.msg import Float64MultiArray
import time
import numpy as np

class Feedback():
    def __init__(self, robot, leader, teleoperator, env):
        self.robot = robot
        self.leader = leader
        self.teleoperator = teleoperator
        self.env = env

        self.follower_hand_inds  = []
        for limb_name in self.robot.limb_names:
            if limb_name in self.robot.ctrl_hand_joint_idx_mapping:
                self.follower_hand_inds.extend(self.robot.ctrl_hand_joint_idx_mapping[limb_name])


        self.use_ros = False
        if leader.leader_config.type == 'puppeteer':
            self.use_ros = True
            self.leader_joint_names = leader.leader_robot.joint_names
            self.follower_joint_names = robot.joint_names
            self.joint_reach_eps = 0.5
            self.joint_max_th = self.robot.joint_limits[:,1] - self.joint_reach_eps
            self.joint_min_th = self.robot.joint_limits[:,0] + self.joint_reach_eps
            self.joint_limit_feedback_enabled = np.ones(len(self.leader_joint_names), dtype=bool)
            self.joint_limit_feedback_enabled[leader.hand_ids] = False  # disable hand joints feedback

            if ROS_VERSION == 'ROS1':
                rospy.init_node('feedback_node', anonymous=True)
                self.pub = rospy.Publisher('/feedback', JointState, queue_size=10)
                self.rate = rospy.Rate(10)
            elif ROS_VERSION == 'ROS2':
                try: rclpy.init()
                except: pass
                self.node = Node('feedback_node')
                self.pub_joint = self.node.create_publisher(JointState, "/leaders/followers_joint_feedback", 10)

                self.pub_joint_limits = self.node.create_publisher(JointState, "/leaders/feedback_joint_limits", 10)
                self.pub_not_following = self.node.create_publisher(JointState, "/leaders/feedback_not_following", 10)

                self.pub_eef = self.node.create_publisher(EEFTransforms, '/leaders/followers_eef_feedback', 10)
                self.rate = self.node.create_rate(10)

                self.set_base_pose_cli = self.node.create_client(SetBasePose, '/leaders/set_base_pos')
                while not self.set_base_pose_cli.wait_for_service(timeout_sec=1.0):
                    self.node.get_logger().info('Waiting for SetBasePos service...')
                self.req = SetBasePose.Request()
                self.req.joint_state = JointState()
                self.req.joint_state.name = self.leader_joint_names
                init_pose = self.leader.leader_robot.init_qpos
                init_pose[self.leader.hand_ids] = (init_pose[self.leader.hand_ids] - self.leader.hand_offsets)/ self.leader.hand_scales
                self.req.joint_state.position = init_pose.tolist()
                future = self.set_base_pose_cli.call_async(self.req)
        else:
            print("Feedback is not used for this leader type for now.")
        return

    def send_feedback(self):
        if self.leader.leader_config.type == 'puppeteer':
            empty_msg = JointState()
            empty_msg.header.stamp = self.node.get_clock().now().to_msg()
            empty_msg.name = self.follower_joint_names
            empty_msg.position = np.zeros(len(self.follower_joint_names)).tolist()
            initial_phase, dt = 0.0, 0.005
            while True:
                qpos = self.teleoperator.last_target_qpos
                if qpos is None or not self.leader.is_ready or not self.env.initialized:
                    initial_phase = 0
                    time.sleep(0.1)
                    self.pub_not_following.publish(empty_msg)
                    continue
                elif self.env.initialized and self.leader.is_ready and initial_phase < 1.0:
                    initial_phase += dt
                    time.sleep(0.01)
                    self.pub_not_following.publish(empty_msg)
                    continue

                if self.leader.direct_joint_mapping:
                    joint_limit_reaching_feedback = np.zeros(len(qpos))
                    exceed_max = self.joint_max_th - qpos
                    joint_limit_reaching_feedback[exceed_max < 0] = exceed_max[exceed_max < 0]  # minus signal to reduce the joint position
                    exceed_min = self.joint_min_th - qpos
                    joint_limit_reaching_feedback[exceed_min > 0] = exceed_min[exceed_min > 0]  # plus signal to increase the joint position
                    joint_limit_reaching_feedback = joint_limit_reaching_feedback * self.joint_limit_feedback_enabled

                    leader_command = self.leader.last_command
                    if leader_command is not None:
                        # scale leader_command first
                        follower_qpos = self.env.get_current_qpos()
                        leader_qpos  = leader_command
                        # leader_qpos[hand_inds] = (leader_qpos[hand_inds] - self.leader.hand_offsets)/self.leader.hand_scales
                        not_following_feedback = follower_qpos - leader_qpos
                        not_following_feedback[self.leader.hand_inds] = np.clip(not_following_feedback[self.leader.hand_inds], -0.5, 0.0)
                    else:
                        not_following_feedback = np.zeros(len(qpos))

                    msg = JointState()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.name = self.leader_joint_names
                    msg.position = joint_limit_reaching_feedback.tolist()
                    self.pub_joint_limits.publish(msg)

                    msg = JointState()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.name = self.follower_joint_names
                    msg.position = not_following_feedback.tolist()
                    self.pub_not_following.publish(msg)

                    # # Visualization
                    # if self.teleoperator.viz is not None:
                    #     self.teleoperator.viz.logs = {
                    #         'joint_limit': {self.joint_names[int(i)]: joint_limit_reaching_feedback[int(i)] for i in range(len(qpos)) if joint_limit_reaching_feedback[int(i)] != 0},
                    #         'not_following': {self.follower_joint_names[int(i)]: not_following_feedback[int(i)] for i in range(len(qpos)) if not_following_feedback[int(i)] != 0},
                    #     }

                    total_joint_feedback = joint_limit_reaching_feedback + not_following_feedback
                    msg = JointState()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.name = self.leader_joint_names
                    msg.position = total_joint_feedback.tolist()
                    self.pub_joint.publish(msg)
                else:
                    #
                    # calculate eef pose feedback - leader will solve ik for the joint positions
                    # 1. Get current
                    joint_limit_reaching_feedback = np.zeros(len(qpos))
                    follower_qpos = self.env.get_current_qpos()
                    msg = None
                    if self.teleoperator.target_ee_poses is not None:
                        msg = EEFTransforms()
                        for limb_name in self.robot.limb_names:
                            pin_solver = self.teleoperator.ik_solvers[limb_name]
                            arm_qpos = follower_qpos[self.robot.robot_config.ctrl_arm_joint_idx_mapping[limb_name]]
                            pin_qpos = pin.neutral(pin_solver.model)
                            pin_qpos[pin_solver.idx_mapping] = arm_qpos
                            pin.forwardKinematics(pin_solver.model, pin_solver.data, pin_qpos)
                            curr_pose: pin.SE3 = pin.updateFramePlacement(pin_solver.model, pin_solver.data,
                                                                          pin_solver.ee_frame_id)
                            # curr_pose = pin_solver.base_pose.actInv(curr_pose)
                            tt = self.teleoperator.target_ee_poses[limb_name]
                            pos, quat = tt[0:3], tt[3:7]
                            xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
                            pose_vec = np.concatenate([pos, xyzw])
                            target_pose = pin.XYZQUATToSE3(pose_vec)

                            err = -pin.log(curr_pose.actInv(target_pose)).vector

                            err_msg = Float64MultiArray()
                            err_msg.data = err.tolist()
                            msg.eef_transforms.append(err_msg)
                            msg.frame_names.append(limb_name)
                        msg.header.stamp = self.node.get_clock().now().to_msg()

                        gripper_feedback = np.zeros(len(self.leader_joint_names))
                        follower_hand_pos = follower_qpos[self.follower_hand_inds]
                        leader_hand_pos = self.leader.leader_states[self.leader.hand_ids]
                        diff = ((follower_hand_pos - leader_hand_pos)) / self.leader.hand_scales
                        diff = np.clip(diff, -0.5, 0.0)
                        gripper_feedback[self.leader.hand_ids] = diff.flatten()
                        gripper_msg = JointState()
                        gripper_msg.header.stamp = self.node.get_clock().now().to_msg()
                        gripper_msg.name = self.leader_joint_names
                        gripper_msg.position = gripper_feedback.tolist()
                        self.pub_joint.publish(gripper_msg)

                    if msg is not None:
                        self.pub_eef.publish(msg)

                time.sleep(0.01)




        return
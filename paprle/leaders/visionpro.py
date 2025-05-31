import grpc
from avp_stream.grpc_msg import *
from threading import Thread
from avp_stream.utils.grpc_utils import *
import time
import pickle
import numpy as np
from avp_stream import VisionProStreamer
import copy
from pytransform3d import transformations as pt
from pytransform3d import rotations, coordinates
from threading import Thread, Lock
YUP2ZUP = np.array([[[1, 0, 0, 0],
                    [0, 0, -1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]]], dtype = np.float64)

OPERATOR2AVP_RIGHT = np.array(
    [
        [-1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
)

OPERATOR2AVP_LEFT = np.array(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)
def two_mat_batch_mul(batch_mat: np.ndarray, left_rot: np.ndarray):
    result = np.tile(np.eye(4), [batch_mat.shape[0], 1, 1])
    result[:, :3, :3] = np.matmul(left_rot[None, ...], batch_mat[:, :3, :3])
    result[:, :3, 3] = batch_mat[:, :3, 3] @ left_rot.T
    return result

def three_mat_mul(left_rot: np.ndarray, mat: np.ndarray, right_rot: np.ndarray):
    result = np.eye(4)
    rotation = left_rot @ mat[:3, :3] @ right_rot
    pos = left_rot @ mat[:3, 3]
    result[:3, :3] = rotation
    result[:3, 3] = pos
    return result

def project_average_rotation(quat_list: np.ndarray):
    gravity_dir = np.array([0, 0, -1])

    last_quat = quat_list[-1, :]
    last_mat = rotations.matrix_from_quaternion(last_quat)
    gravity_quantity = gravity_dir @ last_mat  # (3, )
    max_gravity_axis = np.argmax(np.abs(gravity_quantity))
    same_direction = gravity_quantity[max_gravity_axis] > 0

    next_axis = (max_gravity_axis + 1) % 3
    next_next_axis = (max_gravity_axis + 2) % 3
    angles = []
    for i in range(quat_list.shape[0]):
        next_dir = rotations.matrix_from_quaternion(quat_list[i])[:3, next_axis]
        next_dir[2] = 0  # Projection to non gravity direction
        next_dir_angle = coordinates.spherical_from_cartesian(next_dir)[2]
        angles.append(next_dir_angle)

    angle = np.mean(angles)
    final_mat = np.zeros([3, 3])
    final_mat[:3, max_gravity_axis] = gravity_dir * same_direction
    final_mat[:3, next_axis] = [np.cos(angle), np.sin(angle), 0]
    final_mat[:3, next_next_axis] = np.cross(
        final_mat[:3, max_gravity_axis], final_mat[:3, next_axis]
    )
    return rotations.quaternion_from_matrix(final_mat, strict_check=False)


class LPFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False
class LPRotationFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.is_init = False

        self.y = None

    def next(self, x: np.ndarray):
        assert x.shape == (4,)

        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()

        self.y = rotations.quaternion_slerp(self.y, x, self.alpha, shortest_path=True)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False
class CustomVisionProStreamer(VisionProStreamer):
    def __init__(self, ip, record=False, replay=None):
        self.ip = ip
        self.record = record
        self.recording = []
        self.latest = None
        self.axis_transform = YUP2ZUP[0][:3,:3]
        self.shutdown = False
        self.stream_running = False
        self.transform_lock = Lock()
        if replay is not None:
            with open(replay, "rb") as f:
                self.replay = pickle.load(f)
            print("Replay loaded from", replay)
        else:
            self.replay = None



    def replay_stream(self):
        while not self.shutdown and self.replay is not None:
            for id, transformations in enumerate(self.replay):
                if self.shutdown:
                    break
                new_trans = copy.deepcopy(transformations)
                new_trans['timestamp'] = time.time()
                self.latest = new_trans
                self.stream_running = True
                time.sleep(0.01)

    def stream(self):
        while not self.shutdown:
            self.stream_running = False
            request = handtracking_pb2.HandUpdate()
            try:
                with grpc.insecure_channel(f"{self.ip}:12345") as channel:
                    stub = handtracking_pb2_grpc.HandTrackingServiceStub(channel)
                    responses = stub.StreamHandUpdates(request)
                    for response in responses:
                        left_joints = process_matrices(
                            response.left_hand.skeleton.jointMatrices
                        )
                        right_joints = process_matrices(
                            response.right_hand.skeleton.jointMatrices
                        )
                        left_joints = two_mat_batch_mul(left_joints, OPERATOR2AVP_LEFT.T)
                        right_joints = two_mat_batch_mul(right_joints, OPERATOR2AVP_RIGHT.T)

                        transformations = {
                            "left_wrist": three_mat_mul(
                                self.axis_transform,
                                process_matrix(response.left_hand.wristMatrix)[0],
                                OPERATOR2AVP_LEFT,
                            ),
                            "right_wrist": three_mat_mul(
                                self.axis_transform,
                                process_matrix(response.right_hand.wristMatrix)[0],
                                OPERATOR2AVP_RIGHT,
                            ),
                            "left_fingers": left_joints,
                            "right_fingers": right_joints,
                            "head": rotate_head(
                                three_mat_mul(
                                    self.axis_transform,
                                    process_matrix(response.Head)[0],
                                    np.eye(3),
                                )
                            ),
                            "left_pinch_distance": get_pinch_distance(response.left_hand.skeleton.jointMatrices),
                            "right_pinch_distance": get_pinch_distance(response.right_hand.skeleton.jointMatrices),
                            "timestamp": time.time(),
                        }
                        if self.record:
                            self.recording.append(transformations)
                        with self.transform_lock:
                            self.latest = transformations
                        self.stream_running = True
            except Exception as e:
                pass
                #print(f"An error occurred: {e}")


            # with open("recording.pkl", "wb") as f:
            #     pickle.dump(self.recording, f)


from threading import Thread
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass

class VisionPro:
    def __init__(self, follower_robot, leader_config, env_config, render_mode='none', verbose=False, *args, **kwargs):
        self.follower_robot = follower_robot
        self.leader_config = leader_config
        self.env_config = env_config
        self.is_ready = False
        self.require_end = False
        self.shutdown = False
        self.motion_scale = self.leader_config.motion_scale
        self.left_finger_angles, self.right_finger_angles = None, None
        self.left_finger_good_for_init, self.right_finger_good_for_init = None, None
        self.left_pos_filter = LPFilter(0.1)
        self.right_pos_filter = LPFilter(0.1)
        self.left_rot_filter = LPRotationFilter(0.4)
        self.right_rot_filter = LPRotationFilter(0.4)

        self.streamer = CustomVisionProStreamer(leader_config.ip, record=False, replay='ep_recording.pkl')
        if self.streamer.replay is not None:
            self.streamer_thread = Thread(target=self.streamer.replay_stream)
        else:
            self.streamer_thread = Thread(target=self.streamer.stream)
        self.streamer_thread.start()

        self.viz_thread = Thread(target=self.__render_mujoco)
        self.viz_thread.start()

        self.leader_viz_info = {'color': 'blue',  'log': "VisionPro is ready!"}
        self.end_detection_thread = Thread(target=self.detect_end_signal)
        self.end_detection_thread.start()

    def __render_mujoco(self):
        self.viz = MuJoCoParserClass('hand', rel_xml_path='models/leaders/hand.xml', VERBOSE=True)
        self.viz.init_viewer()
        self.viz.update_viewer(
            azimuth=90.0,
            distance=0.73,
            elevation=-71,
            lookat=[0.066,0.033,1.214]
        )
        I3 = np.eye(3)
        while not self.shutdown:
            if self.streamer.latest is None or not self.streamer.stream_running:
                time.sleep(0.01)
                continue
            transform = copy.deepcopy(self.streamer.latest)
            head_pq = pt.pq_from_transform(transform['head'][0])
            right_wrist_pq = pt.pq_from_transform(transform['right_wrist'])
            left_wrist_pq = pt.pq_from_transform(transform['left_wrist'])
            transform['right_fingers'] = transform['right_wrist'] @ transform['right_fingers']
            transform['left_fingers'] = transform['left_wrist'] @ transform['left_fingers']
            right_finger_pq, left_finger_pq = [], []
            empty  = np.zeros(7)
            empty[4] = 1.0
            for i in range(25):
                try:
                    right_finger_pq.append(pt.pq_from_transform(transform['right_fingers'][i]))
                except:
                    right_finger_pq.append(empty)
                try:
                    left_finger_pq.append(pt.pq_from_transform(transform['left_fingers'][i]))
                except:
                    left_finger_pq.append(empty)


            whole_q = np.concatenate([
                head_pq, right_wrist_pq, left_wrist_pq,
                *right_finger_pq, *left_finger_pq
            ], axis=0)

            self.viz.forward(whole_q)

            if self.left_finger_angles is not None:
                flat_threshold = (0.01, np.deg2rad(15))
                tip_index = np.array([4, 9, 14, 19, 24])
                palm_bone_index = np.array([1, 5, 10, 15, 20])
                mask = (self.left_finger_angles  > flat_threshold[0])*(self.left_finger_angles < flat_threshold[1])
                for i, valid in enumerate(mask):
                    tip_pq = left_finger_pq[tip_index[i]]
                    palm_bone_pq = left_finger_pq[palm_bone_index[i]]
                    if valid:
                        self.viz.plot_arrow_fr2to(p_fr=palm_bone_pq[:3], p_to=tip_pq[:3], rgba=[0,1,0,1], r=0.005)
                    else:
                        self.viz.plot_arrow_fr2to(p_fr=palm_bone_pq[:3], p_to=tip_pq[:3], rgba=[1,0,0,1], r=0.005)
            if self.left_finger_good_for_init is not None:
                if self.left_finger_good_for_init:
                    self.viz.plot_T(p=left_wrist_pq[:3], R=I3, PLOT_SPHERE=0.005, PLOT_AXIS=False, label='Left Wrist (Good for Init)', sphere_rgba=[0,1,0,1])
                else:
                    self.viz.plot_T(p=left_wrist_pq[:3], R=I3, PLOT_SPHERE=0.005, PLOT_AXIS=False, label='Left Wrist (Bad for Init)', sphere_rgba=[1,0,0,1])

            if self.right_finger_angles is not None:
                flat_threshold = (0.01, np.deg2rad(15))
                tip_index = np.array([4, 9, 14, 19, 24])
                palm_bone_index = np.array([1, 5, 10, 15, 20])
                mask = (self.right_finger_angles > flat_threshold[0])*(self.right_finger_angles < flat_threshold[1])
                for i, valid in enumerate(mask):
                    tip_pq = right_finger_pq[tip_index[i]]
                    palm_bone_pq = right_finger_pq[palm_bone_index[i]]
                    if valid:
                        self.viz.plot_arrow_fr2to(p_fr=palm_bone_pq[:3], p_to=tip_pq[:3], rgba=[0,1,0,1], r=0.005)
                    else:
                        self.viz.plot_arrow_fr2to(p_fr=palm_bone_pq[:3], p_to=tip_pq[:3], rgba=[1,0,0,1], r=0.005)
            if self.right_finger_good_for_init is not None:
                if self.right_finger_good_for_init:
                    self.viz.plot_T(p=right_wrist_pq[:3], R=I3, PLOT_SPHERE=0.005, PLOT_AXIS=False, label='Right Wrist (Good for Init)', sphere_rgba=[0,1,0,1])
                else:
                    self.viz.plot_T(p=right_wrist_pq[:3], R=I3, PLOT_SPHERE=0.005, PLOT_AXIS=False, label='Right Wrist (Bad for Init)', sphere_rgba=[1,0,0,1])

            if np.linalg.norm(head_pq[:3] - right_wrist_pq[:3]) < 0.1:
                self.viz.plot_T(p=head_pq[:3],R=I3,PLOT_AXIS=False, label='Invalid Right Wrist')
            else:
                self.viz.plot_T(p=right_wrist_pq[:3], R=I3, PLOT_AXIS=False, label='Right Wrist')
            if np.linalg.norm(head_pq[:3] - left_wrist_pq[:3]) < 0.1:
                self.viz.plot_T(p=head_pq[:3] + np.array([0,0,0.02]),R=I3,PLOT_AXIS=False, label='Invalid Left Wrist')
            else:
                self.viz.plot_T(p=left_wrist_pq[:3], R=I3, PLOT_AXIS=False, label='Left Wrist')

            self.viz.plot_T(p=np.zeros(3), R=np.eye(3, 3), PLOT_AXIS=True, axis_len=0.5, axis_width=0.005)
            self.viz.render()
            time.sleep(0.01)


    def launch_init(self, initial_qpos):
        # launch detecting initialization process
        while (self.streamer.latest is None) or ((time.time() - self.streamer.latest['timestamp']) > 0.01):
            #print("Curr gripper values:", self.leader_states[self.hand_ids])
            print("Waiting for VisionPro Streaming....", end="\r")
            time.sleep(0.1)

        self.init_thread = Thread(target=self.initialize)
        self.init_thread.start()

        return

    def initialize(self):
        iteration, initialize_progress = 0.0, 0.0
        dt, threshold_time = 0.03, 3

        init_left_hand_wrist_pose_list, init_right_hand_wrist_pose_list = [], []
        while not initialize_progress >= threshold_time:
            if self.shutdown: return

            start_time = time.time()
            transform = copy.deepcopy(self.streamer.latest)
            left_hand_wrist_pose = transform['left_wrist']
            left_hand_joints = transform['left_fingers']
            right_hand_wrist_pose = transform['right_wrist']
            right_hand_joints = transform['right_fingers']

            if len(init_left_hand_wrist_pose_list) == 0:
                init_left_hand_wrist_pose_list.append(pt.pq_from_transform(left_hand_wrist_pose))
                init_right_hand_wrist_pose_list.append(pt.pq_from_transform(right_hand_wrist_pose))

            left_hand_continue_init, left_finger_angles = self._is_hand_pose_good_for_init(
                left_hand_wrist_pose[:3,3],
                left_hand_joints[:,:3,3],
                init_left_hand_wrist_pose_list[-1][:3],
            )
            right_hand_continue_init, right_finger_angles = self._is_hand_pose_good_for_init(
                right_hand_wrist_pose[:3,3],
                right_hand_joints[:,:3,3],
                init_right_hand_wrist_pose_list[-1][:3],
            )
            continue_init = left_hand_continue_init and right_hand_continue_init
            self.left_finger_angles = left_finger_angles
            self.right_finger_angles = right_finger_angles
            self.left_finger_good_for_init = left_hand_continue_init
            self.right_finger_good_for_init = right_hand_continue_init

            # Stop initialization process and clear data if not continue init
            if not continue_init:
                init_left_hand_wrist_pose_list.clear()
                init_right_hand_wrist_pose_list.clear()
                initialize_progress = 0.0
            else:
                initialize_progress += dt
                init_left_hand_wrist_pose_list.append(pt.pq_from_transform(left_hand_wrist_pose))
                init_right_hand_wrist_pose_list.append(pt.pq_from_transform(right_hand_wrist_pose))


            print(f"Initializing VisionPro... {initialize_progress:.2f}/{threshold_time:.2f}")
            self.leader_viz_info['color'] = 'blue'
            self.leader_viz_info['log'] = f"Make your hands flat to initialize.... {initialize_progress:.2f}/{threshold_time}"
            left_time = max(dt -  (time.time() - start_time), 0.0)
            time.sleep(left_time)
            iteration += 1

        print("VisionPro Initialization Complete!")
        self.leader_viz_info['color'] = 'red'
        self.leader_viz_info['log'] = 'Initializing Leader...'

        left_init_collect_data = np.stack(init_left_hand_wrist_pose_list)
        right_init_collect_data = np.stack(init_right_hand_wrist_pose_list)
        init_frame_pos = (
            left_init_collect_data[:, :3],
            right_init_collect_data[:, :3],
        )
        init_frame_quat = (
            left_init_collect_data[:, 3:7],
            right_init_collect_data[:, 3:7],
        )

        left_calibrated_pos, left_calibrated_rot = self._compute_init_frame(
            init_frame_pos[0], init_frame_quat[0]
        )
        right_calibrated_pos, right_calibrated_rot = self._compute_init_frame(
            init_frame_pos[1], init_frame_quat[1]
        )


        self.init_eef_poses = [
            pt.transform_from(left_calibrated_rot, left_calibrated_pos),
            pt.transform_from(right_calibrated_rot, right_calibrated_pos)
        ]
        self.leader_viz_info['color'] = 'green'
        self.leader_viz_info['log'] = 'Visionpro initialized successfully!'

        print("[VisionPro] VisionPro initialized successfully!")

        transform = copy.deepcopy(self.streamer.latest)
        self.last_left_hand_wrist_pose = transform['left_wrist']
        self.last_right_hand_wrist_pose = transform['right_wrist']
        self.last_left_hand_pq = pt.pq_from_transform(transform['left_wrist'])
        self.last_right_hand_pq = pt.pq_from_transform(transform['right_wrist'])
        local_poses, hand_poses = {}, {}
        for limb_id in range(2):
            leader_limb_name = 'left' if limb_id == 0 else 'right'
            follower_limb_name = self.leader_config.limb_mapping[leader_limb_name]
            local_poses[follower_limb_name] = np.eye(4)
            hand_poses[follower_limb_name] = 0.0
        self.last_command = (local_poses, hand_poses)

        self.left_finger_angles = None
        self.right_finger_angles = None
        self.left_finger_good_for_init = None
        self.right_finger_good_for_init = None
        self.is_ready = True
        self.require_end = False
        return

    def close_init(self):
        self.init_thread.join()
        return

    def detect_end_signal(self):
        dt = 0.03
        iteration = 0
        enough_to_end, threshold_time = 0.0, 3.0
        while not self.shutdown:
            if self.streamer.latest is None or not self.is_ready:
                time.sleep(dt)
                continue
            if self.streamer.stream_running == False or ((self.streamer.latest['timestamp'] - time.time()) > 0.1):
                self.require_end = True
                time.sleep(dt)
            start_time = time.time()
            left_pinch_distance = np.array([self.streamer.latest['left_pinch_distance']])
            right_pinch_distance = np.array([self.streamer.latest['right_pinch_distance']])
            hand_closed = (left_pinch_distance < 0.05) and (right_pinch_distance < 0.05)
            if hand_closed:
                enough_to_end += dt
                self.leader_viz_info['color'] = (enough_to_end * 255 / threshold_time, 255 - enough_to_end * 255 / threshold_time, 0)
                self.leader_viz_info['log'] = f"Running... End signal detected! {enough_to_end:.2f}/{threshold_time:.2f}"
                print(f"Running... End signal detected! {enough_to_end:.2f}/{threshold_time:.2f}", end="\r")
            else:
                if enough_to_end > 0.0:
                    print(f"Running... End signal detected! 0.0/{threshold_time:.2f}", end="\r")
                enough_to_end = 0.0
            if enough_to_end >= threshold_time:
                self.require_end = True
                self.leader_viz_info['color'] = 'red'
                self.leader_viz_info['log'] = "End signal detected! Resetting the leader and follower"
                self.is_ready = False
                enough_to_end = 0.0
                print("[VisionPro] End signal detected!, resetting the controller...")
            left_time = max(dt -  (time.time() - start_time), 0.0)
            time.sleep(left_time)
            iteration += 1

        return

    def get_status(self):

        if self.streamer.stream_running == False or ((self.streamer.latest['timestamp'] - time.time()) > 0.1):
            print("VisionPro is not streaming, please wait for the reset.")
            self.require_end = True
            self.is_ready = False
            return self.last_command

        transform = copy.deepcopy(self.streamer.latest)
        head_pose = transform['head'][0]
        left_hand_wrist_pose = transform['left_wrist']
        right_hand_wrist_pose = transform['right_wrist']
        if np.linalg.norm(head_pose[:3,3] - left_hand_wrist_pose[:3,3]) < 0.1:
            print("Warning: Possible invalid left wrist pose detected, using previous pose.")
            left_hand_wrist_pose = self.last_left_hand_wrist_pose
        if np.linalg.norm(head_pose[:3,3] - right_hand_wrist_pose[:3,3]) < 0.1:
            print("Warning: Possible invalid right wrist pose detected, using previous pose.")
            right_hand_wrist_pose = self.last_right_hand_wrist_pose

        left_hand_wrist_pq = pt.pq_from_transform(left_hand_wrist_pose)
        right_hand_wrist_pq = pt.pq_from_transform(right_hand_wrist_pose)

        left_hand_wrist_pq[:3] = self.left_pos_filter.next(left_hand_wrist_pq[:3])
        right_hand_wrist_pq[:3] = self.right_pos_filter.next(right_hand_wrist_pq[:3])
        left_hand_wrist_pq[3:] = self.left_rot_filter.next(left_hand_wrist_pq[3:])
        right_hand_wrist_pq[3:] = self.right_rot_filter.next(right_hand_wrist_pq[3:])
        left_hand_wrist_pose = pt.transform_from_pq(left_hand_wrist_pq)
        right_hand_wrist_pose = pt.transform_from_pq(right_hand_wrist_pq)

        if np.linalg.norm(left_hand_wrist_pq[:3] - self.last_left_hand_pq[:3]) > 0.05: # or np.linalg.norm(rotations.quaternion_diff(left_hand_wrist_pq[3:], self.last_left_hand_pq[3:])) > 0.5):
            print("Warning: Left hand wrist pose is moving too fast, using previous pose.")
            left_hand_wrist_pose = self.last_left_hand_wrist_pose
            left_hand_wrist_pq = self.last_left_hand_pq
        if np.linalg.norm(right_hand_wrist_pq[:3] - self.last_right_hand_pq[:3]) > 0.05: # or np.linalg.norm(rotations.quaternion_diff(right_hand_wrist_pq[3:], self.last_right_hand_pq[3:])) > 0.5:
            print("Warning: Right hand wrist pose is moving too fast, using previous pose.")
            right_hand_wrist_pose = self.last_right_hand_wrist_pose
            right_hand_wrist_pq = self.last_right_hand_pq

        # [TODO] output for dextrous hand
        left_hand_joints = transform['left_fingers']
        right_hand_joints = transform['right_fingers']

        left_pinch_distance = np.array([0.1 - self.streamer.latest['left_pinch_distance']]) / 0.1
        left_pinch_distance = np.clip(left_pinch_distance, 0.0, 1.0)
        right_pinch_distance = np.array([0.1 - self.streamer.latest['right_pinch_distance']]) / 0.1
        right_pinch_distance = np.clip(right_pinch_distance, 0.0, 1.0)

        new_eef_poses = [left_hand_wrist_pose, right_hand_wrist_pose]
        local_poses, hand_poses = {}, {}
        for limb_id, (Rt, init_Rt) in enumerate(zip(new_eef_poses, self.init_eef_poses)):
            new_pos = init_Rt[:3, :3].T @ Rt[:3, 3] - init_Rt[:3, :3].T @ init_Rt[:3, 3]
            new_R = init_Rt[:3, :3].T @ Rt[:3, :3]
            new_pos = new_pos * self.motion_scale
            new_Rt = pt.transform_from(R=new_R, p=new_pos)
            leader_limb_name = 'left' if limb_id == 0 else 'right'
            follower_limb_name = self.leader_config.limb_mapping[leader_limb_name]
            local_poses[follower_limb_name] = new_Rt
            hand_poses[follower_limb_name] = left_pinch_distance if limb_id == 0 else right_pinch_distance


        self.last_left_hand_wrist_pose = left_hand_wrist_pose
        self.last_right_hand_wrist_pose = right_hand_wrist_pose
        self.last_left_hand_pq = left_hand_wrist_pq
        self.last_right_hand_pq = right_hand_wrist_pq
        self.last_command = (local_poses, hand_poses)
        return (local_poses, hand_poses)

    def _is_hand_pose_good_for_init(
        self,
        hand_wrist_pose: np.ndarray,
        hand_joints: np.ndarray,
        last_hand_wrist_pose: np.ndarray,
    ):
        # Check whether the hand wrist is moving during initialization
        not_far_away_threshold = 0.05
        hand_not_far_away = (
            np.linalg.norm(hand_wrist_pose[:3] - last_hand_wrist_pose[:3])
            < not_far_away_threshold
        )

        # Check whether the fingers are in a flatten way
        flat_threshold = (0.01, np.deg2rad(15))
        finger_angles = self._compute_hand_joint_angles(hand_joints)

        hand_flat_spread = (
            flat_threshold[0] < np.mean(finger_angles) < flat_threshold[1]
        )

        return hand_not_far_away and hand_flat_spread, finger_angles

    @staticmethod
    def _compute_hand_joint_angles(joints: np.ndarray):
        tip_index = np.array([4, 9, 14, 19, 24])
        palm_bone_index = np.array([1, 5, 10, 15, 20])
        root = joints[0:1, :]
        tips = joints[tip_index]
        palm_bone = joints[palm_bone_index]
        tip_vec = tips - root
        tip_vec = tip_vec / np.linalg.norm(tip_vec, axis=1, keepdims=True)
        palm_vec = palm_bone - root
        palm_vec = palm_vec / np.linalg.norm(palm_vec, axis=1, keepdims=True)
        angle = np.arccos(np.clip(np.sum(tip_vec * palm_vec, axis=1), -1.0, 1.0))
        return angle

    def _compute_init_frame(
        self, init_frame_pos: np.ndarray, init_frame_quat: np.ndarray
    ):
        num_data = init_frame_pos.shape[0]
        weight = (np.arange(num_data) + 1) / np.sum(np.arange(num_data) + 1)
        calibrated_quat = project_average_rotation(init_frame_quat)
        calibrated_wrist_pos = np.sum(weight[:, None] * init_frame_pos, axis=0).astype(np.float32)
        calibrated_wrist_rot = rotations.matrix_from_quaternion(calibrated_quat)

        return calibrated_wrist_pos, calibrated_wrist_rot

    def update_vis_info(self, env_vis_info):
        if env_vis_info is not None:
            env_vis_info['leader'] = self.leader_viz_info
        return env_vis_info

    def close(self):
        self.streamer.shutdown = True
        self.streamer_thread.join()
        self.shutdown = True
        self.viz_thread.join()
        self.end_detection_thread.join()
        if self.init_thread is not None:
            self.init_thread.join()
        print("VisionPro closed successfully!")
        return

if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.follower import Robot
    import time

    follower_config, leader_config, env_config = BaseConfig().parse()
    robot = Robot(follower_config)
    leader = VisionPro(robot, leader_config, env_config, render_mode='human')

    while True:
        time.sleep(0.01)

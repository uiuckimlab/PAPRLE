import re
import mujoco
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass, init_ik_info,add_ik_info,get_dq_from_ik_info
from paprle.envs.mujoco_env_utils.util import np_uv
import numpy as np
from collections import defaultdict
from yourdfpy.urdf import URDF

class MujocoCollisionChecker:
    def __init__(self, robot, render=False, verbose=False):
        self.robot = robot

        self.env = MuJoCoParserClass(self.robot.name, rel_xml_path=robot.xml_path, VERBOSE=verbose)

        self.sim_joint_names = self.env.joint_names
        self.ctrl_joint_idxs, self.mimic_joints_info = self.robot.set_joint_idx_mapping(self.sim_joint_names)

        self.joint_names_for_ik = self.env.rev_joint_names
        self.collision_push_len = 0.01
        self.gap_per_step_cm = 0.01
        self.max_gap_cm = 2.0
        self.acc_limit = 100
        self.max_ik_tick = 1
        self.body_names_to_exclude = self.robot.robot_config.ignore_collision_pairs

    def get_collision_free_pose(self, qpos, verbose=False):
        query_qpos = np.zeros_like(self.env.data.qpos)
        query_qpos[self.ctrl_joint_idxs] = qpos
        if len(self.mimic_joints_info) > 0:
            inds1, inds2 = self.mimic_joints_info[:, 0].astype(np.int32), self.mimic_joints_info[:, 1].astype(np.int32)
            query_qpos[inds1] = query_qpos[inds2] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]

        ch_step = 0
        collision_gap_cm = self.gap_per_step_cm * ch_step
        if collision_gap_cm > self.max_gap_cm:
            collision_gap_cm = self.max_gap_cm
        self.env.model.geom_gap = collision_gap_cm / 100
        self.env.model.geom_margin = collision_gap_cm / 100
        self.env.forward(q=query_qpos)

        collision_log = ''
        for ik_tick in range(self.max_ik_tick):
            #print("ik_tick", ik_tick)
            p_contact_list, f_contact_list = [], []
            contact_body1_list, contact_body2_list = [], []
            hand_contact_list = []
            hand_contact_body1_list, hand_contact_body2_list = [], []
            for c_idx in range(self.env.data.ncon):
                contact = self.env.data.contact[c_idx]
                p_contact = contact.pos
                R_frame = contact.frame.reshape((3, 3))

                f_contact_local = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.env.model, self.env.data, 0, f_contact_local)
                f_contact = R_frame @ f_contact_local[:3]
                f_contact = np_uv(f_contact)

                contact_body1 = self.env.body_names[self.env.model.geom_bodyid[contact.geom1]]
                contact_body2 = self.env.body_names[self.env.model.geom_bodyid[contact.geom2]]

                # check if the contact is with the body to exclude
                print("Contact body1: ", contact_body1, "Contact body2: ", contact_body2)
                if len(self.body_names_to_exclude) > 0:
                    ignore_this_pair = False
                    for body_pair in self.body_names_to_exclude:
                        b1, b2 = body_pair
                        if (re.match(b1, contact_body1) and re.match(b2, contact_body2)) or (re.match(b1, contact_body2) and re.match(b2, contact_body1)):
                            ignore_this_pair = True
                            break
                    if ignore_this_pair: continue
                else: pass

                # find out whether the contact is within a hand or not
                if self.is_inhand_collision(contact_body1, contact_body2):
                    hand_contact_list.append(p_contact)
                    hand_contact_body1_list.append(contact_body1)
                    hand_contact_body2_list.append(contact_body2)
                else:
                    p_contact_list.append(p_contact)
                    f_contact_list.append(f_contact)
                    contact_body1_list.append(contact_body1)
                    contact_body2_list.append(contact_body2)

            if len(hand_contact_list) > 0:
                success, qpos = self.resolve_inhand_collision_psyonic(hand_contact_list, hand_contact_body1_list, hand_contact_body2_list, p_contact_list)
            else:
                qpos = self.env.data.qpos.copy()
            self.env.forward(q=qpos)
            ik_info_collision = init_ik_info()
            n_contact = len(p_contact_list)

        qpos = qpos[self.ctrl_joint_idxs]
        if n_contact > 0:
            collision_log += f'Collision detected. {n_contact} contact points. '
            if verbose:
                return None, collision_log
            else:
                return None
        else:
            collision_log += 'No collision detected. '
            if verbose:
                return qpos, collision_log
            else:
                return qpos

    def is_inhand_collision(self, contact_body1, contact_body2):
        if self.robot.robot_config.eef_type is None: return False
        if 'ability_hand' in self.robot.robot_config.eef_type:
            if '/' not in contact_body1 or '/' not in contact_body2:
                return False
            r1_name, r1_link = contact_body1.split('/')
            r2_name, r2_link = contact_body2.split('/')
            # for the ability hand, collision only happens with thumbs
            if (r1_name == r2_name) and (('thumb' in r1_link) or ('thumb' in r2_link)):
                return True
        else:
            # Probably not in hand - for parallel gripper
            return False

    def resolve_inhand_collision_psyonic(self, contact_list, contact_body1_list, contact_body2_list, body_contact_list):
        if not hasattr(self, 'idx_thumb_q1s'):
            self.thumb_q1_names = [name for name in self.env.joint_names if 'thumb_q1' in name]
            self.idx_thumb_q1s = [self.joint_names.index(joint) for joint in self.thumb_q1_names]
            self.idx_thumb_q2s = [self.joint_names.index(joint.replace('q1', 'q2')) for joint in self.thumb_q1_names]

        qpos = self.env.data.qpos.copy()
        success = False
        for ii in range(100):
            collision_dict = defaultdict(lambda:0)
            for c_idx in range(len(contact_list)):
                contact_body1, contact_body2 = contact_body1_list[c_idx], contact_body2_list[c_idx]
                if 'thumb' in contact_body1:
                    collision_dict[contact_body1] += 1
                if 'thumb' in contact_body2:
                    collision_dict[contact_body2] += 1

            for thumb_name, count in collision_dict.items():
                if count > 1:
                    thumb_idx = self.thumb_q1_names.index(thumb_name)
                    qpos[self.idx_thumb_q1s[thumb_idx]] += 0.1
                    qpos[self.idx_thumb_q2s[thumb_idx]] -= 0.1
            qpos = np.clip(qpos, self.env.joint_ranges[:, 0], self.env.joint_ranges[:, 1])
            self.env.forward(q=qpos)
            new_contact_list = []
            for c_idx in range(self.env.data.ncon):
                contact = self.env.data.contact[c_idx]
                contact_body1 = self.env.body_names[self.env.model.geom_bodyid[contact.geom1]]
                contact_body2 = self.env.body_names[self.env.model.geom_bodyid[contact.geom2]]
                if (contact_body1 == self.body_name_to_exclude) or (contact_body2 == self.body_name_to_exclude):
                    continue
                new_contact_list.append((contact_body1, contact_body2))
            if len(new_contact_list) == len(body_contact_list):
                success = True
                break


        return success, qpos

    def get_joint_limits(self):
        all_joint_limits = []
        for joint in self.ctrl_joint_names:
            idx = self.joint_names.index(joint)
            joint_limits = self.env.joint_ranges[idx]
            all_joint_limits.append(joint_limits)
        all_joint_limits = np.array(all_joint_limits).transpose(1,0)
        return all_joint_limits

    def get_transform(self, link_to, link_from):
        from_p, from_R = self.env.get_pR_body(link_from)
        from_Rt = np.eye(4)
        from_Rt[:3, :3] = from_R
        from_Rt[:3, 3] = from_p

        to_p, to_R = self.env.get_pR_body(link_to)
        to_Rt = np.eye(4)
        to_Rt[:3, :3] = to_R
        to_Rt[:3, 3] = to_p

        Rt = np.linalg.inv(from_Rt) @ to_Rt
        return Rt

import numpy as np

class PowerGripper:
    def __init__(self, robot, leader_config, env_config, retargeting_config, joint_limits=None, **kwargs):
        self.robot = robot
        self.leader_config = leader_config
        self.env_config = env_config
        self.retargeting_config = retargeting_config
        self.joint_limits = joint_limits

        self.other_dof = len(self.retargeting_config.others)
        self.thumb_joint_mapping, self.thumb_joint_limits = [], []
        for id, joint in enumerate(self.retargeting_config.thumb):
            joint_name = getattr(self.retargeting_config, 'prefix', '') + '/' + joint
            joint_id = self.robot.ctrl_joint_names.index(joint_name)
            self.thumb_joint_mapping.append([id, joint_id])
            if joint_limits is not None: self.thumb_joint_limits.append(joint_limits[:,joint_id])

        self.others_joint_mapping, self.others_joint_limits = [], []
        for id, joint in enumerate(self.retargeting_config.others):
            joint_name = getattr(self.retargeting_config, 'prefix', '') + '/' + joint
            joint_id = self.robot.ctrl_joint_names.index(joint_name)
            self.others_joint_mapping.append([id, joint_id])
            if joint_limits is not None: self.others_joint_limits.append(joint_limits[:, joint_id])

        self.thumb_joint_mapping, self.others_joint_mapping = np.array(self.thumb_joint_mapping), np.array(self.others_joint_mapping)
        self.thumb_joint_limits, self.others_joint_limits = np.array(self.thumb_joint_limits), np.array(self.others_joint_limits)
        min_joint_id = np.concatenate([self.others_joint_mapping[:, 1],self.thumb_joint_mapping[:, 1]]).min()
        self.others_joint_mapping[:, 1] -= min_joint_id
        self.thumb_joint_mapping[:, 1] -= min_joint_id
        self.hand_dof = self.robot.hand_dof
        return

    def solve(self, hand_command):
        others, thumb_command = hand_command
        out_command = np.zeros(self.hand_dof)
        if others == 'open':
            out_command[self.others_joint_mapping[:, 1]] = self.others_joint_limits[:,0]
        elif others == 'close':
            out_command[self.others_joint_mapping[:, 1]] = self.others_joint_limits[:,1]
        out_command[self.thumb_joint_mapping[:, 1]] = np.array(thumb_command)[self.thumb_joint_mapping[:, 0]]
        return out_command

    def reset(self):
        return
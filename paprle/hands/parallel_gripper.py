
import numpy as np
class ParallelGripper:
    def __init__(self, robot, leader_config, env_config, *args, **kwargs):
        self.robot = robot
        self.leader_config = leader_config
        self.env_config = env_config
        return

    def solve(self, hand_command):
        if isinstance(hand_command, np.ndarray):
            out_command = hand_command
        elif hand_command == 'open':
            out_command = [0.0]
        elif hand_command == 'close':
            out_command = [1.0]
        return out_command

    def reset(self):
        return
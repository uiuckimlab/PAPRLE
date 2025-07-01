import numpy as np

class BaseEnv:
    def __init__(self, robot, leader_config, env_config, verbose=False, render_mode=False, **kwargs):
        self.robot = robot
        self.leader_config = leader_config
        self.env_config = env_config

        self.verbose = verbose
        self.render_mode = render_mode

        self.initialized = False
        self.view_im = None

        self.num_joints = 10 # Number of total joints
        self.vis_info = {}
        return

    def close(self) -> None:
        """
            Close the environment.
            If there is any background thread or process, it should be terminated here.
        """
        return NotImplementedError

    def reset(self) -> np.ndarray:
        """
            Reset the environment to the initial state, and return the initial position of the robot.
            Returns:
                init_env_qpos: initial position of the robot
        """
        init_env_qpos = np.zeros(10)
        return init_env_qpos

    def initialize(self, initial_qpos: np.ndarray) -> None:
        """
            Initialize the environment with the given initial position.
            Args:
                initial_qpos: initial position of the robot
        """
        return NotImplementedError

    def step(self, qpos: np.ndarray) -> tuple:
        """
            Step the environment with the given qpos
            Args:
                qpos: action to be taken
            Returns:
                obs: observation after taking the action
                rew: reward received after taking the action
                done: whether the episode is done or not
                info: additional information
        """
        obs = np.zeros(10)
        rew = 0.0
        done = False
        info = {}
        return obs, rew, done, info



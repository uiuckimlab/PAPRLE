import warnings

ENV_DICT = {}
try:
    from paprle.envs.mujoco_env import MujocoEnv

    ENV_DICT['mujoco'] = MujocoEnv
except Exception as e:
    warnings.warn('Mujoco not found, or some error happened while importing MujocoEnv')
    warnings.warn("error:" + str(e))

try:
    from paprle.envs.isaacgym_env import IsaacGymEnv
    ENV_DICT['isaacgym'] = IsaacGymEnv
except Exception as e:
    warnings.warn('IsaacGym not found, or some error happened while importing IsaacGymEnv')
    warnings.warn("error:" + str(e))

from paprle.envs.ros_env import ROSEnv
ENV_DICT['ros'] = ROSEnv

try:
    from paprle.envs.ros_env import ROSEnv
    ENV_DICT['ros'] = ROSEnv
except Exception as e:

    warnings.warn('ros1 not found, or some error happened while importing ROS1Env')
    warnings.warn("error:" + str(e))

try:
    from paprle.envs.ros2_env import ROS2Env
    ENV_DICT['ros2'] = ROS2Env
except Exception as e:
    warnings.warn('ros2 not found, or some error happened while importing ROS2Env')
    warnings.warn("error:" + str(e))


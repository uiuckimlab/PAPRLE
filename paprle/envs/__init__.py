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
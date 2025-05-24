import warnings

ENV_DICT = {}
try:
    from paprle.envs.mujoco_env import MujocoEnv

    ENV_DICT['mujoco'] = MujocoEnv
except Exception as e:
    warnings.warn('Mujoco not found, or some error happened while importing MujocoEnv')
    warnings.warn("error:" + str(e))

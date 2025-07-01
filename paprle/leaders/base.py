# Template for the Leader class
class Leader:
    def __init__(self, robot, leader_config, env_config, render_mode='human', verbose=False, *args, **kwargs):
        self.is_ready = False
        self.require_end = False
        self.shutdown = False
        return

    def reset(self, ):
        return

    def launch_init(self, init_env_qpos):
        return

    def close_init(self):
        return

    def get_status(self):
        raise NotImplementedError

    def update_vis_info(self, env_vis_info):
        return env_vis_info

    def close(self):
        return

import argparse
from argparse import RawTextHelpFormatter
import os
from omegaconf import OmegaConf
from paprle.utils.config_utils import add_info_robot_config, sanity_check_leader_config, change_working_directory

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseConfig:
    def __init__(self):
        change_working_directory()

    def parse(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
        parser.add_argument('--follower','-f', type=str, default='papras_6dof')
        parser.add_argument('--leader', '-l', type=str, default='sliders')
        parser.add_argument('--env', '-e', type=str, default='mujoco')

        parser.add_argument('--render-teleop', action='store_true', default=False)
        parser.add_argument('--render-env', action='store_true', default=False)

        parser.add_argument('--off-collision', action='store_true', default=False)
        parser.add_argument('--off-feedback', action='store_true', default=False)

        leader_list = os.listdir('configs/leader') + os.listdir('configs/leader/puppeteers/') + os.listdir('configs/leader/sim_puppeteers/')
        # Help
        parser.add_argument('--help', action='help',
                            help=f'Possible options for Followers are: {[k.replace(".yaml", "") for k in os.listdir("configs/follower")]}\n'
                                 f'Possible options for Leaders are: {[k.replace(".yaml", "") for k in leader_list]}\n'
                                 f'Possible options for Environments are: {[k.replace(".yaml", "") for k in os.listdir("configs/env")]}')
        args, unknown = parser.parse_known_args()

        follower_config_file = f'configs/follower/{args.follower}.yaml'
        if not os.path.exists(follower_config_file):
            raise FileNotFoundError(f"Follower config file {follower_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml','') for k in os.listdir('configs/follower')]}")
        if args.leader.startswith('puppeteer'):
            leader_config_file = f'configs/leader/puppeteers/{args.leader}.yaml'
        elif args.leader.startswith('sim_puppeteer'):
            leader_config_file = f'configs/leader/sim_puppeteers/{args.leader}.yaml'
        else:
            leader_config_file = f'configs/leader/{args.leader}.yaml'
        if not os.path.exists(leader_config_file):
            raise FileNotFoundError(f"Device config file {leader_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml','') for k in leader_list]}")
        env_config_file = f'configs/env/{args.env}.yaml'
        if not os.path.exists(env_config_file):
            raise FileNotFoundError(f"Env config file {env_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml', '') for k in os.listdir('configs/env')]}")

        print("-------Configuration-------")
        print(f"🎮 Leader: {args.leader}")
        print(f"🤖 Follower: {args.follower}")
        print(f"🌏 Env: {args.env}")
        print("---------------------------")

        override_config = OmegaConf.load(follower_config_file)
        cli_config = OmegaConf.from_dotlist(unknown)
        self.follower_config = OmegaConf.merge( override_config, cli_config)
        self.follower_config.robot_cfg = add_info_robot_config(self.follower_config.robot_cfg)

        override_config = OmegaConf.load(leader_config_file)
        cli_config = OmegaConf.from_dotlist(unknown)
        self.leader_config = OmegaConf.merge(override_config, cli_config)
        self.leader_config = sanity_check_leader_config(self.leader_config, self.follower_config)

        override_config = OmegaConf.load(env_config_file)
        cli_config = OmegaConf.from_dotlist(unknown)
        self.env_config = OmegaConf.merge(override_config, cli_config)

        # add additional config
        self.env_config.render_teleop = args.render_teleop
        self.env_config.render_env = args.render_env
        self.env_config.off_collision = args.off_collision
        self.env_config.off_feedback = args.off_feedback

        return self.follower_config, self.leader_config, self.env_config


if __name__ == '__main__':
    BaseConfig().parse()
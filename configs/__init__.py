import argparse
from argparse import RawTextHelpFormatter
import os
from omegaconf import OmegaConf
from paprle.utils.config_utils import add_info_robot_config, sanity_check_leader_config

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseConfig:
    def __init__(self):
        # Change working directory to the root of the project
        is_root_dir = 'configs' in os.listdir()
        if not is_root_dir:
            os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

    def parse(self, verbose=True):
        parser = argparse.ArgumentParser(add_help=False, formatter_class=RawTextHelpFormatter)
        parser.add_argument('--follower','-f', type=str, default='papras_6dof')
        parser.add_argument('--leader', '-l', type=str, default='keyboard')
        parser.add_argument('--env', '-e', type=str, default='mujoco')

        parser.add_argument('--render-teleop', action='store_true', default=False)
        parser.add_argument('--render-env', action='store_true', default=False)

        # Help
        parser.add_argument('--help', action='help',
                            help=f'Possible options for Followers are: {[k.replace(".yaml", "") for k in os.listdir("configs/follower")]}\n'
                                 f'Possible options for Leaders are: {[k.replace(".yaml", "") for k in os.listdir("configs/leader")]}\n'
                                 f'Possible options for Environments are: {[k.replace(".yaml", "") for k in os.listdir("configs/env")]}')
        args, unknown = parser.parse_known_args()

        follower_config_file = f'configs/follower/{args.follower}.yaml'
        if not os.path.exists(follower_config_file):
            raise FileNotFoundError(f"Follower config file {follower_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml','') for k in os.listdir('configs/follower')]}")
        leader_config_file = f'configs/leader/{args.leader}.yaml'
        if not os.path.exists(leader_config_file):
            raise FileNotFoundError(f"Device config file {leader_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml','') for k in os.listdir('configs/leader')]}")
        env_config_file = f'configs/env/{args.env}.yaml'
        if not os.path.exists(env_config_file):
            raise FileNotFoundError(f"Env config file {env_config_file} does not exist. \n"
                                    f"Possible options are {[k.replace('.yaml', '') for k in os.listdir('configs/env')]}")

        print("-------Configuration-------")
        print(f"🎮 Leader: {args.leader}")
        print(f"🤖 Follower: {args.follower}")
        print(f"🌏 Env: {args.env}")
        print("---------------------------")

        override_config = OmegaConf.load(f'configs/follower/{args.follower}.yaml')
        cli_config = OmegaConf.from_dotlist(unknown)
        self.follower_config = OmegaConf.merge( override_config, cli_config)
        self.follower_config.robot_cfg = add_info_robot_config(self.follower_config.robot_cfg)

        override_config = OmegaConf.load(f'configs/leader/{args.leader}.yaml')
        cli_config = OmegaConf.from_dotlist(unknown)
        self.leader_config = OmegaConf.merge(override_config, cli_config)
        self.leader_config = sanity_check_leader_config(self.leader_config, self.follower_config)

        override_config = OmegaConf.load(f'configs/env/{args.env}.yaml')
        cli_config = OmegaConf.from_dotlist(unknown)
        self.env_config = OmegaConf.merge(override_config, cli_config)

        return self.follower_config, self.leader_config, self.env_config


if __name__ == '__main__':
    BaseConfig().parse()
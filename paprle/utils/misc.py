import time
import os
import pickle

def make_episode(robot_config, device_config, env_config, folder_name='demo_data'):
    episode_name = f'F_{robot_config.robot_cfg.name}_L_{device_config.name}_{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    save_dir = f'{folder_name}/' + episode_name
    os.makedirs(save_dir, exist_ok=True)
    episode_info = {'episode_name': episode_name, 'robot_config': robot_config, 'device_config': device_config,
                    'env_config': env_config}
    with open(f'{save_dir}/episode_info.pkl', 'wb') as f:
        pickle.dump(episode_info, f)
    return save_dir


def detect_ros_version():
    ros_distro = os.environ.get("ROS_DISTRO", "")
    if "ROS_VERSION" in os.environ:
        if os.environ["ROS_VERSION"] == "1":
            return "ROS1"
        elif os.environ["ROS_VERSION"] == "2":
            return "ROS2"
    elif ros_distro:
        ros1_distros = ['melodic', 'noetic', 'kinetic']
        ros2_distros = ['foxy', 'galactic', 'humble', 'iron', 'rolling']
        if ros_distro in ros1_distros:
            return "ROS1"
        elif ros_distro in ros2_distros:
            return "ROS2"
    return "Unknown"

def import_pinocchio():
    # get python version
    import sys
    py = sys.version_info

    if py.major == 3 and py.minor == 10:
        removed_p = []
        for p in sys.path:
            if 'noetic' in p:
                sys.path.remove(p)
                removed_p.append(p)
    elif py.major == 3 and py.minor == 8:
        removed_p = []
        for p in sys.path:
            if 'humble' in p:
                sys.path.remove(p)
                removed_p.append(p)
        sys.path.append("/usr/lib/python3/dist-packages")
        sys.path.append("/opt/ros/noetic/lib/python3.8/site-packages/")

    try:
        import pinocchio as pin
    except ImportError:
        raise

    sys.path.extend(removed_p)
    return pin

import_pinocchio()
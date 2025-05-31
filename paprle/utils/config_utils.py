import os
def change_working_directory():
    """
    Change working directory to the root of the project
    """
    is_root_dir = '.PROJECT_ROOT' in os.listdir()
    if not is_root_dir:
        os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../../")
        print(f"Changed working directory to {os.getcwd()}")

def add_info_robot_config(robot_config, verbose=True):
    """
    Add additional information to robot_config
    """

    # ctrl_joint_names and ctrl_joint_idx_mapping
    ctrl_joint_names, ctrl_joint_idx_mapping, ctrl_joint_type = [], {}, {}
    ctrl_arm_joint_idx_mapping, ctrl_hand_joint_idx_mapping = {}, {}
    arm_dof_info, hand_dof_info = {}, {}
    for limb_name, limb_joint_names in robot_config.limb_joint_names.items():
        # Arm joints
        ctrl_joint_names.extend(limb_joint_names)
        num_joint = len(limb_joint_names)
        ctrl_joint_type[limb_name] = [0] * num_joint
        arm_dof_info[limb_name] = len(limb_joint_names)

        # Hand joints
        ctrl_joint_names.extend(robot_config.hand_joint_names[limb_name])
        num_joint += len(robot_config.hand_joint_names[limb_name])
        ctrl_joint_type[limb_name].extend([1] * len(robot_config.hand_joint_names[limb_name]))
        hand_dof_info[limb_name] = len(robot_config.hand_joint_names[limb_name])

        # Add to ctrl_joint_idx_mapping
        ctrl_joint_idx_mapping[limb_name] = list(range(len(ctrl_joint_names) - num_joint, len(ctrl_joint_names)))
        ctrl_arm_joint_idx_mapping[limb_name] = list(range(len(ctrl_joint_names) - num_joint, len(ctrl_joint_names) - len(robot_config.hand_joint_names[limb_name])))
        ctrl_hand_joint_idx_mapping[limb_name] = list(range(len(ctrl_joint_names) - len(robot_config.hand_joint_names[limb_name]), len(ctrl_joint_names)))

    robot_config.ctrl_joint_names = ctrl_joint_names
    robot_config.ctrl_joint_idx_mapping = ctrl_joint_idx_mapping
    robot_config.ctrl_joint_type = ctrl_joint_type

    robot_config.ctrl_arm_joint_idx_mapping = ctrl_arm_joint_idx_mapping
    robot_config.ctrl_hand_joint_idx_mapping = ctrl_hand_joint_idx_mapping

    robot_config.num_limbs = len(robot_config.limb_joint_names)
    robot_config.arm_dof = arm_dof_info
    robot_config.hand_dof = hand_dof_info
    robot_config.total_arm_dof = sum(arm_dof_info.values())
    robot_config.total_hand_dof = sum(hand_dof_info.values())

    if verbose:
        print("Added Information to robot_config:")
        print("- robot_config.ctrl_joint_names: ", robot_config.ctrl_joint_names)
        print("- robot_config.ctrl_joint_idx_mapping: ", robot_config.ctrl_joint_idx_mapping)
        print("- robot_config.ctrl_joint_type: ", robot_config.ctrl_joint_type)
        print("- robot_config.num_limbs: ", robot_config.num_limbs)
        print("- robot_config.arm_dof: ", robot_config.arm_dof)
        print("- robot_config.hand_dof: ", robot_config.hand_dof)

    print("---------------------------")
    return robot_config


def sanity_check_leader_config(leader_config, robot_config, verbose=True):
    if leader_config.output_type not in ['joint_pos', 'delta_eef_pose']:
        raise ValueError(f"Invalid output type: {leader_config.output_type}. Must be 'joint_pos' or 'delta_eef_pose'.")

    robot_name = robot_config.robot_cfg.name

    if 'puppeteer' in leader_config.type:
        if len(leader_config.direct_mapping) == 0:
            leader_config.direct_mapping = {}
            for leader_limb_name in leader_config.limb_joint_names.keys():
                leader_config.direct_mapping[leader_limb_name] = leader_limb_name

        direct_mapping_possible = robot_name in leader_config.direct_mapping_available_robots
        if direct_mapping_possible:
            leader_config.limb_mapping = {}
            for leader_limb_name in leader_config.limb_joint_names.keys():
                follower_limb_name = leader_config.direct_mapping[leader_limb_name]
                if follower_limb_name not in robot_config.robot_cfg.limb_joint_names:
                    raise ValueError(f"Trying to directly map {leader_limb_name} to follower robot, but {follower_limb_name} is not in robot_config.limb_joint_names.")
                leader_config.limb_mapping[leader_limb_name] = follower_limb_name
        elif not direct_mapping_possible:
            if leader_config.output_type == 'joint_pos':
                print("Warning: Direct joint mapping is not available for this robot. Changing to eef mapping.")
                leader_config.output_type = 'delta_eef_pose'
                print("- leader_config.output_type: ", leader_config.output_type)

            leader_limb_names = list(leader_config.limb_joint_names.keys())
            follower_limb_names = list(robot_config.robot_cfg.limb_joint_names.keys())
            leader_config.limb_mapping = {}
            for leader_limb_name, follower_limb_name in zip(leader_limb_names, follower_limb_names):
                leader_config.limb_mapping[leader_limb_name] = follower_limb_name
            # answer = input(f"Trying to map limbs {leader_limb_names} to {leader_config.limb_mapping.values()}. Do you want to change the mapping? [y/n]: ")
            # if answer.lower() == 'y':
            #     print(f"Please enter the mapping, Available follower limbs are {follower_limb_names}")
            #     for leader_limb_name in leader_limb_names:
            #         while True:
            #             answer = input(f"Enter the follower limb name for {leader_limb_name}: ")
            #             if answer not in follower_limb_names:
            #                 print(f"Invalid limb name: {answer}. Available names are {follower_limb_names}.")
            #                 continue
            #             else:
            #                 leader_config.limb_mapping[leader_limb_name] = answer
            #                 print(f"Mapping {leader_limb_name} to {answer}")
            #                 break
            # leader_config.limb_mapping['robot1'] = 'right_arm'
            # leader_config.limb_mapping['robot2'] = 'left_arm'

            print("Mapping completed.")
            print("- leader_config.limb_mapping: ", leader_config.limb_mapping)
    elif 'visionpro' in leader_config.type:
        if leader_config.output_type != 'delta_eef_pose':
            raise ValueError(f"Invalid output type for VisionPro leader: {leader_config.output_type}. Must be 'delta_eef_pose'.")
        robot_name = robot_config.robot_cfg.name
        leader_config.limb_mapping = {'left': '', 'right': ''}
        for id, limb_name in enumerate(robot_config.robot_cfg.limb_joint_names.keys()):
            if id == 1:
                leader_config.limb_mapping['left'] = limb_name
            elif id == 0:
                leader_config.limb_mapping['right'] = limb_name
        print("Mapping completed.")
        print("- leader_config.limb_mapping: ", leader_config.limb_mapping)
        if robot_name == 'g1':
            leader_config.limb_mapping['left'] = 'left_arm'
            leader_config.limb_mapping['right'] = 'right_arm'



    # [TODO] add more sanity check for leader_config

    return leader_config
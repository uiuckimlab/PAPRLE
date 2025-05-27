import numpy as np
from paprle.envs.base_env import BaseEnv
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass
import mujoco



# mujoco==3.1.6 mujoco_python_viewer==0.1.4
# referring to https://github.com/sjchoi86/yet-another-mujoco-tutorial-v3/
class MujocoEnv(BaseEnv):
    def __init__(self, robot, device_config, env_config, verbose=False, render_mode=False, **kwargs):
        super().__init__(robot, device_config, env_config, verbose=verbose, render_mode=render_mode, **kwargs)

        self.sim = MuJoCoParserClass(self.robot.name, rel_xml_path=robot.xml_path, VERBOSE=verbose)

        self.simulate_steps = getattr(env_config, 'simulate', False)
        self.dt = robot.control_dt
        self.HZ = 1/self.dt

        self.ctrl_mode = getattr(env_config, 'ctrl_mode', 'P')
        self.kp = getattr(env_config, 'kp', 1000)
        self.kd = getattr(env_config, 'kd', 2 * np.sqrt(self.kp) * 0.7)
        mass_matrix = np.ndarray(shape=(len(self.sim.data.qvel), len(self.sim.data.qvel)), dtype=np.float64, order='C')
        mujoco.mj_fullM(self.sim.model, mass_matrix, self.sim.data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(self.sim.data.qvel), len(self.sim.data.qvel)))
        self.mass_matrix = mass_matrix[self.sim.ctrl_joint_idxs, :][:, self.sim.ctrl_joint_idxs]

        self.mujoco_nstep = int(self.sim.HZ // self.HZ)
        self.VERBOSE = verbose
        self.tick = 0

        self.name = self.sim.name
        self.state_prev = self.get_state()
        self.action_pref = self.sample_action()

        self.state_dim = len(self.state_prev)
        self.obs_dim = len(self.get_observation())
        self.action_dim = self.sim.n_ctrl

        self.base_mats = getattr(self.robot, 'base_pose', None)
        self.num_eef, self.eef_names = self.robot.num_limbs, self.robot.eef_names

        self.sim_joint_names = self.sim.ctrl_joint_names if self.simulate_steps else self.sim.joint_names
        self.ctrl_joint_idxs, self.mimic_joints_info = self.robot.set_joint_idx_mapping(self.sim_joint_names)

        self.render_mode = render_mode
        self.vis_info = None
        if self.render_mode:
            self.sim.init_viewer(
                viewer_title=self.robot.name, viewer_width=1200, viewer_height=800,
                viewer_hide_menus=True,
            )
            self.viewer_args = self.robot.robot_config.viewer_args.mujoco
            self.sim.update_viewer(
                azimuth=self.viewer_args.azimuth, distance=self.viewer_args.distance,
                elevation=self.viewer_args.elevation, lookat=self.viewer_args.lookat,
                VIS_TRANSPARENT=False,
            )

        self.initialized = True
        return

    def initialize(self, initial_qpos: np.ndarray) -> None:
        self.set_qpos(initial_qpos)
        return

    def reset(self):
        self.sim.reset()
        self.set_qpos(self.robot.init_qpos)

        if self.render_mode:
            self.sim.update_viewer(
                azimuth=self.viewer_args.azimuth, distance=self.viewer_args.distance,
                elevation=self.viewer_args.elevation, lookat=self.viewer_args.lookat,
                VIS_TRANSPARENT=False,
                )

        self.ee_target_mats = None
        return self.sim.data.qpos[self.ctrl_joint_idxs]

    def step(self, action):
        self.tick = self.tick + 1
        #self.set_qpos(action)
        new_qpos = np.zeros(self.dof)
        new_qpos[self.ctrl_joint_idxs] = action
        if len(self.mimic_joints_info):
            new_qpos[self.mimic_joints_info[:, 0].astype(np.int32)] = new_qpos[self.mimic_joints_info[:, 1].astype(np.int32)] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]

        self.curr_qpos = new_qpos
        if self.simulate_steps:
            if self.ctrl_mode == 'P':
                self.set_position_target(new_qpos)
            else:
                self.sim.step(ctrl=action, nstep=self.mujoco_nstep)
        else:
            self.set_qpos(action)
        if self.render_mode:
            self.render()
        obs = self.get_observation()
        rew = self.get_reward()
        done = self.get_done()
        info = self.get_info()
        return obs, rew, done, info

    def set_position_target(self, qpos_target):
        for i in range(self.mujoco_nstep):
            curr_qpos = self.sim.data.qpos[self.sim.ctrl_qpos_idxs]
            curr_qvel = self.sim.data.qvel[self.sim.ctrl_qvel_idxs]
            pos_err = qpos_target - curr_qpos
            vel_err = -curr_qvel
            torque = (self.kp * pos_err + self.kd * vel_err)
            torques = np.dot(self.mass_matrix, torque) + self.sim.data.qfrc_bias[self.sim.ctrl_qpos_idxs]
            self.sim.step(ctrl=torques, nstep=1, INCREASE_TICK=False)

        #self.sim.step()
        self.sim.tick = self.sim.tick + 1

    def get_state(self):
        qpos = self.sim.data.qpos[self.sim.ctrl_qpos_idxs] # joint position
        qvel = self.sim.data.qvel[self.sim.ctrl_qvel_idxs] # joint velocity
        # Contact information
        contact_info = np.zeros(self.sim.n_sensor)
        #contact_idxs = np.where(self.sim.get_sensor_values(sensor_names=self.sim.sensor_names) > 0.2)[0]
        #contact_info[contact_idxs] = 1.0 # 1 means contact occurred
        # Concatenate information
        state = np.concatenate([
            qpos,
            qvel/10.0, # scale
            contact_info
        ])
        return state

    def get_state_as_command(self):
        qpos = self.sim.data.qpos[self.ctrl_joint_idxs]
        return qpos

    def get_observation(self):
        return self.get_state()

    def get_reward(self):
        return

    def get_done(self):
        return

    def get_info(self):
        return {}

    def sample_action(self):
        """
            Sample action (8)
        """
        a_min = self.sim.ctrl_ranges[:, 0]
        a_max = self.sim.ctrl_ranges[:, 1]
        action = a_min + (a_max - a_min) * np.random.rand(len(a_min))
        return action

    @property
    def dof(self):
        return self.sim.n_dof

    def set_qpos(self, qpos):
        new_qpos = np.zeros(self.dof)
        new_qpos[self.ctrl_joint_idxs] = qpos
        if len(self.mimic_joints_info):
            new_qpos[self.mimic_joints_info[:, 0].astype(np.int32)] = new_qpos[self.mimic_joints_info[:, 1].astype(np.int32)] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]
        self.curr_qpos = new_qpos
        self.sim.forward(q=new_qpos, INCREASE_TICK=True)
        self.sim.data.qvel[:] = 0.0

    def close(self):
        if self.render_mode:
            self.sim.close_viewer()
        self.sim = None
        return

    def render(self, mode='human'):
        self.sim.plot_T(p=np.zeros(3),R=np.eye(3,3), PLOT_AXIS=True,axis_len=0.5,axis_width=0.005)
        self.sim.plot_T(p=np.array([0,0,0.5]),R=np.eye(3),PLOT_AXIS=False, label='Tick:[%d]'%(self.sim.tick))
        self.sim.plot_joint_axis(axis_len=0.02,axis_r=0.004,joint_names=self.sim_joint_names) # joint axis
        self.sim.plot_contact_info(h_arrow=0.3,rgba_arrow=[1,0,0,1],PRINT_CONTACT_BODY=True) # contact
        self.sim.render()
        # print(
        #     f"azimuth: {self.sim.viewer.cam.azimuth}\n"
        #     f"distance: {self.sim.viewer.cam.distance}\n"
        #     f"elevation: {self.sim.viewer.cam.elevation}\n"
        #     f"lookat: {self.sim.viewer.cam.lookat.tolist()}")


if __name__ == '__main__':
    from configs import BaseConfig
    from paprle.utils.config_utils import change_working_directory
    change_working_directory()
    from paprle.follower import Robot

    robot_config, device_config, env_config = BaseConfig().parse()
    robot = Robot(robot_config)
    env = MujocoEnv(robot, device_config, env_config, verbose=False, render_mode='mujoco')
    obs = env.reset()

    min_qpos = robot.joint_limits[:, 0] + 0.3
    max_qpos = robot.joint_limits[:, 1] - 0.3
    interpolate_trajectory = np.linspace(min_qpos, max_qpos, num=1000)
    while True:
        env.initialize(interpolate_trajectory[0])
        for i in range(100):
            action = interpolate_trajectory[i]
            obs, rew, done, info = env.step(action)
            if done:
                break
import numpy as np
from pytransform3d import transformations as pt
from yourdfpy.urdf import URDF
from pytransform3d import rotations
from paprle.envs.mujoco_env_utils.mujoco_parser import MuJoCoParserClass, init_ik_info
class MujocoViz:
    def  __init__(self, robot, env=None, verbose=False):
        self.robot = robot
        self.viewer_args = robot.robot_config.viewer_args

        if env is None:
            self.env = MuJoCoParserClass(self.robot.name, rel_xml_path=robot.xml_path, VERBOSE=verbose)
        else:
            # don't recreate the env if we already have one
            self.env = env

        if 'base_pose' in self.robot.robot_config:
            self.base_mats = {limb_name: pt.transform_from_pq(pq) for limb_name, pq in self.robot.robot_config.base_pose.items()}
        else:
            self.base_mats = None

        self.sim_joint_names = self.env.joint_names
        self.ctrl_joint_idxs, self.mimic_joints_info = self.robot.set_joint_idx_mapping(self.sim_joint_names)
        self.eef_names = self.robot.robot_config.end_effector_link
        self.dof = self.env.n_dof

        self.ik_errs  = None
        self.ee_target_mats = None

        self.logs = {}

    def init_viewer(self, *args, **kwargs):
        return self.env.init_viewer(*args, **kwargs)

    def update_viewer(self, *args, **kwargs):
        return self.env.update_viewer(*args, **kwargs)

    def reset(self):
        self.env.reset()
        self.env.update_viewer(
            azimuth=self.viewer_args.azimuth, distance=self.viewer_args.distance,
            elevation=self.viewer_args.elevation, lookat=self.viewer_args.lookat,
            VIS_TRANSPARENT=True,
            )
        self.ee_target_mats = None
        return


    def set_ee_target(self, ee_targets, ik_errs=None):
        self.ee_target_mats = {}
        for limb_name, ee_target in ee_targets.items():
            ee_target_mat = pt.transform_from_pq(ee_target)
            self.ee_target_mats[limb_name] = ee_target_mat
        self.ik_errs = ik_errs

    def set_qpos(self, qpos):
        new_qpos = np.zeros_like(self.env.data.qpos)
        new_qpos[self.ctrl_joint_idxs] = qpos
        if len(self.mimic_joints_info) > 0:
            ind1, ind2 = self.mimic_joints_info[:, 0].astype(np.int32), self.mimic_joints_info[:, 1].astype(np.int32)
            new_qpos[ind1] = new_qpos[ind2] * self.mimic_joints_info[:, 2] + self.mimic_joints_info[:, 3]
        self.curr_qpos = new_qpos
        self.env.forward(q=new_qpos, INCREASE_TICK=True)

    def render(self):
        if self.ee_target_mats is not None:
            for limb_name in self.robot.limb_names:
                if limb_name not in  self.ee_target_mats: continue
                base_mat, ee_target_mat = self.base_mats[limb_name], self.ee_target_mats[limb_name]
                base_pose = self.robot.robot_config.base_pose[limb_name]
                ee_pos = rotations.q_prod_vector(base_pose[3:], ee_target_mat[:3, 3]) + base_pose[:3]
                ee_rot = rotations.concatenate_quaternions(base_pose[3:], rotations.quaternion_from_matrix(ee_target_mat[:3,:3]))
                ee_rot = rotations.matrix_from_quaternion(ee_rot)
                self.env.plot_T(ee_pos, ee_rot, PLOT_AXIS=True, PLOT_SPHERE=True, sphere_r=0.02, axis_len=0.12, axis_width=0.005)
                self.p_from = ee_pos
                p_to = self.env.get_p_body(self.eef_names[limb_name])
                self.actual_err = np.linalg.norm(self.p_from - p_to)
                self.env.plot_line_fr2to(self.p_from, p_to, label=f"err: {self.actual_err:.3f}")

        self.env.plot_T(p=np.zeros(3),R=np.eye(3,3), PLOT_AXIS=True,axis_len=0.5,axis_width=0.005)
        self.env.plot_T(p=np.array([0,0,0.5]),R=np.eye(3),PLOT_AXIS=False, label='Tick:[%d]'%(self.env.tick))
        #self.env.plot_joint_axis(axis_len=0.02,axis_r=0.004,joint_names=self.joint_names) # joint axis
        self.env.plot_contact_info(h_arrow=0.3,rgba_arrow=[1,0,0,1],PRINT_CONTACT_BODY=True) # contact

        if len(self.logs) > 0:
            # feedback specific visualization
            for feedback_name, feedbacks in self.logs.items():
                for i, (joint_name, feedback_val) in enumerate(feedbacks.items()):
                    p_joint, R_joint = self.env.get_pR_joint(joint_name=joint_name)
                    self.env.plot_T(p=p_joint, R=np.eye(3), PLOT_AXIS=False, label=f"{joint_name}/{feedback_name}: %.3f" % feedback_val)


        self.env.render()
        return

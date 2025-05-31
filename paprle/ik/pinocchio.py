import os, sys
from paprle.ik.base import BaseIKSolver
from paprle.utils.misc import import_pinocchio
pin = import_pinocchio()
from typing import List, Optional, Dict
import numpy as np

class PinocchioIKSolver(BaseIKSolver):
    def __init__(self, robot_name, ik_config):
        self.robot_name = robot_name
        self.config = ik_config

        self.ik_damping = ik_config.ik_damping
        self.ik_eps = ik_config.eps
        self.dt = ik_config.dt
        self.ee_name = ik_config.ee_link
        self.base_name = ik_config.base_link
        self.repeat = getattr(ik_config, 'repeat', 1)

        urdf_path = os.path.abspath(ik_config.urdf_path)
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(str(urdf_path),
                                                                                      package_dirs=[ik_config.asset_dir])
        self.data: pin.Data = self.model.createData()
        self.collision_data = self.collision_model.createData()

        frame_mapping: Dict[str, int] = {}
        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        if self.ee_name not in frame_mapping:
            raise ValueError(
                f"End effector name {self.ee_name} not find in robot with path: {urdf_path}."
            )

        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[self.ee_name]
        self.base_frame_id = frame_mapping[self.base_name] if self.base_name in frame_mapping else 0

        # Current state
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        self.base_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.base_frame_id
        )
        self.ee_pose = self.base_pose.actInv(self.ee_pose)
        self.ik_err = 0.0

        joint_names = self.get_joint_names()
        self.idx_mapping = [joint_names.index(name) for name in self.config.joint_names]

        # joint_limits
        self.lower_limit = np.array(self.model.lowerPositionLimit)
        self.upper_limit = np.array(self.model.upperPositionLimit)

    def reset(self):
        self.qpos = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, self.qpos)
        self.ee_pose = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        self.base_pose: pin.SE3 = pin.updateFramePlacement(
            self.model, self.data, self.base_frame_id
        )
        self.ee_pose = self.base_pose.actInv(self.ee_pose)
        self.ik_err = 0.0

        return

    def solve(self, pos, quat):
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        pose_vec = np.concatenate([pos, xyzw])
        oMdes = pin.XYZQUATToSE3(pose_vec)
        oMdes = self.base_pose.act(oMdes)
        qpos = self.qpos.copy()
        candidates = []
        for r in range(self.repeat):
            for k in range(30):
                pin.forwardKinematics(self.model, self.data, qpos)
                ee_pose = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
                J = pin.computeFrameJacobian(self.model, self.data, qpos, self.ee_frame_id)
                iMd = ee_pose.actInv(oMdes)
                err = pin.log(iMd).vector
                if np.linalg.norm(err) < self.ik_eps:
                    break

                v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
                qpos = pin.integrate(self.model, qpos, v * self.dt)

            candidates.append((err, qpos))
            #print("found ik solution with iter ", k)
        if len(candidates) > 1:
            candidates = sorted(candidates, key=lambda x: np.linalg.norm(x[0]))
            err, qpos = candidates[0]
        self.set_current_qpos(qpos, oMdes=oMdes)
        return qpos[self.idx_mapping]

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        xyzw_pose = pin.SE3ToXYZQUAT(oMf)
        return np.concatenate(
            [
                xyzw_pose[:3],
                np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
            ]
        )

    def get_current_qpos(self) -> np.ndarray:
        return self.qpos.copy()

    def set_current_qpos(self, qpos: np.ndarray, oMdes=None):
        self.qpos = qpos
        pin.forwardKinematics(self.model, self.data, self.qpos)
        #self.base_pose: pin.SE3 = pin.updateFramePlacement(
        #    self.model, self.data, self.base_frame_id
        #)
        self.ee_pose = pin.updateFramePlacement(
            self.model, self.data, self.ee_frame_id
        )
        self.ee_pose = self.base_pose.actInv(self.ee_pose)#self.ee_pose.act(self.base_pose.inverse())
        if oMdes is not None:
            iMd = self.ee_pose.actInv(oMdes)
            err = np.linalg.norm(pin.log(iMd).vector)
            self.ik_err = err

    def get_ee_name(self) -> str:
        return self.ee_name

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:

        try:
            # Pinocchio by default add a dummy joint name called "universe"
            names = list(self.model.names)
            return names[1:]
        except:
            names = []
            for f in self.model.frames:
                if f.type == pin.JOINT:
                    names.append(f.name)
            return names


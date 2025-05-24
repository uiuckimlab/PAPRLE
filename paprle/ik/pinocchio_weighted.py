import numpy as np
from paprle.ik.pinocchio import pin, PinocchioIKSolver


class PinocchioWeightedIKSolver(PinocchioIKSolver):
    def solve(self, pos, quat):
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        pose_vec = np.concatenate([pos, xyzw])
        oMdes = pin.XYZQUATToSE3(pose_vec)
        oMdes = self.base_pose.act(oMdes)
        qpos = self.qpos.copy()
        candidates = []
        for r in range(self.repeat):
            if r > 0:
                qpos = pin.neutral(self.model)

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
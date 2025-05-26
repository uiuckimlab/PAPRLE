## When setting up a new robot
- [ ] Add named poses

## When setting up a new leader
- [ ] Write about communication recommendations
- [ ] Write about how the motion mapping `direct_scaling`, `leader_reprojection`, `follower_reprojection` works
- [ ] code to figure out ending pose
## Implementation
- [x] add leaders: dualsense
- [x] add leaders: joycon
- [x] add leaders: keybooard
- [x] add followers: papras_7dof_dual_arm
- [x] add followers: papras_6dof_dual_arm
- [x] IsaacGym
- [x] Make better import of pinocchio (paprle.ik.pinocchio)

- [ ] check power gripper (paprle.hands.power_gripper)
- [ ] check gripper-trigger joint - make vis better in urdf & xmls

- [ ] Leader reprojection - resolve it with direct_joint_mapping boolean and output_type
- [ ] Stop when  ik err is too high

- [ ] Implement Feedback
- [ ] organize feedback code in hw interface side

- [ ] add leaders: visionpro
- [ ] add leaders: offline trajectory

- [ ] add followers: ur5_dual_arm
- [ ] add followers: papras_stand
- [ ] add followers: papras_orthrus
- [ ] add followers: g1
- [ ] add followers: other humanoids
- [ ] add followers: OMY
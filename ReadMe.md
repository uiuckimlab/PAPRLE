# PAPRLE (Plug-And-Play Robotic Limb Environment)
## Overview
![concept](docs/media/paprle_concept_comp.gif)

This repository provides source code for the teleoperation system of PAPRLE.

PAPRLE supports diverse follower robot configurations and also diverse leader configurations. 

For installation, please refer to [Installation](docs/Installation.md).

## Supported Environment
- [x] Mujoco
- [x] Isaacgym
- [x] ROS1 (Gazebo or Hardware)
- [x] ROS2 (Gazebo or Hardware)

## Supported Robots (Follower)
- Using [PAPRAS](https://uiuckimlab.github.io/papras-pages), we can build various configurations of robotic arms. 
PAPRLE is designed to support such diverse configurations, and below are the configurations we have tested so far:
  - [x] `papras_6dof` : One-arm 6-DOF PAPRAS
  - [x] `papras_7dof` : One-arm 7-DOF PAPRAS
  - [x] `papras_6dof_2arm_table` : Dual-arm 6-DOF PAPRAS on Table
  - [x] `papras_7dof_2arm_table` : Dual-arm 7-DOF PAPRAS on Table
  - [x] `papras_orthrus` : Dual-arm 7-DOF PAPRAS on Spot (Orthrus)
  - [x] `papras_stand` : Dual-arm 7-DOF PAPRAS on Stand
        
PAPRAS is a fully open-source robotic arm, and you can find the CAD files and other detailed information on the site.
For more information about PAPRAS, please visit [this site](https://uiuckimlab.github.io/papras-pages). 
- Also, PAPRLE has been tested with other robots.
  - [x] `G1` : Unitree G1
  - [x] `OpenManipulator-Y` : OpenManipulator-Y from Robotis
  - [x] `PiPER`: PiPER 

## Supported Devices (Leader)
- [x] Keyboard
- [x] Puppeteer
- [x] VisionPro
- [x] Joycon
- [x] PS5 DualSense


## Citation
This repository is based on [PAPRLE](https://uiuckimlab.github.io/papras-pages). <br>
If you find this code useful in your research, please consider citing:
```
@article{kwon2025paprle,
      title={{PAPRLE (Plug-And-Play Robotic Limb Environment): A Modular Ecosystem for Robotic Limbs}}, 
      author={Obin Kwon and Sankalp Yamsani and Noboru Myers and Sean Taylor and Jooyoung Hong and Kyungseo Park and Alex Alspach and Joohyung Kim},
      year={2025},
      journal={arXiv preprint arXiv:2507.05555},
      url={https://arxiv.org/abs/2507.05555}, 
}
```

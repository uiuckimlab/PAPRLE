# PAPRLE (Plug-And-Play Robotic Limb Environment)
## Table of Contents
- [Overview](#overview)
- [Supported Devices, Robots, and Environments](#supported-controllers-robots-and-environments)
- [Usages](#usage)sss

## Overview
This repository provides a teleoperation system for diverse robot configurations.
The system is built with the goal of providing a plug-and-play experience, enabling users to quickly set up and use different robotic limbs in various environments.
![concept](docs/media/paprle_concept_comp.gif)

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
- Also, PAPRLE has been tested with other robots on Mujoco and IsaacGym.
  - [x] `G1` : Unitree G1
  - [x] `H1` : Unitree H1
  - [x] `UR5` : UR5
  - [x] `OpenManipulator-Y` : OpenManipulator-Y from Robotis 


## Supported Devices (Leader)
- [ ] Keyboard
- [ ] Puppeteer
- [ ] VisionPro
- [ ] Joycon
- [ ] PS5 DualSense

# PAPRLE (Plug-And-Play Robotic Limb Environment)
## Table of Contents
- [Overview](#overview)
- [Supported Devices, Robots, and Environments](#supported-controllers-robots-and-environments)
- [Usage](#usage)

## Overview
This repository provides a teleoperation system for diverse robots (mainly for PAPRAS).
The system is built with the goal of providing a plug-and-play experience, enabling users to quickly set up and use different robotic limbs in various environments.
So you can find how to setup the new robot configuration with PAPRAS, and how to control the robot with different controllers.
You can check which controllers, robots, and envs are supported in the following section.

![concept](docs/media/paprle_concept_comp.gif)

## Supported Environment
- [x] Mujoco
- [x] Isaacgym
- [x] ROS1 (Gazebo or Hardware)
- [x] ROS2 (Gazebo or Hardware)

## Supported Robots
- Gallery


## Installation
**Recommended Environment**
- Ubuntu 20.04 or Ubuntu 22.04
- python 3.8 or 3.10

**1. Clone the repository**

```bash
git clone https://github.com/uiuckimlab/PAPRLE.git
```

**2. Create a virtual environment or use conda**
```bash
# Create a virtual environment
virtualenv <env_name> -p python3.8
virutalenv <env_name>/bin/activate
```
or
```bash
# Create a conda environment
conda create -n <env_name> python=3.8 # or python=3.10
conda activate <env_name>
```

**3. Install**
```bash
pip install -e .
```


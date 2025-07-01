
## Environment Setup
**Recommended Environment**
- Ubuntu 20.04, python 3.8
- Ubuntu 22.04, python 3.10

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

**4. Usage**

You can run the teleoperation system with the following command:
```bash
python run_teleop.py -f <robot_name> -l <leader_name> -e <env_name>
```
For example:
``` bash
python run_teleop.py -f papras_kitchen -l sliders -e mujoco # Teleoperate PAPARAS Kitchen configuration with Sliders in Mujoco
```
```bash
python run_teleop.py -f g1 -l keyboard -e isaacgym # Teleoperate Unitree G1 with keyboard in Isaacgym
```
More options can be found with:
```bash
python run_teleop.py --help
```


## Hardware
You can run your own robot if the robot is based on ROS1/ROS2.
- We also provide a GUI to write config file for the new follower setup.

If you are interested in using PAPRAS and Pluggable Puppeteer, please refer to:

- PAPRAS Interface: https://github.com/uiuckimlab/PAPRAS-V0-Public/

- PAPRAS Models + PAPRLE Leaders: https://github.com/uiuckimlab/PAPRLE_hw

Also, we utilize `moveit` for initializing and terminating in data collection process.

- Please install `pymoveit2`: https://github.com/AndrejOrsula/pymoveit2/tree/main
- Place this package in your ROS workspace, and source the workspace before running teleop code. 


## Simulator
#### Isaacgym (Only for Python 3.8) 
- Pre-installation is needed: [Download IsaacGym](https://developer.nvidia.com/isaac-gym)
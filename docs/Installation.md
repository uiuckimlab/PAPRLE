
## Installation
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

## Hardware

For hardware setup, please refer to

PAPRAS Interface: https://github.com/uiuckimlab/PAPRAS-V0-Public/

PAPRAS Models + PAPRLE Leaders: https://github.com/uiuckimlab/PAPRLE_hw



## Simulator
#### Isaacgym
- [Download IsaacGym](https://developer.nvidia.com/isaac-gym)
- Recommended python version is python 3.8, but still you can install with python 3.10
  - Just comment "python_requires='>=3.6,<3.9'," in line 42 of setup.py (isaacgym/python/setup.py)
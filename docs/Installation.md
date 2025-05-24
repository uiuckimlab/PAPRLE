## Environment Setup
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
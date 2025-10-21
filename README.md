# CBCT teeth generation with point diffusion

## Usage
Tested with Python 3.11 in a conda environment.
```
pip install -r requirements.txt
```

## Data loading

Put the ```data``` folder in the main working directory.
```
data
└───dentition
    └───patient000
        └───verts
    └───patient001
    └───patient002
    └───...
└───train_patients.txt
└───val_patients.txt
```

## Step by Step Tutorial
### 1. Create virtual environment using python 3.11
```
conda create -n pcdiff python=3.11 -y
```
### 2. Install the correct Python and Pytorch
```
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118 

```
### 3. Install all requiredthe libraries and dependencies
```
pip install -r requirements.txt
``` 

## Steps to activate tmux and virtual environment
```
tmux
conda activate venv
bash
source ~/.bashrc
nvcc --version
``` 

## Training
```
python3 train_generation.py --bs [] --workers [] --niter []
python3 train_generation.py --bs 8 --workers 8 --niter 100 --grad_clip 1.0

``


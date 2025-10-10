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

## Training
```
python3 train_generation.py --bs [] --workers [] --niter []
python3 train_generation.py --bs 8 --workers 8 --niter 10000

```

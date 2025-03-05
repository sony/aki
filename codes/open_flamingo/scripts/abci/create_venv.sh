#!/bin/bash

# load environmental modules
module load cuda/12.6/12.6.1 cudnn/9.5/9.5.1

# create venv
python -m venv ~/venv/AKI
source ~/venv/AKI/bin/activate

# install packages
pip install --no-cache-dir -e codes
pip install tensorboard webdataset==0.2.100 scipy==1.14.1 wandb==0.16.6 && pip uninstall numpy -y && pip install numpy==1.26.4

pip install flash-attn --no-build-isolation
pip install accelerate bitsandbytes
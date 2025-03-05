#!/bin/bash

# create a virtual environment
python3 -m venv AKI
source ~/AKI/bin/activate

# install dependencies
pip install --no-cache-dir -e codes
pip install tensorboard webdataset==0.2.100 scipy==1.14.1 wandb==0.16.6 && pip uninstall numpy -y && pip install numpy==1.26.4 && pip install hydra-core==1.3.2

# install the demo package
pip install gradio==4.44.1
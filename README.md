# Revolve2

Revolve2 is a Python package for optimization, geared towards modular robots and evolutionary computing.

The current fork implements developmental representations.

## Installation steps 


**Simulator**

This version of Isaac Gym works only on Linux and needs exactly python 3.8.

Download and install IsaacGym_Preview_4_Package:

https://developer.nvidia.com/isaac-gym or https://drive.google.com/file/d/1msrQnu-ls_tj7gq92TXeZHFeACRrht6h/view?usp=sharing

pip install <isaacgym_path>/python






**Revolve2 installation (plasticoding_v3 version)**



python3.8 -m pip install virtualenv


python3.8 -m virtualenv .venv


source .venv/bin/activate




git clone https://github.com/karinemiras/revolve2.git

git checkout plasticoding_v3

sudo apt install libcereal-dev

./revolve2/dev_requirements.sh



**Analysis libs**


pip3 install pycairo


pip install opencv-python


pip3 install squaternion


pip3 install -U scikit-learn


pip3 install colored


pip3 install seaborn


pip3 install statannot


pip install greenlet

#some libs might not be on the list

## Replicating GRN epistasis experiments and analysis


./experiments/default_study/run-experiments.sh experiments/other_studies/GRNbody.sh
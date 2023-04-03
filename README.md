<img  align="right" width="150" height="150"  src="/docs/source/logo.png">

# Revolve2
Revolve2 is a Python package for optimization, geared towards modular robots and evolutionary computing.
Its primary features are a modular robot framework, wrappers around physics simulators, and evolutionary optimizers.
It consists of multiple smaller python packages and optional supplementary packages that contain varying functionality that is not always required.

## Documentation
[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/)

## Installation and instructions for the plasticoding_v2 branch.

The plasticoding_v2 branch in this fork is rebased up to **v0.3.8-beta1** of the main branch in the original repo.


**Simulators:**

You can use either Issac Gym or Mujoco as simulators. 

Isaac Gym works only on Linux and needs exactly python 3.8 and a pre-installation: 
https://developer.nvidia.com/isaac-gym
pip install <isaacgym_path>/python

Mujoco will be installed in the dev script below (comment installations out of the dev if you dont need them).

===

**Virtual environment**

python3.8 -m pip install virtualenv

python3.8 -m virtualenv .venv

source .venv/bin/activate

===

**Repo**:

git clone https://github.com/karinemiras/revolve2.git

git checkout plasticoding_v2

./revolve2/dev_requirements.sh

===

**Branch's **libs**:**

pip3 install pycairo

pip install opencv-python

pip3 install squaternion

pip3 install -U scikit-learn

pip3 install colored

pip3 install seaborn

pip3 install statannot

pip install greenlet

==

To **Run** a **single** experiment:

python3 experiments/body_openbrain_evo/optimize.py --mainpath yourpath --simulator isaac

or

python3 experiments/body_openbrain_evo/optimize.py --mainpath yourpath --simulator mujoco

To **Run** a **batch** of experiments in the background (screens):

./experiments/body_openbrain_evo/run-experiments.sh

To **Check** the status of the **batch** with:

./experiments/body_openbrain_evo/check-experiments.sh

and/or 

screen -list

To **Analyze** the results the batch with:

./experiments/body_openbrain_evo/run-analysis.sh





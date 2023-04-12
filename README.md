<img  align="right" width="150" height="150"  src="/docs/source/logo.png">

# Revolve2
Revolve2 is a Python package for optimization, geared towards modular robots and evolutionary computing.
Its primary features are a modular robot framework, wrappers around physics simulators, and evolutionary optimizers.
It consists of multiple smaller python packages and optional supplementary packages that contain varying functionality that is not always required.

The **plasticodinv_v2** branch is Karine Miras' personal branch, but it welcomes students.

## Installation steps the plasticoding_v2 branch

The **plasticoding_v2** branch in this fork is rebased up to **v0.3.8-beta1** of the main branch in the original repo.


**Simulators:**

You can use either Issac Gym or Mujoco as a simulator (MAKE A CHOICE - though installing both is also ok)

1) **Isaac Gym** works only on Linux and needs exactly python 3.8 and a pre-installation: 
https://developer.nvidia.com/isaac-gym
pip install <isaacgym_path>/python

2) **Mujoco** is installed simply by the dev_requirements.sh below.

ps: The dev_requirements.sh below tries to install both Isaac and Mujoco. If you don't want some of them, comment their pip out.

===

**Virtual environment**

python3.8 -m pip install virtualenv

python3.8 -m virtualenv .venv

source .venv/bin/activate

===

**Repository**:

git clone https://github.com/karinemiras/revolve2.git

git checkout plasticoding_v2

**Main installation**:

linux: sudo apt install libcereal-dev
OR
mac: brew install cereal

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


## Running plasticoding_v2 branch

To **Run** a **single** experiment:

python3 experiments/default_study/optimize.py --mainpath /home/mystuffff  --population_size 5 --offspring_size 5 --num_generations 2 --headless 0 --simulator isaac

ps: for Mujoco set --simulator mujoco

To **Run** and analyze a **batch** of experiments in the background (screens):

./experiments/default_study/run-experiments.sh

To **Check** the status of the **batch** with:

./experiments/default_study/check-experiments.sh

and/or 

screen -list

ps: to parameterize your own batch, make your own version of paramsdefault.sh and provide it to run-experiments.sh and other bashes

To only **Analyze** the results the batch with:

./experiments/default_study/run-analysis.sh


## Revolve's documentation
If you are having problems with the installation, consult the official documentation:
[ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/)

There you can also find some basic tutorials (they do not cover the extra features from the plasticoding_v2 branch).

For problems with the main installation or simulator's installation contact the CI-Group scientific programmer.
For problems with features of the plasticoding_v2 branch contact Karine Miras.

## Trouble shooting

-     The error below means you have garbage (less than at least one finished generation) in your experiment folder: delete the folder.
    pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],

    TypeError: 'NoneType' object is not subscriptable_


-       If you get an error related to 'egg' when installing isaac gym, try moving isaac's pip to the end of the list in the dev shell.
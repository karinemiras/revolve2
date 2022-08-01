#!/bin/bash

study = "default_study"
# make sure to provide experiments names in alphabetic order
experiments = "plane,tilted5,tilted10,tilted15"
runs = 15

python experiments/default_study/consolidate.py $study $experiments $runs;
python experiments/default_study/plot_intermethod.py $study $experiments $runs;
python experiments/default_study/snapshots_bests.py $study $experiments $runs;
python experiments/default_study/bests_snap_2d.py $study $experiments $runs;
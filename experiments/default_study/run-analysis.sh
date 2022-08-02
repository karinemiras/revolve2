#!/bin/bash

study="default_study"
# make sure to provide experiments names in alphabetic order
experiments="plane,tilted5,tilted10,tilted15"
runs=10
generations=(1,92)
final_gen=92


python experiments/default_study/snapshots_bests.py $study $experiments $runs $generations;
#python experiments/default_study/bests_snap_2d.py $study $experiments $runs $generations;
#python experiments/default_study/consolidate.py $study $experiments $runs $final_gen;
#python experiments/default_study/plot_intermethod.py $study $experiments $runs $generations;

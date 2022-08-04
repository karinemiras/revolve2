#!/bin/bash

study="default_study"
# make sure to provide experiments names in alphabetic order
experiments=("seasonal" "seasonalplastic")
runs=10
generations=(200)
final_gen=200


python experiments/default_study/snapshots_bests.py $study $experiments $runs $generations;
python experiments/default_study/bests_snap_2d.py $study $experiments $runs $generations;
python experiments/default_study/consolidate.py $study $experiments $runs $final_gen;

#python experiments/default_study/plot_static.py $study $experiments $runs $generations;

python experiments/default_study/plot_seasonal.py $study $experiments $runs $generations;

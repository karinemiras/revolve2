#!/bin/bash

# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh

# delimiter is comma, example:
#experiments="exp1,epx2"
# exps order is the same for all params

#experiments=("defaultexperiment")
experiments=("plane,tilted")

# these params are the same for all exps
# gens for boxplots and snapshots
generations=(100)
#gen for lineplots
final_gen=100
runs=30

mainpath="karine"
study="default_study"

python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath;
python experiments/${study}/plot_static.py $study $experiments $runs $generations $mainpath;

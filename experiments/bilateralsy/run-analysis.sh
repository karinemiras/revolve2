#!/bin/bash

# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh pathPARAMSFILE/PARAMSFILE.sh

DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file


python experiments/default_study/snapshots_bests.py $study $experiments $runs $generations $outputs_path $loop $body_phenotype $bisymmetry;
python experiments/default_study/bests_snap_2d.py $study $experiments $runs $generations $outputs_path;

comparison='basic_plots'
python experiments/default_study/consolidate.py $study $experiments $runs $final_gen $comparison $outputs_path;
python experiments/default_study/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;

./experiments/bilateralsy/analyze_robustness.sh experiments/bilateralsy/bilateralsy.sh

comparison='basic_plots_better'
runs="1,2,3,4,5,7,8,12,13,14,18,20,24,25,26,28,31,34,35,36,38,39|1,2,3,6,8,9,10,14,16,18,19,20,21,25,26,27,28,29,30,31,32,33,35,38,40"
python experiments/${study_path}/consolidate.py $study $experiments $runs $final_gen $comparison $outputs_path;
python experiments/default_study/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;

comparison='basic_plots_worse'
runs="6,11,15,17,22,27,30,37|1,2,3,6,8,9,10,14,16,18,19,20,21,25,26,27,28,29,30,31,32,33,35,38,40"
python experiments/${study_path}/consolidate.py $study $experiments $runs $final_gen $comparison $outputs_path;
python experiments/default_study/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;

comparison='better_worse'
runs="1,2,3,4,5,7,8,12,13,14,18,20,24,25,26,28,31,34,35,36,38,39|6,11,15,17,22,27,30,37"
python experiments/${study_path}/consolidate.py $study $experiments $runs $final_gen $comparison $outputs_path;
python experiments/${study_path}/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;

















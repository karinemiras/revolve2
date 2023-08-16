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


python experiments/${study_path}/snapshots_bests.py $study $experiments $runs $generations $outputs_path $loop $body_phenotype $bisymmetry;
#python experiments/${study_path}/bests_snap_2d.py $study $experiments $runs $generations $outputs_path;
#
#comparison='basic_plots'
#python experiments/${study_path}/consolidate.py $study $experiments $runs $final_gen $comparison $outputs_path;
#python experiments/${study_path}/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;
#



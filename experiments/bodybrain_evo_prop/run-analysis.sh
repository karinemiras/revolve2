#!/bin/bash

# run this script from the root (revolve folder): ./experiments/bodybrain_evo/run-analysis.sh PARAMSFILE

DIR="$(dirname "${BASH_SOURCE[0]}")"
study="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=paramsdefault
  else
    params_file=$1
fi

source $DIR/$params_file.sh

python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $outputs_path;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $outputs_path;

python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $outputs_path;
comparison='basic_plots'
python experiments/${study}/plot_static.py $study $experiments $runs $generations $comparison $outputs_path;




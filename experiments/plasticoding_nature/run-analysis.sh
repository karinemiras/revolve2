#!/bin/bash

study="plasticoding_nature"
# make sure to provide experiments names in alphabetic order
# arrays use comas in this case
#experiments=("exp1,epx2")
experiments=("seasonal,seasonalplastic")
mainpath="karine"
runs=10
generations=(200)
final_gen=200

python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath;


#python experiments/${study}/plot_static.py $study $experiments $runs $generations $mainpath;

python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $mainpath;

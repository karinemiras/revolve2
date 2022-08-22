#!/bin/bash

# delimiter is comma, example:
#experiments="exp1,epx2"
# exps order is the same for all params

experiments="seasonale.200.200.150,plastice.200.200.150"

# these params are the same for all exps
# gens for boxplots and snapshots
generations=(150)
#gen for lineplots
final_gen=150
runs=10
mainpath="karine"
study="plasticoding_nature"

python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath;


#python experiments/${study}/plot_static.py $study $experiments $runs $generations $mainpath;

python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $mainpath;

#!/bin/bash

# delimiter is comma, example:
#experiments="exp1,epx2"
# exps order is the same for all params

experiments="nonplasticforthright,plasticforthright,nonplasticbackforth,plasticbackforth,fullplasticforthright,fullplasticbackforth,onlyforth"

# these params are the same for all exps
# gens for boxplots and snapshots
generations=(150)
#gen for lineplots
final_gen=150
runs=20
mainpath="/home/ripper8/projects/working_data"
analysis="analysis"
study="plasticoding_cppntasks"


python experiments/${study}/snapshots_bests.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/bests_snap_2d.py $study $experiments $runs $generations $mainpath;
python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath $analysis;

experiments="fullplasticforthright,nonplasticforthright,plasticforthright"
comparison='forthright'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
python experiments/${study}/plot_trajectory.py $study $experiments $runs $final_gen $comparison $mainpath;

experiments="fullplasticbackforth,nonplasticbackforth,plasticbackforth"
comparison='backforth'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
python experiments/${study}/plot_trajectory.py $study $experiments $runs $final_gen $comparison $mainpath;

experiments="fullplasticbackforth"
comparison='backforth'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
experiments="fullplasticforthright"
comparison='forthright'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;


experiments="fullplasticforthright,nonplasticforthright,plasticforthright,z_onlyforth"
comparison='forthright'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
experiments="fullplasticbackforth,nonplasticbackforth,plasticbackforth,z_onlyforth"
comparison='backforth'
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;


experiments="onlyforth,fullplasticforthright,nonplasticforthright,plasticforthright"
comparison='forthright'
python experiments/${study}/plot_seasonal2.py $study $experiments $runs $generations $comparison $mainpath $analysis;
experiments="onlyforth,fullplasticbackforth,nonplasticbackforth,plasticbackforth"
comparison='backforth'
python experiments/${study}/plot_seasonal2.py $study $experiments $runs $generations $comparison $mainpath $analysis;


experiments="plasticforthright"
comparison='forthright'
python experiments/${study}/brainchanges.py $study $experiments $runs $generations $mainpath $comparison;
experiments="plasticbackforth"
comparison='backforth'
python experiments/${study}/brainchanges.py $study $experiments $runs $generations $mainpath $comparison;

# METADATA:

# backforth means that in cond 1 speed_y should be max and in cond 2 speed_y should be min
# back and forth are visually left and right respectively (in practice, first it goes forth and then back)

# forthright means that in cond 1 speed_y should be max and in cond 2 speed_x should be max
# forth and right are visually right and down respectively



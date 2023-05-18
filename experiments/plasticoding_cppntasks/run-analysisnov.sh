#!/bin/bash

# delimiter is comma, example:
#experiments="exp1,epx2"
# exps order is the same for all params


experiments="novfullplasticforthright,novnonplasticforthright,novplasticforthright,novfullplasticbackforth,novnonplasticbackforth,novplasticbackforth"

#these params are the same for all exps
#gens for boxplots and snapshots
generations=(50)
#gen for lineplots
final_gen=50
runs=4
mainpath="karine"
analysis="analysisnovel"
study="plasticoding_cppntasks"

python experiments/${study}/consolidate.py $study $experiments $runs $final_gen $mainpath $analysis;

comparison='forthright'
experiments="novfullplasticforthright,novnonplasticforthright,novplasticforthright"
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
python experiments/${study}/heatmaps_pca.py $study $experiments $runs $generations $comparison $mainpath $analysis;

comparison='backforth'
experiments="novfullplasticbackforth,novnonplasticbackforth,novplasticbackforth"
python experiments/${study}/plot_seasonal.py $study $experiments $runs $generations $comparison $mainpath $analysis;
python experiments/${study}/heatmaps_pca.py $study $experiments $runs $generations $comparison $mainpath $analysis;
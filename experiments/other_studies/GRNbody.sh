#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/home/ripper8/projects/working_data"

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of three vars below is coma. example:
#experiments="exp1,epx2"
# exps order is the same for all three vars
# exps names should not be fully contained in each other

study="GRNv3knockouts"
experiments="reg2m2"
tfs="reg10m2,reg2m2"


# conditions have effect only for isaac
# conditions: friction_dynamicfriction_yrotation_idleparam_idleparam
seasons_conditions="1.0_1.0_0_0_0,1.0_1.0_0_0_0"

####

nruns=30

runs=""
for i in $(seq 1 $nruns);
do
  runs=("${runs}${i},")
done
runs=${runs::-1}

watchruns=$runs

# use num_generations>=50 for more interesting results
num_generations="100"

# use population_size>=100 for more interesting results
population_size="100"

# use offspring_size>=100 for more interesting results
offspring_size="100"

# bash loop frequency: adjust seconds according to exp size, e.g, 300.
# (low values for short experiments will try to spawn and log too often)
delay_setup_script=300

# for issac, recommended not more than two in the rippers
num_terminals=2

# gens for boxplots, snapshots, videos (by default the last gen)
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="0,$num_generations"

mutation_prob=0.9

crossover_prob=1

# use simulation_time>=20 for more interesting results
simulation_time=30

max_modules=20

### PARAMS END ###
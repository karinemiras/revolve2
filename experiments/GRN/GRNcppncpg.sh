#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/home/ripper8/projects/working_data"

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter of arrays is coma.
# exps order is the same for all params
# exps names should not be fully contained in each other

study="GRN"

experiments="GRNcppncpg"

# conditions have effect only for isaac
# conditions: friction_dynamicfriction_yrotation_idleparam_idleparam
seasons_conditions="1.0_1.0_0_0_0"

#bileteral symmetry
bisymmetry="0"

nruns=1

runs=""
for i in $(seq 1 $nruns);
do
  runs=("${runs}${i},")
done
runs=${runs::-1}

watchruns=$runs

simulator="isaac"
#simulator="mujoco"

loop="open"

body_phenotype="evolvable"

# use num_generations=100 for more interesting results
num_generations="1"

# use population_size=100 for more interesting results
population_size="100"

# use offspring_size=100 for more interesting results
offspring_size="100"

# bash loop frequency: adjust seconds according to exp size, e.g, 300.
# (low values for short experiments will try to spawn and log too often)
delay_setup_script=5

# for issac, recommended not more than two in the rippers
num_terminals=2

# gens for boxplots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=1

crossover_prob=0

# use simulation_time=30 for more interesting results
simulation_time=30

max_modules=30

### PARAMS END ###

#!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/home/ripper8/projects/working_data"

# DO NOT use underline ( _ ) in the study and experiments names
# delimiter is space, example:
#experiments=("exp1" "epx2")
# exps order is the same for all params
# exps names should not be fully contained in each other

study="defaultstudy"

experiments=("defaultexperiment")
seasons_conditions=("1.0_1.0_0_0_0")

runs=2

watchruns="1,2"

simulator="isaac"
#simulator="mujoco"

loop="open"

body_phenotype="evolvable"

# use num_generations=100 for more interesting results
num_generations="2"

# use population_size=100 for more interesting results
population_size="5"

# use offspring_size=100 for more interesting results
offspring_size="5"

# bash loop frequency: adjust seconds according to exp size, e.g, 300.
# (low values for short experiments will try to spawn and log too often)
delay_setup_script=10

# for issac, recommended not more than two in the rippers
num_terminals=1

# gens for boxplots, snapshots, videos (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=1

crossover_prob=0

# use simulation_time=30 for more interesting results
simulation_time=5

### PARAMS END ###
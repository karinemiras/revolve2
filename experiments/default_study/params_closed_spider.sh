    #!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/home/ripper8/projects/working_data"

# DO NOT use underline ( _ ) in the experiments names
# delimiter is space, example:
#experiments=("exp1" "epx2")
# exps order is the same for all params
# exps names should not be fully contained in each other

study="closed_spider"

experiments=("defaultexperiment")
seasons_conditions=("1.0_1.0_0_0_0")

runs=20

simulator="isaac"

loop="closed"

body_phenotype="spider"

num_generations="50"

population_size="50"

offspring_size="30"

delay_setup_script=300

num_terminals=2

# gens for boxplots and snapshots (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="$num_generations"

mutation_prob=0.8

crossover_prob=0.8

simulation_time=30

### PARAMS END ###
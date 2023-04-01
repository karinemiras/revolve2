    #!/bin/bash


### PARAMS INI ###

# this should be the path for the output files (choose YOUR OWN dir!)
outputs_path="/home/ripper8/projects/working_data"

# DO NOT use underline ( _ ) in the experiments names
# delimiter is space, example:
#experiments=("exp1" "epx2")
# exps order is the same for all params
# exps names should not be fully contained in each other

study="default_study"
experiments=("proprioceptionsin3")
seasons_conditions=("1.0_1.0_0_0_0")

runs=10

# use num_generations=50 for interesting results, and num_generations=3 for quick test
num_generations="50"

# use population_size=50 for interesting results, and population_size=11 for quick test
population_size="50"

# use offspring_size=30 for interesting results, and offspring_size=11 for quick test
offspring_size="30"

# use delay_setup_script=600 . adjust according to exp size. (low values for shor experiments will try to spawn and log too often)
delay_setup_script=600

# recommended not more than two in the rippers
num_terminals=2

# gens for boxplots and snapshots (by default the last gen)
#generations="1,$num_generations"
generations="$num_generations"

# max gen to filter lineplots  (by default the last gen)
final_gen="$num_generations"

# frequency (update) for querying the controller
control_frequency=5

mutation_prob=0.8

crossover_prob=0.8

### PARAMS END ###
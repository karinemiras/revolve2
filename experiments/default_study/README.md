# Running plasticoding


To **Run** a **single** experiment:


python3 experiments/default_study/optimize.py --mainpath /home/mystuffff  --population_size 5 --offspring_size 5 --num_generations 2 --headless 0 




To **Run** a **batch** of experiments in the background:


./experiments/default_study/run-experiments.sh


To **Run** a **batch** of experiments using current terminal:


./experiments/default_study/setup-experiments.sh experiments/default_study/paramsdefault.sh


To **Check** the status of the **batch**:


./experiments/default_study/check-experiments.sh experiments/default_study/paramsdefault.sh



and/or 


screen -list


To only **Analyze** the results of the batch:


./experiments/default_study/run-analysis.sh experiments/default_study/paramsdefault.sh


To  **Watch** the best robots of the batch:


./experiments/default_study/watch_and_record.sh experiments/default_study/paramsdefault.sh



ps: to parameterize your own batch, make your own version of paramsdefault.sh and provide it to run-experiments.sh and other bashes



## Trouble shooting


-     The error below means you have garbage (less than at least one finished generation) in your experiment folder: delete the folder.

    pool_measures[i] = MeasureRelative(genotype_measures=pool_measures[i],


    TypeError: 'NoneType' object is not subscriptable_



-       If you get an error related to 'egg' when installing isaac gym, try moving isaac's pip to the end of the list in the dev shell.

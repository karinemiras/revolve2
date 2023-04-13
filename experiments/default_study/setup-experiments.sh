#!/bin/bash
#set -e
#set -x


DIR="$(dirname "${BASH_SOURCE[0]}")"
study_path="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=$DIR/paramsdefault.sh
  else
    params_file=$1
fi

source $params_file

mkdir ${outputs_path}/${study}
mkdir ${outputs_path}/${study}/analysis


possible_screens=()

# defines possible ports for screens
for t in $(seq 1 $((${num_terminals}))); do
    possible_screens+=($t)
done


while true
	do

   # printf "\n  >>>> loop ... \n"

    # discover free terminals

    active_screens=()
    free_screens=()
    active_experiments=()


    declare -a arr="$(screen -list)"


    for obj in ${arr[@]}; do

       screenstudy="$(cut -d'_' -f2 <<<"$obj")"

        if [[ "$screenstudy" == *"${study}"* ]]; then
           #   printf "\n screen ${obj} is on\n"
              screen="$(cut -d'_' -f3 <<<"$obj")"
              active_experiments+=("$(cut -d'_' -f3 -<<<"$obj")_$(cut -d'_' -f4 -<<<"$obj")")
              active_screens+=($screen)
          fi
    done

   for possible_screen in "${possible_screens[@]}"; do
       if [[ ! " ${active_screens[@]} " =~ " ${possible_screen} " ]]; then
           free_screens+=($possible_screen)
     fi
      done


    # discover unfinished experiments

    to_do=()
    unfinished=()
    for i in $(seq $runs)
    do
        run=$(($i))

        for experiment in "${experiments[@]}"
        do

      #   printf  "\n${experiment}_${run} \n"
         file="${outputs_path}/${study}/${experiment}_${run}.log";

         #check experiments status
         if [[ -f "$file" ]]; then

              lastgen=$(grep -c "Finished generation" $file);
              finishedgen=$(grep -c "Finished optimizing." $file);

            #  echo "latest finished gens ${lastgen}";

               if [ "$finishedgen" == 0 ]; then
                 unfinished+=("${experiment}_${run}")

                # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
             fi
         else
             # not started yet
        #     echo " None";
               unfinished+=("${experiment}_${run}")
               # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
         fi

        done
    done


    # spawns N experiments (N is according to free screens)

    max_fs=${#free_screens[@]}
    to_do_now=("${to_do[@]:0:$max_fs}")

    p=0
    for to_d in "${to_do_now[@]}"; do

        exp=$(cut -d'_' -f1 <<<"${to_d}")
        run=$(cut -d'_' -f2 <<<"${to_d}")
        idx=$( echo ${experiments[@]/${exp}//} | cut -d/ -f1 | wc -w | tr -d ' ' )

        # nice -n19 python3  experiments/${study}/optimize.py
        screen -d -m -S _${study}_${free_screens[$p]}_${to_d} -L -Logfile ${outputs_path}/${study}/${exp}_${run}".log" \
               python3  experiments/${study_path}/optimize.py --mainpath ${outputs_path} \
               --experiment_name ${exp} --seasons_conditions ${seasons_conditions[$idx]} --run ${run} --study=${study} \
               --num_generations ${num_generations} --population_size ${population_size} --offspring_size ${offspring_size} \
               --loop ${loop} --body_phenotype ${body_phenotype} --simulator ${simulator} --simulation_time ${simulation_time} \
               --crossover_prob ${crossover_prob} --mutation_prob ${mutation_prob};

        printf "\n >> (re)starting ${study}_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done

   # if all experiments are finished, run analysis and make videos
   # (NOTE: IF THE SCREEN IS LOCKED, YOU JUST GET VIDEO WITH A LOCKED SCREEN...)
   if [ -z "$unfinished" ]; then

      printf "\nanalysis...\n"
      ./experiments/${study_path}/run-analysis.sh $params_file

      ./experiments/${study_path}/makevideos.sh $params_file

      pkill -f ${study}_loop
   fi

    # use this longer period for longer experiments
    sleep $delay_setup_script;

done




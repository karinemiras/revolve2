#!/bin/bash
#set -e
#set -x


study="default_study"
experiments=("diversity")
fitness_measure="pool_diversity"
runs=6
generations=10
num_terminals=3


possible_screens=()

# defines possible ports for screens
for t in $(seq 1 $((${num_terminals}))); do
    possible_screens+=($t)
done


while true
	do

    printf "\n  >>>> loop ... \n"

    # discover free terminals

    active_screens=()
    free_screens=()
    active_experiments=()


    declare -a arr="$(screen -list)"


    for obj in ${arr[@]}; do

        if [[ "$obj" == *"screen_"* ]]; then
          printf "\n screen ${obj} is on\n"
          screen="$(cut -d'_' -f2 <<<"$obj")"
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
    for i in $(seq $runs)
    do
        run=$(($i))

        for experiment in "${experiments[@]}"
        do

         printf  "\n${experiment}_${run} \n"
         file="experiments/${study}/${experiment}_${run}/log.txt";

         #check experiments status
         if [[ -f "$file" ]]; then

             value=$(grep "Finished generation" $file|tail -n1|sed -E "s/\Finished generation ([0-9]+).*/\1/g");
             echo " ${value} "

             # unfinished TODO change != to < ?
             if [ "$value" != "$checkpoints" ]; then

                # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
             fi
         else
             # not started yet
             echo " None";
              # only if not already running
                if [[ ! " ${active_experiments[@]} " =~ " ${experiment}_${run} " ]]; then
                   to_do+=("${experiment}_${run}")
                fi
         fi

        done
    done


    # spawns N experiments (N is according to free screens)

    max_fs=${#free_screens[@]}
    to_do=("${to_do[@]:0:$max_fs}")


    p=0
    for to_d in "${to_do[@]}"; do

        exp=$(cut -d'_' -f1 <<<"${to_d}")
        run=$(cut -d'_' -f2 <<<"${to_d}")

        echo screen -d -m -S screen_${free_screens[$p]}_${to_d} -L -Logfile data/${study}/${exp}_${run}".log" nice -n19 python3  experiments/${study}/optimize.py --experiment_name ${exp} --fitness_measure ${fitness_measure} --run ${run};

        printf "\n >> (re)starting screen_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done

    sleep 300;
   #sleep 1800;

done

# screen -ls  | egrep "^\s*[0-9]+.exp_" | awk -F "." '{print $1}' |  xargs kill
# killall screen
# screen -r naaameee
# screen -list

#!/bin/bash
#set -e
#set -x


study="default_study"
# DO NOT use _
experiments=("plane" "tilted5" "tilted10")
seasons_conditions=("1.0_1.0_0" "1.0_1.0_5" "1.0_1.0_10")
runs=20
num_generations="200"

# recommended 10-15
num_terminals=15

mkdir data/${study}

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
         file="/storage/karine/${study}/${experiment}_${run}.log";

         #check experiments status
         if [[ -f "$file" ]]; then

              lastgen=$(grep -c "Finished generation" $file);
              echo "latest finished gen ${lastgen}";

             if [ "$lastgen" != "$num_generations" ]; then

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
        idx_fit=$( echo ${experiments[@]/${exp}//} | cut -d/ -f1 | wc -w | tr -d ' ' )

        # nice -n19 python3  experiments/${study}/optimize.py
        screen -d -m -S screen_${free_screens[$p]}_${to_d} -L -Logfile /storage/karine/${study}/${exp}_${run}".log" python3  experiments/${study}/optimize.py --experiment_name ${exp} --seasons_conditions ${seasons_conditions[$idx_fit]} --run ${run};

        printf "\n >> (re)starting screen_${free_screens[$p]}_${to_d} \n\n"
        p=$((${p}+1))

    done

    sleep 1800;

done

# run from revolve root
# screen -ls  | egrep "^\s*[0-9]+.screen_" | awk -F "." '{print $1}' |  xargs kill
# killall screen
# screen -r naaameee
# screen -list


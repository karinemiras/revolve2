#!/bin/bash
#set -e
#set -x

# run this script from the root (revolve folder): ./experiments/default_study/run-analysis.sh PARAMSFILE

DIR="$(dirname "${BASH_SOURCE[0]}")"
study="$(basename $DIR)"

if [ $# -eq 0 ]
  then
     params_file=paramsdefault
  else
    params_file=$1
fi

source $DIR/$params_file.sh

# discover unfinished experiments

screen -list

to_do=()
for i in $(seq $runs)
do
    run=$(($i))

    for experiment in "${experiments[@]}"
    do

     file="${outputs_path}/${study}/${experiment}_${run}.log";
     printf  "\n${file}\n"

     #check experiments status
     if [[ -f "$file" ]]; then

            lastgen=$(grep -c "Finished generation" $file);
            echo " latest finished gen ${lastgen}";

     else
         # not started yet
         echo " None";
     fi

    done
done

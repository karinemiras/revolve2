#!/bin/bash
#set -e
#set -x

study="plasticoding_nature"
# arrays delimiter is space
experiments=("seasonale.200.200.150" "plastice.200.200.150")
runs=10
mainpath="karine"

# discover unfinished experiments

to_do=()
for i in $(seq $runs)
do
    run=$(($i))

    for experiment in "${experiments[@]}"
    do

     printf  "\n${experiment}_${run} \n"
     file="/storage/${mainpath}/${study}/${experiment}_${run}.log";

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

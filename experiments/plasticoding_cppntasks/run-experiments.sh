#!/bin/bash
study="plasticoding_cppntasks"
mainpath="/home/ripper8/projects/working_data"
screen -d -m -S run_loop -L -Logfile ${mainpath}/${study}/setuploop.log ./experiments/${study}/setup-experiments.sh
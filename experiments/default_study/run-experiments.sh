#!/bin/bash

study="default_study"
screen -d -m -S run_loop -L -Logfile /storage/karine/${study}/setuploop.log ./experiments/default_study/setup-experiments.sh
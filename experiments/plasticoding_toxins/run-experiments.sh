#!/bin/bash

study="plasticoding_toxins"
mainpath="karine"
screen -d -m -S run_loop -L -Logfile $/storage/{mainpath}/${study}/setuploop.log ./experiments/${study}/setup-experiments.sh
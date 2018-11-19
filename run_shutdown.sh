#!/bin/bash
source activate pytorch_p36
sh run_scheduler_train.sh | tee scheduler_file.txt
sudo shutdown -h now
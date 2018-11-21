#!/bin/bash
stdbuf -o 0 python3 train.py | tee exp3train_file.txt

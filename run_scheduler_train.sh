#!/bin/bash
stdbuf -o 0 python3 train.py | tee scheduler_file.txt
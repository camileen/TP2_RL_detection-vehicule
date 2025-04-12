#!/bin/bash

# Create .venv directory
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP2_RL_detection-vehicule/.venv
cd /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP2_RL_detection-vehicule/.venv
python3 -m venv iat-tp2 # create a virtual environment named iat-tp2

# Create results directory
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP2_RL_detection-vehicule/results/metrics
mkdir -p /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP2_RL_detection-vehicule/results/q-tables


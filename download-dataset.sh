#!/bin/bash

# Download the dataset zip from Kaggle
if [ ! -d /mnt/c/Users/byoub/Code/INSA-Lyon/IAT/TP2_RL_detection-vehicule/dataset/ ]; then
  curl -L -o /mnt/c/Users/byoub/Downloads/vehicle-detection-image-set.zip\
  https://www.kaggle.com/api/v1/datasets/download/brsdincer/vehicle-detection-image-set

  # Unzip the dataset
  unzip /mnt/c/Users/byoub/Downloads/vehicle-detection-image-set.zip \
    -d /mnt/c/Users/byoub/Downloads
fi



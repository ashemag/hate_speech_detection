#!/bin/sh
export SEED=$1

export NAME=test

export PATH_DATA=data/

python deep_learning_experiments.py --num_epochs 100 --model CNN --name $NAME --seed $SEED --embedding_key learn --embedding_level word

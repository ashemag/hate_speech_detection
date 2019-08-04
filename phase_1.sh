#!/bin/sh

####################################################### BASELINE EXPERIMENTS ######################################################
#
##python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 28 --embedding_key twitter --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 27 --embedding_key twitter --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 26 --embedding_key twitter --embedding_level word

#python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_1 --seed 28 --embedding_key bert --embedding_level word  --experiment_flag 1
python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_1 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 1
python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_1 --seed 26 --embedding_key bert --embedding_level word  --experiment_flag 1

##python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 28 --embedding_key NA --embedding_level tdidf
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 27 --embedding_key NA --embedding_level tdidf
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name baseline --seed 26 --embedding_key NA --embedding_level tdidf
##
#
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 28 --embedding_key NA --embedding_level tdidf
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 27 --embedding_key NA --embedding_level tdidf
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 26 --embedding_key NA --embedding_level tdidf
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 28 --embedding_key twitter --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 27 --embedding_key twitter --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 26 --embedding_key twitter --embedding_level word
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 28 --embedding_key bert --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 27 --embedding_key bert --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name baseline --seed 26 --embedding_key bert --embedding_level word
#
###
###
###python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 28 --embedding_key twitter --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 27 --embedding_key twitter --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 26 --embedding_key twitter --embedding_level word
##
###python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 28 --embedding_key bert --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 27 --embedding_key bert --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 26 --embedding_key bert --embedding_level word
##
###python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 28 --embedding_key NA --embedding_level tdidf
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 27 --embedding_key NA --embedding_level tdidf
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name baseline --seed 26 --embedding_key NA --embedding_level tdidf
##
##
###
###python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 28 --embedding_key twitter --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 27 --embedding_key twitter --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 26 --embedding_key twitter --embedding_level word
##
###python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 28 --embedding_key bert --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 27 --embedding_key bert --embedding_level word
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 26 --embedding_key bert --embedding_level word
##
###python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 28 --embedding_key NA --embedding_level tdidf
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 27 --embedding_key NA --embedding_level tdidf
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name baseline_5_layer --seed 26 --embedding_key NA --embedding_level tdidf
##
##
##
##

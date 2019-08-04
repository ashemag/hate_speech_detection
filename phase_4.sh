#!/usr/bin/env bash
##
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-1 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .1
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-1 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .1
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-1 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .1
#
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-3 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .3
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-3 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .3
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-3 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .3
#
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-5 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .5
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-5 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .5
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-5 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .5
#
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-7 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .7
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-7 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .7
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name dropout-7 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4 --dropout .7
#


python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_4 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4
python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_4 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4
python deep_learning_experiments.py --num_epochs 100 --model CNN --name tuned_phase_4 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4

#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 4
##
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 28 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 27 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_4 --seed 26 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 28 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 27 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 26 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 4
##
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_4 --seed 26 --embedding_key bert --embedding_level word--experiment_flag 4
#
##
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 4
###
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4
##
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 28 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 27 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_4 --seed 26 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
##
##
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 4
##python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 4
#
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 4
#
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 28 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 27 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_4 --seed 26 --embedding_key tdidf --embedding_level tdidf --experiment_flag 4

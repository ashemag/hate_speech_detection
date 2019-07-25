#!/usr/bin/env bash

python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 28 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 27 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_3 --seed 26 --embedding_key NA --embedding_level tdidf --experiment_flag 3



python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 28 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 27 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 26 --embedding_key NA --embedding_level tdidf --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_3 --seed 26 --embedding_key bert --embedding_level word--experiment_flag 3



python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 28 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 27 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_3 --seed 26 --embedding_key NA --embedding_level tdidf --experiment_flag 3



python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 28 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 27 --embedding_key twitter --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 26 --embedding_key twitter --embedding_level word --experiment_flag 3

python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 28 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 27 --embedding_key bert --embedding_level word --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 26 --embedding_key bert --embedding_level word --experiment_flag 3
#
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 28 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 27 --embedding_key NA --embedding_level tdidf --experiment_flag 3
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_3 --seed 26 --embedding_key NA --embedding_level tdidf --experiment_flag 3

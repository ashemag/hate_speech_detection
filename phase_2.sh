#!/usr/bin/env bash

python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 28 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 27 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 26 --embedding_key twitter --embedding_level word
#
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 28 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 27 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 26 --embedding_key word2vec --embedding_level word

python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 28 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 27 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model CNN --name phase_2 --seed 26 --embedding_key NA --embedding_level tdidf




python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 28 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 27 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 26 --embedding_key NA --embedding_level tdidf

python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 28 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 27 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 26 --embedding_key twitter --embedding_level word
#
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 28 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 27 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model MLP --name phase_2 --seed 26 --embedding_key word2vec --embedding_level word



python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 28 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 27 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 26 --embedding_key twitter --embedding_level word

#python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 28 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 27 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 26 --embedding_key word2vec --embedding_level word

python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 28 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 27 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model DENSENET --name phase_2 --seed 26 --embedding_key NA --embedding_level tdidf



python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 28 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 27 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 26 --embedding_key twitter --embedding_level word
#
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 28 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 27 --embedding_key word2vec --embedding_level word
#python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 26 --embedding_key word2vec --embedding_level word

python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 28 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 27 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model LSTM --name phase_2 --seed 26 --embedding_key NA --embedding_level tdidf

#!/usr/bin/env bash

python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 28 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 27 --embedding_key twitter --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 26 --embedding_key twitter --embedding_level word

python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 28 --embedding_key word2vec --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 27 --embedding_key word2vec --embedding_level word
python deep_learning_experiments.py --num_epochs 100 --model CNN --name context_embeddings --seed 26 --embedding_key word2vec --embedding_level word

python deep_learning_experiments.py --num_epochs 100 --model MLP --name context_embeddings --seed 28 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model MLP --name context_embeddings --seed 27 --embedding_key NA --embedding_level tdidf
python deep_learning_experiments.py --num_epochs 100 --model MLP --name context_embeddings --seed 26 --embedding_key NA --embedding_level tdidf

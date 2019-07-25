"""
Script for converting experiments.csv
to latex table for dissertation
"""

import pandas as pd
from globals import ROOT_DIR
import os
import csv
import numpy as np

# data = pd.read_csv(os.path.join(ROOT_DIR,'results/experiments.csv'), header=header, index_col=0, squeeze=True).to_dict()

def process_csv(filename = os.path.join(ROOT_DIR,'results/experiments.csv')):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        print(reader.fieldnames)
        for row in reader:
            data[row['title']] = row
    return data

data = process_csv()
models = ['MLP', 'CNN', 'LSTM', 'DENSENET']
metrics = ['test_f_score_hateful', 'test_f_score_abusive', 'test_f_score', 'test_precision', 'test_recall']
embeddings = ['NA_tdidf', 'twitter_word', 'bert_word']
output_str = ''
EXPERIMENT_KEY = 'phase_3'
for i, model in enumerate(models):
    for j, embed in enumerate(embeddings):

        if model == 'logistic_regression':
            if embed == 'NA_tdidf':
                embed = 'tdidf'
            if embed == 'twitter_word':
                embed = 'twitter'
            if embed == 'bert_word':
                embed = 'bert'

        if EXPERIMENT_KEY == 'baseline' and model == 'LSTM':
            key = model + '_' + EXPERIMENT_KEY + '_5_layer_' + embed
        else:
            key = model + '_' + EXPERIMENT_KEY + '_' + embed

        if embed == 'bert_word' or embed == 'bert':
            embed_key = 'Bert'
        elif embed == 'twitter_word' or embed == 'twitter':
            embed_key = 'Twit'
        else:
            embed_key = 'TFIDF'
        if model == 'DENSENET':
            title = 'DENSE-' + embed_key
        elif model == 'logistic_regression':
            title = 'LR-' + embed_key
        else:
            title = model + '-' + embed_key
        output_str += title
        for metric in metrics:
            value = data[key][metric]
            if value != '-':
                value = value.split(' ± ')
                mean = np.around(float(value[0]), 2)
                std = np.around(float(value[1]), 2)
                output_str += ' & {} $\pm$ {}'.format(mean, std)
        output_str += ' \\\ '
        if j == len(embeddings) - 1 and (i != 0 or j != 0) and (i != len(models) - 1):
            output_str += ' \midrule '
        if i == len(models) - 1 and j == len(embeddings) - 1:
            output_str += ' \\bottomrule '
        output_str += " \n"
print(output_str)
exit()

import sys
sys.path.append("..")
import configparser
from data_providers import TextDataProvider
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from globals import ROOT_DIR
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import time
from utils import prepare_output_file
LABEL_MAPPING = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}

config = configparser.ConfigParser()
config.read('../config.ini')
path_data = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_DATA'])
path_labels = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_LABELS'])


def get_f_scores(y_true, preds, output, type_key, round_param=4):
    f_score = f1_score(y_true, preds, average='weighted')
    precision = precision_score(y_true, preds, average='weighted')
    recall = recall_score(y_true, preds, average='weighted')

    print("F Score {:.2f}".format(f_score))
    output['{}_f_score'.format(type_key)] = np.around(f_score, round_param)
    output['{}_precision'.format(type_key)] = np.around(precision, round_param)
    output['{}_recall'.format(type_key)] = np.around(recall, round_param)

    f1_scores = f1_score(y_true, preds, average=None)
    precision_scores = precision_score(y_true, preds, average=None)
    recall_scores = recall_score(y_true, preds, average=None)

    for i in range(len(f1_scores)):
        output[type_key + '_f_score_' + LABEL_MAPPING[i]] = np.around(f1_scores[i], round_param)
        output[type_key + '_precision_' + LABEL_MAPPING[i]] = np.around(recall_scores[i], round_param)
        output[type_key + '_recall_' + LABEL_MAPPING[i]] = np.around(recall_scores[i], round_param)


def results(model, type_key, x, y_true, output, round_param=4):
    # get accuracy
    acc = model.score(x, y_true)
    print("Accuracy {:.2f}".format(acc))
    output['{}_acc'.format(type_key)] = np.around(acc, round_param)

    # get f score metrics
    preds = model.predict(x)
    get_f_scores(y_true, preds, output, type_key)
# get_confusion_matrix(y_true, preds, output, type_key)


def populate_missing_params(output):
    """
    Fills data with fields we are ignored for LR
    """
    missing_params_class = []  # train_loss_class_hateful
    missing_params_overall = ['loss']
    for type_key in ['train', 'valid', 'test']:
        for item in missing_params_class:
            for label in range(4):
                output['{}_{}_class_{}'.format(type_key, item, LABEL_MAPPING[label])] = '-'
        for item in missing_params_overall:
            output['{}_{}'.format(type_key, item)] = '-'


def output_to_csv(output, file_action_key='a+', experiment_name='logistic_regression_cv_baseline'):
    """
    Output results to .csv
    """
    output['title'] = experiment_name
    output['epoch'] = '-'
    output['learning_rate'] = '-'

    results_dir = os.path.join(ROOT_DIR, 'results/{}'.format(experiment_name))
    if not os.path.isdir(os.path.join(results_dir)):
        print("Directory added")
        os.mkdir(results_dir)
    prepare_output_file(filename=os.path.join(results_dir, 'results.csv'), output=[output],
                        file_action_key=file_action_key)


import gensim


def process_word_embeddings(data):
    data_copy = {}
    for key, value in data.items():
        if 'x' in key:
            data_copy[key] = gensim.matutils.unitvec(np.array(value).mean(axis=1)).astype(np.float32)
        else:
            data_copy[key] = value
    return data_copy


"""
Experiments 

"""
experiment_seeds = [26, 27, 28]
for i, seed in enumerate(experiment_seeds):
    print("=== Experiment with seed {} running ===".format(seed))
    embedding_key = 'bert'
    experiment_key = 1
    data = TextDataProvider(path_data, path_labels, experiment_key).generate_word_level_embeddings(embedding_key, seed)
    data_copy = process_word_embeddings(data)
    # data_copy = TextDataProvider(path_data, path_labels, 1).generate_tdidf_embeddings(seed)
    print("=== Model Started Training ===")
    start = time.time()
    model = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial')
    model = model.fit(data_copy['x_train'], data_copy['y_train'])

    print("=== Model Completed Training ({:2f} min) ===".format((time.time() - start) / 60))

    output = {}
    output['seed'] = seed
    populate_missing_params(output)  # so that we can add to same sheet as Neural Nets
    for type_key in ['train', 'valid', 'test']:
        print("=== Processing {} set ===".format(type_key))
        results(model, type_key, data_copy['x_{}'.format(type_key)], data_copy['y_{}'.format(type_key)], output)
        print('\n')
    file_action_key = 'w' if i == 0 else 'a+'
    output_to_csv(output, file_action_key, experiment_name='logistic_regression_phase_{}_{}'.format(experiment_key, embedding_key))
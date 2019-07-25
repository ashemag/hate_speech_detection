"""
Runs baseline experiments
"""
from comet_ml import Experiment # necessary for comet to function
import argparse
import configparser
import torch
from globals import ROOT_DIR
from experiment_builder import ExperimentBuilder
from data_providers import TextDataProvider, ImbalancedDatasetSampler, DataProvider
import os
from models.cnn import word_cnn
from models.densenet import densenet
from models.lstm import lstm
from models.multilayer_perceptron import multi_layer_perceptron
import time
import numpy as np

# PARAMS

VERBOSE = True

config = configparser.ConfigParser()
config.read('config.ini')


def get_args():
    """
    To parse user parameters
    :return:
    """
    parser = argparse.ArgumentParser(description='CNN Hate Speech Detection Experiment.')
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--name', type=str, default='CNN_Experiment')
    parser.add_argument('--embedding_key', type=str, default='NA')
    parser.add_argument('--embedding_level', type=str, default='NA')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=.5)
    parser.add_argument('--experiment_flag', type=int, default=1)

    if VERBOSE:
        arg_str = [(str(key), str(value)) for (key, value) in vars(parser.parse_args()).items()]
        print(arg_str)
    return parser.parse_args()


def extract_data(embedding_key, embedding_level, seed, experiment_flag):
    path_data = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_DATA'])
    path_labels = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_LABELS'])
    data_provider = TextDataProvider(path_data, path_labels, experiment_flag)

    if embedding_level == 'word':
        return data_provider.generate_word_level_embeddings(embedding_key, seed)
    elif embedding_level == 'tdidf':
        return data_provider.generate_tdidf_embeddings(seed)


def wrap_data(batch_size, seed, data_local):
    train_set = DataProvider(inputs=data_local['x_train'], targets=data_local['y_train'], seed=seed)
    train_data_local = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   sampler=ImbalancedDatasetSampler(train_set))

    valid_set = DataProvider(inputs=data_local['x_valid'], targets=data_local['y_valid'], seed=seed)
    valid_data_local = torch.utils.data.DataLoader(valid_set,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   shuffle=False)

    test_set = DataProvider(inputs=data_local['x_test'], targets=data_local['y_test'], seed=seed)
    test_data_local = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  shuffle=False)

    return train_data_local, valid_data_local, test_data_local


def fetch_model(model_local, embedding_level, dropout):
    if model_local == 'MLP':
        return multi_layer_perceptron()
    if model_local == 'CNN':
        return word_cnn(dropout=dropout)
    if model_local == 'DENSENET':
        return densenet()
    if model_local == 'LSTM':
        return lstm()
    else:
        raise ValueError("Model key not found {}".format(embedding_level))


def fetch_model_parameters(args_local):
    return fetch_model(model_local=args_local.model,
                       embedding_level=args_local.embedding_level,
                       dropout=args_local.dropout)


def generate_device(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():  # checks whether a cuda gpu is available and whether the gpu flag is True
        device_local = torch.cuda.current_device()
        print("Using {} GPU(s)".format(torch.cuda.device_count()))
    else:
        print("Using CPU")
        device_local = torch.device('cpu')  # sets the device to be CPU

    return device_local


if __name__ == "__main__":
    args = get_args()
    device = generate_device(args.seed)
    data, data_map = extract_data(args.embedding_key, args.embedding_level, args.seed, args.experiment_flag)

    print("Wrapping data")
    train_set, valid_set, test_set = wrap_data(args.batch_size, args.seed, data)

    # create model
    sample_key = data['x_train'][0]
    input_shape = tuple([args.batch_size] + list(np.array(data_map[sample_key]['embedded_tweet']).shape))
    print("Fetching model with input shape {}".format(input_shape))
    model = fetch_model_parameters(args)

    # OUTPUT
    folder_title = '_'.join([args.model, args.name, args.embedding_key, args.embedding_level])
    print("=== Writing to folder {} ===".format(folder_title))
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(folder_title)
    start = time.time()

    hyper_params = {
        'seed': args.seed,
        'experiment_name': folder_title,
        'results_dir': results_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
        'input_shape': input_shape,
    }

    experiment = ExperimentBuilder(
        network_model=model,
        device=device,
        hyper_params=hyper_params,
        train_data=train_set,
        valid_data=valid_set,
        test_data=test_set,
        data_map=data_map,
        experiment_flag=args.experiment_flag,
    )

    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
    print("Total time (min): {:0.2f}".format(round((time.time() - start) / float(60), 4)))

"""
Runs baseline experiments
"""
from comet_ml import Experiment # necessary for comet to function
import argparse
import configparser
from torch import optim
from experiment_builder import ExperimentBuilder
from data_providers import *
import os
from models.cnn import *
from models.multilayer_perceptron import multi_layer_perceptron

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
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=.5)

    if VERBOSE:
        arg_str = [(str(key), str(value)) for (key, value) in vars(parser.parse_args()).items()]
        print(arg_str)
    return parser.parse_args()


def extract_data(embedding_key, embedding_level, seed):
    path_data = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_DATA'])
    path_labels = os.path.join(ROOT_DIR, config['DEFAULT']['PATH_LABELS'])
    data_provider = TextDataProvider(path_data, path_labels)

    if embedding_level == 'word':
        output = data_provider.generate_word_level_embeddings(embedding_key, seed)
    elif embedding_level == 'char':
        output = data_provider.generate_char_level_embeddings(seed)
    elif embedding_level == 'tdidf':
        output = data_provider.generate_tdidf_embeddings(seed)

    if VERBOSE:
        print("[Sizes] Training set: {}, Validation set: {}, Test set: {}".format(len(output['x_train']),
                                                                                  len(output['x_valid']),
                                                                                  len(output['x_test'])))
    return output


def wrap_data(batch_size, seed, x_train, y_train, x_valid, y_valid, x_test, y_test):
    train_set = DataProvider(inputs=x_train, targets=y_train, seed=seed)
    train_data_local = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   sampler=ImbalancedDatasetSampler(train_set))

    valid_set = DataProvider(inputs=x_valid, targets=y_valid, seed=seed)
    valid_data_local = torch.utils.data.DataLoader(valid_set,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   shuffle=False)

    test_set = DataProvider(inputs=x_test, targets=y_test, seed=seed)
    test_data_local = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  shuffle=False)

    return train_data_local, valid_data_local, test_data_local


def fetch_model(model, embedding_level, input_shape_local, dropout):
    if model == 'MLP':
        return multi_layer_perceptron(input_shape_local)
    if embedding_level == 'word':
        return word_cnn(input_shape_local, dropout)
    elif embedding_level == 'character':
        return character_cnn(input_shape_local)
    else:
        raise ValueError("Model key not found {}".format(embedding_level))


def fetch_model_parameters(args_local, input_shape_local):
    model_local, criterion_local, optimizer_local = fetch_model(model=args_local.model,
                                                                embedding_level=args_local.embedding_level,
                                                                input_shape_local=input_shape_local,
                                                                dropout=args_local.dropout)

    scheduler_local = optim.lr_scheduler.CosineAnnealingLR(optimizer_local, T_max=args_local.num_epochs, eta_min=0.0001)
    return model_local, criterion_local, optimizer_local, scheduler_local


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
    data = extract_data(args.embedding_key, args.embedding_level, args.seed)
    train_data, valid_data, test_data = wrap_data(args.batch_size, args.seed, **data)
    input_shape = tuple([args.batch_size] + list(np.array(data['x_train']).shape)[1:])
    model, criterion, optimizer, scheduler = fetch_model_parameters(args, input_shape)
    # OUTPUT
    folder_title = '_'.join([args.model, args.name, args.embedding_key, args.embedding_level])
    print("=== Writing to folder {} ===".format(folder_title))
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(folder_title)
    start = time.time()

    hyper_params = {
        'seed': args.seed,
        'experiment_name': folder_title,
        'results_dir': results_dir,
        'criterion': criterion,
        'scheduler': scheduler,
        'num_epochs': args.num_epochs,
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
    }

    experiment = ExperimentBuilder(
        network_model=model,
        device=device,
        hyper_params=hyper_params,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
    print("Total time (min): {:0.2f}".format(round((time.time() - start) / float(60), 4)))

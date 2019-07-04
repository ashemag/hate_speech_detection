"""
Runs baseline experiments
"""
from comet_ml import Experiment
import argparse
import configparser
from torch import optim
from globals import ROOT_DIR
from data_providers import *
import os

from models.logistic_regression import logistic_regression
from models.cnn import *

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
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--name', type=str, default='CNN_Experiment')
    parser.add_argument('--embedding', type=str, default='NA')
    parser.add_argument('--embedding_level', type=str, default='NA')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=.5)

    if VERBOSE:
        arg_str = [(str(key), str(value)) for (key, value) in vars(parser.parse_args()).items()]
        print(arg_str)
    return parser.parse_args()


def extract_data(embedding_key, embedding_level_key, seed):
    path_data, path_labels = config['DEFAULT']['PATH_DATA'], config['DEFAULT']['PATH_LABELS']
    p = TextDataProvider()
    x_train, y_train, x_val, y_val, x_test, y_test = p.extract(path_data, path_labels, embedding_key, embedding_level_key, seed)

    output = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test,
              'y_test': y_test}
    if VERBOSE:
        print("[Sizes] Training set: {}, Validation set: {}, Test set: {}".format(len(output['x_train']),
                                                                                  len(output['x_val']),
                                                                                  len(output['x_test'])))
    return output


def wrap_data(batch_size, seed, x_train, y_train, x_val, y_val, x_test, y_test):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainset = DataProvider(inputs=x_train, targets=y_train, seed=seed)
    train_data = torch.utils.data.DataLoader(trainset,
                                             batch_size=batch_size,
                                             num_workers=2,
                                             sampler=ImbalancedDatasetSampler(trainset))

    validset = DataProvider(inputs=x_val, targets=y_val, seed=seed)
    valid_data = torch.utils.data.DataLoader(validset,
                                             batch_size=batch_size,
                                             num_workers=2)

    testset = DataProvider(inputs=x_test, targets=y_test, seed=seed)
    test_data = torch.utils.data.DataLoader(testset,
                                            batch_size=batch_size,
                                            num_workers=2)

    return train_data, valid_data, test_data


def fetch_model(key, input_shape, num_output_classes, dropout):
    if key == 'CNN_word':
        return word_cnn(input_shape, dropout)
    elif key == 'CNN_character':
        return character_cnn(input_shape)
    elif key == 'logistic_regression_NA':
        return logistic_regression(input_shape, num_output_classes)
    else:
        raise ValueError("Model key not found {}".format(key))


def fetch_model_parameters(args, input_shape, num_output_classes):
    model_local, criterion_local, optimizer_local = fetch_model(key=args.model + '_' + args.embedding_level,
                                                                input_shape=input_shape,
                                                                num_output_classes=num_output_classes,
                                                                dropout=args.dropout)
    if not args.cpu:
        model_local = model_local.to(model_local.device)

    scheduler_local = optim.lr_scheduler.CosineAnnealingLR(optimizer_local, T_max=args.num_epochs, eta_min=0.0001)
    return model_local, criterion_local, optimizer_local, scheduler_local


if __name__ == "__main__":
    label_mapping = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}
    args = get_args()
    data = extract_data(args.embedding, args.embedding_level, args.seed)
    train_data, valid_data, test_data = wrap_data(args.batch_size, args.seed, **data)
    input_shape = tuple([args.batch_size] + list(np.array(data['x_train']).shape)[1:])
    model, criterion, optimizer, scheduler = fetch_model_parameters(args, input_shape, len(label_mapping))

    # OUTPUT
    folder_title = '_'.join([args.model, args.name, args.embedding, args.embedding_level])
    print("Writing to folder {}".format(folder_title))
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(folder_title)
    start = time.time()

    hyper_params = {
        'seed': args.seed,
        'title': folder_title,
        'label_mapping': label_mapping,
        'results_dir': results_dir,
        'criterion': criterion,
        'scheduler': scheduler,
        'num_epochs': args.num_epochs,
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'dropout': args.dropout,
    }

    model.train_evaluate(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        hyper_params=hyper_params
    )

    print("Total time (min): {:2f}".format(round((time.time() - start) / float(60), 4)))

import argparse
import ast
import os
import time

import torch
from torch import optim
from models.cnn import *
from globals import ROOT_DIR
from data_provider import *
import torchvision
import pickle

BATCH_SIZE = 64
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9
VERBOSE = True


def get_args():
    parser = argparse.ArgumentParser(description='CNN Hate Speech Detection Experiment.')
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cpu', type=bool, default=False)
    args = parser.parse_args()

    if VERBOSE:
        arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
        print(arg_str)
    return args


def prepare_output_file(filename, output=None, clean_flag=False):
    """

    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :return:
    """
    file_exists = os.path.isfile(filename)
    if clean_flag:
        if file_exists:
            os.remove(filename)
    else:
        if output is None:
            raise ValueError("Please specify output to write to output file.")

        with open(filename, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(output.keys()))
            writer.writeheader()
            if VERBOSE:
                print("Writing to file {0}".format(filename))
                print(output)
            writer.writerow(output)


def extract_data():
    # DATA
    if True:
        p = TextDataProvider()
        x_train, y_train, x_val, y_val, x_test, y_test = p.extract('data/80k_tweets.json', 'data/labels.csv')
        d = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test}
        for key, value in d.items():
            path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(key))
            with open(path, 'wb') as f:
                pickle.dump(value, f)
    # else:
    #     res = []
    #     for val in ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']:
    #         path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(val))
    #         with open(path, 'rb') as f:
    #             res.append(pickle.load(f))
    #     x_train, y_train, x_val, y_val, x_test, y_test = res
    if VERBOSE:
        print("SIZES: training set: {}, validation set: {}, test set: {}".format(len(x_train), len(x_val), len(x_test)))
    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_val, y_val, x_test, y_test = extract_data()

    args = get_args()
    model = TextCNN(input_shape=np.array(x_train).shape)
    if not args.cpu:
        model = model.to(model.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=MOMENTUM,
                                nesterov=True,
                                weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)

    # WRAP IN DP
    trainset = DataProvider(inputs=np.array(x_train), targets=np.array(y_train), batch_size=100, make_one_hot=False)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    validset = DataProvider(inputs=np.array(x_val), targets=np.array(y_val), batch_size=100, make_one_hot=False)
    valid_data = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=True, num_workers=2)

    testset = DataProvider(inputs=np.array(x_test), targets=np.array(y_test), batch_size=100, make_one_hot=False)
    test_data = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)

    # OUTPUT
    experiment_name = 'Vanilla_CNN_Dropout2'
    output_dir = os.path.join(ROOT_DIR, 'data/minority_class_experiments.csv')
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(experiment_name)
    start = time.time()
    bpm = model.train_evaluate(
        train_set=train_data,
        valid_full=valid_data,
        test_full=test_data,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler,
        label_mapping={0:'hateful', 1:'abusive', 2:'normal', 3:'spam'}
    )
    print(bpm)
    print("Total time (min): {}".format(round((time.time() - start) / float(60), 4)))
    results_dir_bpm = os.path.join(ROOT_DIR, '{}/best_performing_model.csv'.format(results_dir))
    prepare_output_file(output=bpm, filename=results_dir_bpm)

"""
Runs baseline experiments
"""
import argparse
import time
from torch import optim
from globals import ROOT_DIR
from models.cnn import *
from data_provider import *
import pickle
import os
from models.logistic_regression import LogisticRegression

# PARAMS
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-4
VERBOSE = True
FILENAME = 'data/80k_tweets.json'
FILENAME_LABELS = 'data/labels.csv'


def get_args():
    """
    TO parse user parameters
    :return:
    """
    parser = argparse.ArgumentParser(description='CNN Hate Speech Detection Experiment.')
    parser.add_argument('--seed', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument('--model', type=str, default='CNN')
    parser.add_argument('--name', type=str, default='CNN_Experiment')
    parser.add_argument('--embedding', type=str, default='N/A')
    parser.add_argument('--embedding_level', type=str, default='word')

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

        with open(filename, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(output.keys()))
            if not file_exists:
                writer.writeheader()
            if VERBOSE:
                print("Writing to file {0}".format(filename))
                print(output)
            writer.writerow(output)


def extract_data(embedding_key, embedding_level_key, model):
    """
    To get training/valid/test data
    :param key: True if data is saved and can pull from /data
    :return:
    """
    if model == 'CNN':
        saved_flag = False
        if not saved_flag:
            p = CNNTextDataProvider()
            x_train, y_train, x_val, y_val, x_test, y_test = p.extract(FILENAME, FILENAME_LABELS, embedding_key, embedding_level_key)
            d = {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test}

            # for key, value in d.items():
            #     path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(key))
            #     with open(path, 'wb') as f:
            #         pickle.dump(value, f)
        else:
            res = []
            for val in ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']:
                path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(val))
                with open(path, 'rb') as f:
                    res.append(pickle.load(f))
            x_train, y_train, x_val, y_val, x_test, y_test = res
        if VERBOSE:
            print("training set: {}, validation set: {}, test set: {}".format(len(x_train), len(x_val), len(x_test)))
        return x_train, y_train, x_val, y_val, x_test, y_test
    else:
        return LogisticRegressionDataProvider().extract(FILENAME, FILENAME_LABELS)


def wrap_data(x_train, y_train, x_val, y_val, x_test, y_test, seed):
    # WRAP IN DP
    trainset = DataProvider(inputs=np.array(x_train), targets=np.array(y_train), batch_size=BATCH_SIZE, make_one_hot=False,
                            seed=seed)
    train_data = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    validset = DataProvider(inputs=np.array(x_val), targets=np.array(y_val), batch_size=100, make_one_hot=False,
                            seed=seed)
    valid_data = torch.utils.data.DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = DataProvider(inputs=np.array(x_test), targets=np.array(y_test), batch_size=100, make_one_hot=False,
                           seed=seed)
    test_data = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    return train_data, valid_data, test_data


if __name__ == "__main__":
    label_mapping = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}
    args = get_args()
    x_train, y_train, x_val, y_val, x_test, y_test = extract_data(args.embedding, args.embedding_level, args.model)
    train_data, valid_data, test_data = wrap_data(x_train, y_train, x_val, y_val, x_test, y_test, args.seed)
    input_shape = tuple([BATCH_SIZE] + list(np.array(x_train).shape)[1:])

    if args.model == 'CNN':
        if args.embedding_level == 'word':
            model = WordLevelCNN(input_shape=input_shape)
        else:
            model = CharacterLevelCNN(input_shape=input_shape)
    if args.model == 'logistic_regression':
        model = LogisticRegression(input_shape=input_shape, num_output_classes=len(label_mapping))

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = None

    if not args.cpu:
        model = model.to(model.device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.0001)

    # OUTPUT
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format(args.model + '_' + args.name)
    start = time.time()
    bpm = model.train_evaluate(
        train_set=train_data,
        valid_full=valid_data,
        test_full=test_data,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler,
        label_mapping=label_mapping,
        criterion=criterion
    )

    # SAVE RESULTS
    bpm['embedding'] = args.embedding
    bpm['embedding_level'] = args.embedding_level
    bpm['seed'] = args.seed
    print(bpm)
    print("Total time (min): {}".format(round((time.time() - start) / float(60), 4)))
    title = '_'.join([args.model, args.name, args.embedding, args.embedding_level])
    results_dir_bpm = os.path.join(ROOT_DIR, '{}/{}.csv'.format(results_dir, title))
    prepare_output_file(output=bpm, filename=results_dir_bpm)

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
from models.logistic_regression import logistic_regression
from utils import prepare_output_file

# PARAMS
BATCH_SIZE = 64
VERBOSE = True
FILENAME = 'data/80k_tweets.json'
FILENAME_LABELS = 'data/labels.csv'


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

    if VERBOSE:
        arg_str = [(str(key), str(value)) for (key, value) in vars(parser.parse_args()).items()]
        print(arg_str)
    return parser.parse_args()


def extract_data(embedding_key, embedding_level_key, model):
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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


def fetch_model(key, input_shape, num_output_classes):
    if key == 'CNN_word':
        return word_cnn(input_shape)
    elif key == 'CNN_character':
        return character_cnn(input_shape)
    elif key == 'logistic_regression_NA':
        return logistic_regression(input_shape, num_output_classes)
    else:
        raise ValueError("Model key not found {}".format(key))


def fetch_model_parameters(args, input_shape, num_output_classes):
    model_local, criterion_local, optimizer_local = fetch_model(key=args.model + '_' + args.embedding_level,
                                                                input_shape=input_shape,
                                                                num_output_classes=num_output_classes)
    if not args.cpu:
        model_local = model_local.to(model_local.device)

    scheduler_local = optim.lr_scheduler.CosineAnnealingLR(optimizer_local, T_max=args.num_epochs, eta_min=0.0001)
    return model_local, criterion_local, optimizer_local, scheduler_local


if __name__ == "__main__":
    label_mapping = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}
    args = get_args()
    x_train, y_train, x_val, y_val, x_test, y_test = extract_data(args.embedding, args.embedding_level, args.model)
    train_data, valid_data, test_data = wrap_data(x_train, y_train, x_val, y_val, x_test, y_test, args.seed)
    input_shape = tuple([BATCH_SIZE] + list(np.array(x_train).shape)[1:])
    model, criterion, optimizer, scheduler = fetch_model_parameters(args, input_shape, len(label_mapping))

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

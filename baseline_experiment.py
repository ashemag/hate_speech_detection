import argparse
import ast
import os
import torch
from torch import optim
from models.cnn import *
from globals import ROOT_DIR
from data_provider import *
import torchvision
BATCH_SIZE = 64
LEARNING_RATE = .1
WEIGHT_DECAY = 1e-4
MOMENTUM = .9
import pickle

def get_args():
    parser = argparse.ArgumentParser(description='CNN Hate Speech Detection Experiment.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int)
    args = parser.parse_args()
    arg_str = [(str(key), str(value)) for (key, value) in vars(args).items()]
    print(arg_str)
    return args


if __name__ == "__main__":
    # DATA
    p = TextDataProvider()
    data_saved_flag = True
    if data_saved_flag:
        x_train, y_train, x_val, y_val, x_test, y_test = p.extract('data/80k_tweets.json', 'data/labels.csv')
        d = {'x_train': x_train, 'y_train':y_train, 'x_val':x_val, 'y_val':y_val, 'x_test':x_test, 'y_test':y_test}
        for key, value in d.items():
            path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(key))
            with open(path, 'wb') as f:
                pickle.dump(value, f)
    else:
        res = []
        for val in ['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test']:
            path = os.path.join(ROOT_DIR, 'data/{}.obj'.format(val))
            with open(path, 'rb') as f:
                res.append(pickle.load(f))
        x_train, y_train, x_val, y_val, x_test, y_test = res

    args = get_args()
    model = TextCNN(input_shape=np.array(x_train).shape)

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
    output_dir = os.path.join(ROOT_DIR, 'data/minority_class_experiments.csv')
    results_dir = os.path.join(ROOT_DIR, 'results/{}').format('Vanilla_CNN')

    bpm = model.train_evaluate(
        train_set=train_data,
        valid_full=valid_data,
        test_full=test_data,
        num_epochs=args.num_epochs,
        optimizer=optimizer,
        results_dir=results_dir,
        scheduler=scheduler,
    )

    print(bpm)

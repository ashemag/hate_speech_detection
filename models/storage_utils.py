import pickle
import os
import csv
from collections import OrderedDict

def save_statistics(statistics_to_save,file_path):
    '''
    :param statistics_to_save: dict, val type is float
    :param file_path: e.g. file_path = "C:/test_storage_utils/dir2/test.txt"
    '''
    if type(statistics_to_save) is not OrderedDict:
        raise TypeError('statistics_to_save must be OrderedDict instead got {}'.format(type(statistics_to_save)))

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_path,'a+') as f: # append mode + creates if doesn't exist
        header = ""
        line = ""
        for i,key in enumerate(statistics_to_save.keys()):
            val = statistics_to_save[key]
            if i==0:
                line = line + "{:.4f}".format(val)
                header = header + key
            else:
                line = line + "\t" + "{:.4f}".format(val)
                header = header + "\t" + key
        if os.stat(file_path).st_size == 0:  # if empty
            f.write(header+"\n")
        f.write(line+"\n")


def test():
    import numpy as np
    from adversarial_sampling_experiments import globals

    current_epoch = 0
    valid_epoch_acc = 30.1232
    valid_epoch_loss = 10.21321

    train_epoch_acc = valid_epoch_acc
    train_epoch_loss = valid_epoch_loss
    epoch_train_time = 100.11111

    valid_statistics_to_save = OrderedDict({
        'current_epoch': np.around(current_epoch,decimals=0),
        'valid_acc': np.around(valid_epoch_acc, decimals=4),
        'valid_loss': np.around(valid_epoch_loss, decimals=4)
    })

    train_statistics_to_save = OrderedDict({
        'current_epoch': current_epoch,
        'train_acc': np.around(train_epoch_acc, decimals=4),  # round results to 4 decimals.
        'train_loss': np.around(train_epoch_loss, decimals=4),
        'epoch_train_time': epoch_train_time
    })

    valid_path = os.path.join(globals.ROOT_DIR,'test_storage.txt')
    save_statistics(train_statistics_to_save,valid_path)

def main():
    pass

if __name__ == '__main__':
    main()
import numpy as np
import torch
import os
from models import storage_utils
from tqdm import tqdm
import sys
from collections import OrderedDict
import torch.nn as nn
from collections import defaultdict
import pickle
from sklearn.metrics import f1_score

# remove warning for f-score, precision, recall
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class Logger(object):
    def __init__(self,disable=False,stream=sys.stdout,filename=None):
        self.disable = disable
        self.stream = stream
        self.module_name = filename

    def error(self,str):
        if not self.disable: sys.stderr.write(str+'\n')

    def print(self, obj):
        if not self.disable: self.stream.write(str(obj))
        if self.module_name: self.stream.write('. {}'.format(os.path.splitext(self.module_name)[0]))
        self.stream.write('\n')


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.num_epochs = None
        self.train_data = None
        self.optimizer = None
        self.train_file_path = None
        self.cross_entropy = None
        self.scheduler = None
        self.gpu = True

        logger = Logger(stream=sys.stderr)
        logger.disable = False # if disabled does not print info messages.
        logger.module_name = __file__
        gpu_ids = '0'
        if not torch.cuda.is_available():
            print("GPU IS NOT AVAILABLE")
            self.device = torch.device('cpu')  # sets the device to be CPU
            self.gpu = False
        if ',' in gpu_ids:
            self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_ids.split(",")]
        else:
            self.device = torch.device('cuda:{}'.format(int(gpu_ids)))

        if type(self.device) is list:
            self.device = self.device[0]

        logger.print("gpu ids being used: {}".format(gpu_ids))

        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids  # (1)
        self.cuda() # (2)

        '''
        remarks:
        (1) sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use 
        by using the relevant GPU ID)
        (2) this makes model a cuda model. makes it so that gpu is used with it.
        '''

    def get_acc_batch(self, y_batch, y_batch_pred):
        _, y_pred_batch_int = torch.max(y_batch_pred.data, 1)  # argmax of predictions
        acc = np.mean(list(y_pred_batch_int.eq(y_batch.data).cpu()))  # compute accuracy
        return acc

    @staticmethod
    def save_train_epoch_results(batch_statistics,train_file_path):
        statistics_to_save = {"train_acc":0, "train_loss":0, "epoch_train_time":0,"current_epoch":0}
        statistics_to_save["current_epoch"] = batch_statistics["current_epoch"]
        statistics_to_save["epoch_train_time"] = batch_statistics["epoch_train_time"]

        for key, value in batch_statistics.items():
            if key not in ["current_epoch","epoch_train_time"]:
                batch_values = np.array(batch_statistics[key])
                epoch_val = np.mean(batch_values)  # get mean of all metrics of current epoch metrics dict
                statistics_to_save[key] = np.around(epoch_val, decimals=4)

        print(statistics_to_save)
        storage_utils.save_statistics(statistics_to_save,train_file_path)

    def train_evaluate(self, train_set, valid_full, test_full, num_epochs,
                       optimizer, results_dir,
                       label_mapping=None,
                       attack=None,
                       scheduler=None):

        # SET OUTPUT PATH
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        train_results_path = os.path.join(results_dir, 'train_results.txt')
        valid_and_test_results_path = os.path.join(results_dir, 'valid_and_test_results.txt')
        model_save_dir = os.path.join(results_dir, 'model')

        if attack is not None:
            advers_images_path = os.path.join(results_dir, 'advers_images.pickle')
            advs_images_dict = {}

        # SET LOGS
        logger = Logger(stream=sys.stderr, disable=False)
        gpu = self.gpu
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        if scheduler is not None:
            self.scheduler = scheduler

        def train_epoch(current_epoch):
            batch_statistics = defaultdict(lambda: [])
            with tqdm(total=len(train_set)) as pbar_val:
                for i, batch in enumerate(train_set):
                    x, y = batch
                    if gpu:
                        x = x.to(device=self.device)
                        y = y.to(device=self.device)

                    output = self.train_iteration(x, y, label_mapping)
                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    # SET PBAR
                    string_description = " ".join(
                        ["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_val.update(1)
                    # pbar_val.set_description(string_description)

            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)
            return epoch_stats

        def test_epoch(current_epoch):
            batch_statistics = defaultdict(lambda: [])

            with tqdm(total=len(valid_full)) as pbar_val:
                for i, batch in enumerate(valid_full):
                    x_all, y_all = batch
                    if gpu:
                        x_all = x_all.to(device=self.device)
                        y_all = y_all.to(device=self.device)
                    output = self.valid_iteration('valid', x_all, y_all, label_mapping)

                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    string_description = " ".join(["{}:{:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_val.update(1)
                    # pbar_val.set_description(string_description)

            with tqdm(total=len(test_full)) as pbar_test:
                for i, batch in enumerate(test_full):
                    x_all, y_all = batch
                    if gpu:
                        x_all = x_all.to(device=self.device)
                        y_all = y_all.to(device=self.device)

                    output = self.valid_iteration('test', x_all, y_all, label_mapping)

                    # SAVE BATCH STATS
                    for key, value, in output.items():
                        batch_statistics[key].append(output[key].item())

                    string_description = " ".join(
                        ["{}: {:.4f}".format(key, np.mean(value)) for key, value in batch_statistics.items()])
                    pbar_test.update(1)
                    # pbar_test.set_description(string_description)

            epoch_stats = OrderedDict({})
            epoch_stats['current_epoch'] = current_epoch
            for k, v in batch_statistics.items():
                epoch_stats[k] = np.around(np.mean(v), decimals=4)

            test_statistics_to_save = epoch_stats
            return test_statistics_to_save

        bpm = defaultdict(lambda: 0)
        torch.cuda.empty_cache()
        for current_epoch in range(self.num_epochs):
            train_statistics_to_save = train_epoch(current_epoch)
            test_statistics_to_save = test_epoch(current_epoch)

            # save train statistics.
            storage_utils.save_statistics(train_statistics_to_save, file_path=train_results_path)

            # save adversarial images.
            if attack is not None:
                with open(advers_images_path,
                          'wb') as f:  # note you overwrite the file each time but that okay since advs_images_dict grows each epoch.
                    pickle.dump(advs_images_dict, f)

            # save model.
            #self.save_model(model_save_dir, model_save_name='model_epoch_{}'.format(str(current_epoch)))
            logger.print(train_statistics_to_save)

            # save bpm statistics
            if test_statistics_to_save['valid_acc'] > bpm['valid_acc']:
                for key, value in test_statistics_to_save.items():
                    bpm[key] = value
                for key, value in train_statistics_to_save.items():
                    bpm[key] = value

            storage_utils.save_statistics(test_statistics_to_save, file_path=valid_and_test_results_path)
            logger.print(test_statistics_to_save)

            if scheduler is not None: scheduler.step()
            for param_group in self.optimizer.param_groups:
                logger.print('learning rate: {}'.format(param_group['lr']))
        return bpm

    def get_class_stats_helper(self, y_all, y_pred_all, criterion, class_idx):
        # find class specific stats
        y_min = []
        y_pred_min = []
        for i in range(y_all.shape[0]):
            if int(y_all[i].data) == class_idx:
                y_min.append(y_all[i])
                y_pred_min.append(y_pred_all[i])

        loss_min, acc_min = None, None
        if len(y_min) > 0:
            y_min = torch.stack(y_min, dim=0)
            y_pred_min = torch.stack(y_pred_min, dim=0)

            loss_min = (criterion(input=y_pred_min, target=y_min.view(-1)))
            acc_min = self.get_acc_batch(y_min, y_pred_min)
        return loss_min, acc_min

    def get_class_stats(self, type_key, output, y_all, y_pred_all, criterion, label_mapping, class_range=4):
        for class_idx in range(class_range):
            loss_min, acc_min = self.get_class_stats_helper(y_all, y_pred_all, criterion, class_idx)
            if acc_min is None or loss_min is None:
                continue

            class_key = label_mapping[class_idx] if label_mapping is not None else str(class_idx)

            output[type_key + '_acc_class_' + class_key] = acc_min
            output[type_key + '_loss_class_' + class_key] = loss_min
            # _, y_pred_all_int = torch.max(y_pred_all.data, 1)  # argmax of predictions
            # output[type_key + '_f_score_' + class_key] = f1_score(y_all.cpu().detach().numpy(), y_pred_all_int.cpu().detach().numpy(), average="macro")

    def train_iteration(self, x_all, y_all, label_mapping):
        self.train()
        criterion = nn.CrossEntropyLoss().cuda()
        x_all = x_all.float()
        y_pred_all = self.forward(x_all)
        loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        acc_all = self.get_acc_batch(y_all, y_pred_all)

        output = {'train_loss': loss_all.data, 'train_acc': acc_all}
        _, y_pred_all_int = torch.max(y_pred_all.data, 1)  # argmax of predictions
        scores = f1_score(
            y_all.cpu().detach().numpy(),
            y_pred_all_int.cpu().detach().numpy(),
            average=None
        )
        for i in range(len(scores)):
            output['train_f_score' + '_' + label_mapping[i]] = scores[i]
        self.get_class_stats('train', output, y_all, y_pred_all, criterion, label_mapping)
        return output

    def valid_iteration(self, type_key, x_all, y_all, label_mapping):
        with torch.no_grad():
            self.eval() # should be eval but something BN - todo: change later if no problems.
            '''
            Evaluating accuracy on whole batch 
            Evaluating accuracy on min examples 
            '''
            criterion = nn.CrossEntropyLoss().cuda()
            x_all = x_all.float()
            y_pred_all = self.forward(x_all)
            loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
            acc_all = self.get_acc_batch(y_all, y_pred_all)
            output = {type_key + '_loss': loss_all.data, type_key + '_acc': acc_all}

            _, y_pred_all_int = torch.max(y_pred_all.data, 1)  # argmax of predictions
            scores = f1_score(
                y_all.cpu().detach().numpy(),
                y_pred_all_int.cpu().detach().numpy(),
                average=None
            )
            for i in range(len(scores)):
                output[type_key + '_f_score' + '_' + label_mapping[i]] = scores[i]
            self.get_class_stats(type_key, output, y_all, y_pred_all, criterion, label_mapping)
        return output

    def save_model(self, model_save_dir,model_save_name):
        state = dict()
        state['network'] = self.state_dict()  # save network parameter and other variables.
        model_path = os.path.join(model_save_dir,model_save_name)

        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(state, f=model_path)

    def load_model(self, model_path):
        state = torch.load(f=model_path)
        self.load_state_dict(state_dict=state['network'])

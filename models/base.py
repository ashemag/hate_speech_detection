from comet_ml import OfflineExperiment
import numpy as np
import os
from models import storage_utils
from tqdm import tqdm
import sys
from collections import OrderedDict
from collections import defaultdict
import warnings
from sklearn.metrics import f1_score
import torch


# remove warning for f-score, precision, recall
def warn(*args, **kwargs):
    pass


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
        self.criterion = None
        self.scheduler = None
        self.gpu = True
        self.state = dict()

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

    def compute_batch_accuracy(self, y_batch, y_batch_pred):
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

    def _train_epoch(self, epoch_local, train_data, label_mapping, gpu):
        batch_statistics = defaultdict(lambda: [])
        with tqdm(total=len(train_data)) as pbar_train:
            for i, batch in enumerate(train_data):
                x, y = batch
                if gpu:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                output = self.compute_train_iteration(x, y, label_mapping)

                # SAVE BATCH STATS
                for key, value, in output.items():
                    batch_statistics[key].append(value.item())

                pbar_train.update(1)

        epoch_stats = OrderedDict({})
        epoch_stats['epoch'] = epoch_local
        for k, v in batch_statistics.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)
        return epoch_stats

    def _valid_epoch(self, epoch_local, valid_data, label_mapping, gpu):
        batch_statistics = defaultdict(lambda: [])
        with tqdm(total=len(valid_data)) as pbar_val:
            for i, batch in enumerate(valid_data):
                x_all, y_all = batch
                if gpu:
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)
                output = self.compute_valid_iteration('valid', x_all, y_all, label_mapping)

                # SAVE BATCH STATS
                for key, value, in output.items():
                    batch_statistics[key].append(value.item())

                pbar_val.update(1)
        epoch_stats = OrderedDict({})
        epoch_stats['epoch'] = epoch_local
        for k, v in batch_statistics.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)

        return epoch_stats

    def _test_epoch(self, test_data, label_mapping, gpu):
        batch_statistics = defaultdict(lambda: [])
        with tqdm(total=len(test_data)) as pbar_test:
            for i, batch in enumerate(test_data):
                x_all, y_all = batch
                if gpu:
                    x_all = x_all.to(device=self.device)
                    y_all = y_all.to(device=self.device)

                output = self.compute_valid_iteration('test', x_all, y_all, label_mapping)

                # SAVE BATCH STATS
                for key, value, in output.items():
                    batch_statistics[key].append(output[key].item())

                pbar_test.update(1)

        epoch_stats = OrderedDict({})
        for k, v in batch_statistics.items():
            epoch_stats[k] = np.around(np.mean(v), decimals=4)

        return epoch_stats

    @staticmethod
    def log_to_comet(experiment, epoch, stats_to_log):
        for stats in stats_to_log:
            for k, v in stats.items():
                experiment.log_metric(name=k, value=stats[k], step=epoch)

    def train_evaluate(self, train_data, valid_data, test_data, hyper_params):
        logger = Logger(stream=sys.stderr, disable=False)
        self.num_epochs = hyper_params['num_epochs']
        self.optimizer = hyper_params['optimizer']
        self.criterion = hyper_params['criterion']
        self.scheduler = None
        if 'scheduler' in hyper_params:
            self.scheduler = hyper_params['scheduler']

        results_dir = hyper_params['results_dir']
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # set output path
        train_results_path = os.path.join(results_dir, 'train_results_{}.csv'.format(hyper_params['seed']))
        valid_results_path = os.path.join(results_dir, 'valid_results_{}.csv'.format(hyper_params['seed']))
        test_results_path = os.path.join(results_dir, 'test_results.csv')

        experiment = OfflineExperiment(project_name=results_dir.split('/')[-1],
                                       workspace="ashemag",
                                       offline_directory=results_dir,
                                       api_key="bahH25CJj2oHsJiHr6m27pKOh")
        experiment.set_filename('experiment_{}'.format(hyper_params['seed']))
        experiment.set_name('experiment_{}'.format(hyper_params['seed']))

        experiment.log_parameters(hyper_params)
        bpm = defaultdict(lambda: 0)
        torch.cuda.empty_cache()
        for epoch in range(1, int(self.num_epochs + 1)):
            train_statistics = self._train_epoch(epoch, train_data, hyper_params['label_mapping'], self.gpu)
            valid_statistics = self._valid_epoch(epoch, valid_data, hyper_params['label_mapping'], self.gpu)

            self.log_to_comet(experiment, epoch, [train_statistics, valid_statistics])
            # Step
            if self.scheduler is not None:
                self.scheduler.step()
            for param_group in self.optimizer.param_groups:
                learning_rate = param_group['lr']
                logger.print('=== Learning rate: {} ==='.format(learning_rate))
                train_statistics['learning_rate'] = learning_rate

            # save bpm statistics
            if (valid_statistics['valid_f_score_hateful'] + valid_statistics['valid_f_score_abusive'])\
                    > (bpm['valid_f_score_hateful'] + bpm['valid_f_score_abusive']):
                for stats in [train_statistics, valid_statistics]:
                    for key, value in stats.items():
                        bpm[key] = value
                        self.state[key] = value
                self.save_model(model_save_dir=results_dir, state=self.state)

            # save train statistics.
            file_action_key = 'w' if epoch == 1 else 'a+'
            storage_utils.save_statistics(train_statistics, file_path=train_results_path, file_action_key=file_action_key)
            logger.print(train_statistics)
            storage_utils.save_statistics(valid_statistics, file_path=valid_results_path, file_action_key=file_action_key)
            logger.print(valid_statistics)

        print("=== Generating test set evaluation metrics ===")
        self.load_model(model_save_dir=results_dir)
        test_statistics = self._test_epoch(test_data, hyper_params['label_mapping'], self.gpu)
        test_statistics['seed'] = hyper_params['seed']
        test_statistics['title'] = hyper_params['title']
        self.log_to_comet(experiment, bpm['epoch'], [test_statistics])
        merge_dict = OrderedDict(list(bpm.items()) + list(test_statistics.items()))
        storage_utils.save_statistics(merge_dict, file_path=test_results_path, file_action_key='a+')
        logger.print(merge_dict)

    def compute_class_stats_helper(self, y_all, y_pred_all, criterion, class_idx):
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
            acc_min = self.compute_batch_accuracy(y_min, y_pred_min)
        return loss_min, acc_min

    def compute_class_stats(self, type_key, output, y_all, y_pred_all, criterion, label_mapping, class_range=4):
        for class_idx in range(class_range):
            loss_min, acc_min = self.compute_class_stats_helper(y_all, y_pred_all, criterion, class_idx)
            if acc_min is None or loss_min is None:
                continue

            class_key = label_mapping[class_idx] if label_mapping is not None else str(class_idx)

            output[type_key + '_acc_class_' + class_key] = acc_min
            output[type_key + '_loss_class_' + class_key] = loss_min

    def compute_train_iteration(self, x_all, y_all, label_mapping):
        self.train()
        criterion = self.criterion.cuda()
        x_all = x_all.float()
        y_pred_all = self.forward(x_all)
        loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
        self.optimizer.zero_grad()
        loss_all.backward()
        self.optimizer.step()
        acc_all = self.compute_batch_accuracy(y_all, y_pred_all)

        output = {'train_loss': loss_all.data, 'train_acc': acc_all}
        _, y_pred_all_int = torch.max(y_pred_all.data, 1)  # argmax of predictions
        scores = f1_score(
            y_all.cpu().detach().numpy(),
            y_pred_all_int.cpu().detach().numpy(),
            average=None
        )
        for i in range(len(scores)):
            output['train_f_score' + '_' + label_mapping[i]] = scores[i]
        self.compute_class_stats('train', output, y_all, y_pred_all, criterion, label_mapping)
        return output

    def compute_valid_iteration(self, type_key, x_all, y_all, label_mapping):
        with torch.no_grad():
            self.eval()
            '''
            Evaluating accuracy on whole batch 
            Evaluating accuracy on min examples 
            '''
            criterion = self.criterion.cuda()
            x_all = x_all.float()
            y_pred_all = self.forward(x_all)
            loss_all = criterion(input=y_pred_all, target=y_all.view(-1))
            acc_all = self.compute_batch_accuracy(y_all, y_pred_all)
            output = {type_key + '_loss': loss_all.data, type_key + '_acc': acc_all}

            _, y_pred_all_int = torch.max(y_pred_all.data, 1)  # argmax of predictions
            scores = f1_score(
                y_all.cpu().detach().numpy(),
                y_pred_all_int.cpu().detach().numpy(),
                average=None
            )
            for i in range(len(scores)):
                output[type_key + '_f_score' + '_' + label_mapping[i]] = scores[i]
            self.compute_class_stats(type_key, output, y_all, y_pred_all, criterion, label_mapping)
        return output

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        model_save_name = 'model_state'
        torch.save(state, f=os.path.join(model_save_dir, "{}".format(model_save_name)))  # save state at prespecified filepath

    def load_model(self, model_save_dir):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        model_save_name = 'model_state'
        state = torch.load(f=os.path.join(model_save_dir, "{}".format(model_save_name)))
        # saves state['network'] of model into model (self)
        self.load_state_dict(state_dict=state['network'])

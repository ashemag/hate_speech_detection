from comet_ml import OfflineExperiment
from collections import defaultdict, OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from sklearn.metrics import f1_score, precision_score, recall_score

from models.cnn import word_cnn
from utils import prepare_output_file
from globals import ROOT_DIR

LABEL_MAPPING = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}
DEBUG = True


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, device, hyper_params, train_data, valid_data, test_data, scheduler=None):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = hyper_params['experiment_name']
        self.model = network_model
        self.model.reset_parameters()
        self.device = device
        self.seed = hyper_params['seed']
        self.scheduler = scheduler

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu

        # re-initialize network parameters
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.optimizer = hyper_params['optimizer']

        # Generate the directory names
        self.experiment_folder = hyper_params['results_dir']
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_criteria = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = hyper_params['num_epochs']
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.starting_epoch = 0
        self.state = dict()

        # Comet visualizations
        if not DEBUG:
            self.experiment = OfflineExperiment(project_name=self.experiment_folder.split('/')[-1],
                                           workspace="ashemag",
                                           offline_directory="{}/{}".format(self.experiment_folder, 'comet'))
            self.experiment.set_filename('experiment_{}'.format(hyper_params['seed']))
            self.experiment.set_name('experiment_{}'.format(hyper_params['seed']))
            self.experiment.log_parameters(hyper_params)

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y, stats, experiment_key='train'):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        # sets model to training mode
        # (in case batch normalization or other methods have different procedures for training and evaluation)
        self.train()
        x = x.float()
        x = x.to(self.device)
        y = y.to(self.device)
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        out = self.model.forward(x)  # forward the data in the model
        # loss = F.cross_entropy(input=out, target=y)  # compute loss
        loss = self.criterion(out, y)
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        stats['{}_acc'.format(experiment_key)].append(accuracy)
        stats['{}_loss'.format(experiment_key)].append(loss.data.detach().cpu().numpy())
        self.compute_f_metrics(stats, y, predicted, experiment_key)

    def run_evaluation_iter(self, x, y, stats, experiment_key='valid'):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        with torch.no_grad():
            self.eval()  # sets the system to validation mode
            x = x.float()
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model.forward(x)  # forward the data in the model
            loss = self.criterion(out, y)

            # loss = F.cross_entropy(out, y)  # compute loss
            _, predicted = torch.max(out.data, 1)  # get argmax of predictions
            accuracy = np.mean(list(predicted.eq(y.data).cpu()))

            stats['{}_acc'.format(experiment_key)].append(accuracy)  # compute accuracy
            stats['{}_loss'.format(experiment_key)].append(loss.data.detach().cpu().numpy())
            self.compute_f_metrics(stats, y, predicted, experiment_key)

    def save_model(self, model_save_dir, model_save_name, model_idx):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        # Save state each epoch
        path = os.path.join(model_save_dir, "{}_{}.pt".format(model_save_name, str(model_idx)))
        torch.save(self.state_dict(), f=path)

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        """
        path = os.path.join(model_save_dir, "{}_{}.pt".format(model_save_name, str(model_idx)))
        self.load_state_dict(torch.load(f=path))

    def remove_excess_models(self):
        dir_list_list = [dir_names for (_, dir_names, _) in os.walk(self.experiment_folder)]
        for dir_list in dir_list_list:
            if 'saved_models' in dir_list:
                path = os.path.abspath(os.path.join(self.experiment_folder, 'saved_models'))
                file_list_list = [file_names for (_, _, file_names) in os.walk(path)]
                for file_list in file_list_list:
                    for file in file_list:
                        epoch = file.split('_')[-1]
                        epoch = epoch.replace('.pt', '')
                        if int(epoch) != self.best_val_model_idx:
                            os.remove(os.path.join(path, file))

    @staticmethod
    def compute_f_metrics(stats, y_true, predicted, type_key):

        f1score_overall = f1_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average='weighted'
        )

        stats[type_key + '_f_score'].append(f1score_overall)

        precision_overall = precision_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average='weighted'
        )

        stats[type_key + '_precision'].append(precision_overall)

        recall_overall = recall_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average='weighted'
        )

        stats[type_key + '_recall'].append(recall_overall)

        f1scores = f1_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average=None
        )
        precision = precision_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average=None
        )

        recall = recall_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average=None
        )

        for i in range(len(f1scores)):
            stats[type_key + '_f_score_' + LABEL_MAPPING[i]].append(f1scores[i])
            stats[type_key + '_precision_' + LABEL_MAPPING[i]].append(precision[i])
            stats[type_key + '_recall_' + LABEL_MAPPING[i]].append(recall[i])

    def save_best_performing_model(self, epoch_stats, epoch_idx):
        criteria = epoch_stats['valid_f_score_abusive'] + epoch_stats['valid_f_score_hateful']
        if criteria > self.best_val_model_criteria:  # if current epoch's mean val acc is greater than the saved best val acc then
            self.best_val_model_criteria = criteria  # set the best val model acc to be current epoch's val accuracy
            self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

    @staticmethod
    def iter_logs(stats, start_time, index):
        # Log results to terminal
        out_string = "".join(["{}: {:0.4f}\n".format(key, value)
                              for key, value in stats.items() if key != 'epoch'])
        epoch_elapsed_time = (time.time() - start_time) / 60  # calculate time taken for epoch
        epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
        print("\n===Epoch {}===\n{}===Elapsed time: {} mins===".format(index, out_string, epoch_elapsed_time))

    def run_experiment(self, round_param=4):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        train_stats = OrderedDict()
        for epoch_idx in range(self.num_epochs):
            epoch_start_time = time.time()
            epoch_stats = defaultdict(list)
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    self.run_train_iter(x=x, y=y, stats=epoch_stats)  # take a training iter step
                    pbar_train.update(1)
                    pbar_train.set_description(
                        "{} Epoch {}: loss: {:.4f}, accuracy: {:.4f}".format('Train', epoch_idx,
                                                                             epoch_stats['train_loss'][-1],
                                                                             epoch_stats['train_acc'][-1]))

            with tqdm.tqdm(total=len(self.valid_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.valid_data:  # get data batches
                    self.run_evaluation_iter(x=x, y=y, stats=epoch_stats)  # run a validation iter
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description(
                        "{} Epoch {}: loss: {:.4f}, accuracy: {:.4f}".format('Valid', epoch_idx,
                                                                             epoch_stats['valid_loss'][-1],
                                                                             epoch_stats['valid_acc'][-1]))

            # learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                epoch_stats['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # save to train stats
            for key, value in epoch_stats.items():
                epoch_stats[key] = np.mean(value)
                if not DEBUG:
                    self.experiment.log_metric(name=key, value=epoch_stats[key], step=epoch_idx)

            epoch_stats['epoch'] = epoch_idx
            train_stats["epoch_{}".format(epoch_idx)] = epoch_stats

            if DEBUG:
                self.iter_logs(epoch_stats, epoch_start_time, epoch_idx)

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model",
                            model_idx=epoch_idx)
            self.save_best_performing_model(epoch_stats=epoch_stats, epoch_idx=epoch_idx)

        ### EXPERIMENTS END ###
        # save train statistics
        prepare_output_file(filename="{}/{}".format(self.experiment_folder, "train_statistics_{}.csv".format(self.seed)),
                            output=list(train_stats.values()))

        print("Generating test set evaluation metrics with best model index {}".format(self.best_val_model_idx))

        self.load_model(model_save_dir=self.experiment_saved_models,
                        model_idx=self.best_val_model_idx,
                        model_save_name="train_model")

        test_stats = defaultdict(list)
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                self.run_evaluation_iter(x=x, y=y, stats=test_stats, experiment_key='test')
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description("loss: {:.4f}, accuracy: {:.4f}".format(test_stats['test_loss'][-1],
                                                                                  test_stats['test_acc'][-1]))
        # save to test stats
        for key, value in test_stats.items():
            test_stats[key] = np.mean(value)

        merge_dict = dict(list(test_stats.items()) +
                          list(train_stats["epoch_{}".format(self.best_val_model_idx)].items()))

        merge_dict['epoch'] = self.best_val_model_idx
        merge_dict['seed'] = self.seed
        merge_dict['title'] = self.experiment_name
        merge_dict['num_epochs'] = self.num_epochs

        for key, value in merge_dict.items():
            if isinstance(value, float):
                merge_dict[key] = np.around(value, 4)

        print(merge_dict)
        prepare_output_file(filename="{}/{}".format(self.experiment_folder, "results.csv"),
                            output=[merge_dict])

        self.remove_excess_models()
        return train_stats, test_stats

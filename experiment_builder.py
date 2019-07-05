from comet_ml import OfflineExperiment
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from sklearn.metrics import f1_score, precision_score, recall_score

from storage_utils import save_statistics

LABEL_MAPPING = {0: 'hateful', 1: 'abusive', 2: 'normal', 3: 'spam'}


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, device, hyper_params, train_data, valid_data, test_data):
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

        if torch.cuda.device_count() > 1:
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu

        # re-initialize network parameters
        self.train_data = train_data
        self.val_data = valid_data
        self.test_data = test_data
        self.optimizer = hyper_params['optimizer']

        # Generate the directory names
        self.experiment_folder = hyper_params['results_dir']
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_criteria = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = hyper_params['num_epochs']
        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        self.starting_epoch = 0
        self.state = dict()

        # Comet visualizations
        experiment = OfflineExperiment(project_name=self.experiment_folder.split('/')[-1],
                                       workspace="ashemag",
                                       offline_directory=self.experiment_folder)
        experiment.set_filename('experiment_{}'.format(hyper_params['seed']))
        experiment.set_name('experiment_{}'.format(hyper_params['seed']))
        experiment.log_parameters(hyper_params)

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

        x = x.to(self.device).float()
        y = y.to(self.device)

        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(input=out, target=y)  # compute loss

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        stats['{}_acc'.format(experiment_key)] = accuracy
        stats['{}_loss'.format(experiment_key)] = loss.data.detach().cpu().numpy()
        self.compute_f_metrics(stats, y, predicted, 'train')

    def run_evaluation_iter(self, x, y, stats, experiment_key='valid'):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode

        x = x.to(self.device).float()
        y = y.to(self.device)
        out = self.model.forward(x)  # forward the data in the model
        loss = F.cross_entropy(out, y)  # compute loss
        _, predicted = torch.max(out.data, 1)  # get argmax of predictions
        stats['{}_acc'.format(experiment_key)] = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy
        stats['{}_loss'.format(experiment_key)] = loss.data.detach().cpu().numpy()
        self.compute_f_metrics(stats, y, predicted, experiment_key)

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
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
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_val_model_idx'], state['best_val_model_criteria'], state

    @staticmethod
    def compute_f_metrics(stats, y_true, predicted, type_key):

        f1score_overall = f1_score(
            y_true.cpu().detach().numpy(),
            predicted.cpu().detach().numpy(),
            average='weighted'
        )

        stats[type_key + '_f_score'] = f1score_overall

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
            stats[type_key + '_f_score_' + LABEL_MAPPING[i]] = f1scores[i]
            stats[type_key + '_precision_' + LABEL_MAPPING[i]] = precision[i]
            stats[type_key + '_recall_' + LABEL_MAPPING[i]] = recall[i]

    def save_best_performing_model(self, epoch_stats, epoch_idx):
        metrics_to_aggregate = [epoch_stats[item] for item in ['valid_f_score_abusive', 'valid_f_score_hateful']]
        metrics = [np.mean(metric) for metric in metrics_to_aggregate]
        criteria = np.mean(metrics)
        if criteria > self.best_val_model_criteria:  # if current epoch's mean val acc is greater than the saved best val acc then
            self.best_val_model_criteria = criteria  # set the best val model acc to be current epoch's val accuracy
            self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        train_stats = defaultdict(list)
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            epoch_stats = defaultdict(list)

            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for idx, (x, y) in enumerate(self.train_data):  # get data batches
                    self.run_train_iter(x=x, y=y, stats=epoch_stats)  # take a training iter step
                    pbar_train.update(1)
                    pbar_train.set_description("loss: {:.4f}, accuracy: {:.4f}".format(epoch_stats['train_loss'],
                                                                                       epoch_stats['train_acc']))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for x, y in self.val_data:  # get data batches
                    self.run_evaluation_iter(x=x, y=y, stats=epoch_stats)  # run a validation iter
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss: {:.4f}, accuracy: {:.4f}".format(epoch_stats['valid_loss'],
                                                                                     epoch_stats['valid_acc']))

            self.save_best_performing_model(epoch_stats, epoch_idx)

            # get mean of all metrics of current epoch metrics dict,
            # to get them ready for storage and output on the terminal.
            for key, value in epoch_stats.items():
                train_stats[key].append(np.mean(value))

            train_stats['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=train_stats, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # Log results to terminal
            out_string = "_".join(["{}_{:.2f}".format(key, np.mean(value)) for key, value in epoch_stats.items()])
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")

            # Save state each epoch
            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_val_model_criteria'] = self.best_val_model_criteria
            self.state['best_val_model_idx'] = self.best_val_model_idx

            self.save_model(model_save_dir=self.experiment_saved_models,
                            model_save_name="train_model",
                            model_idx=epoch_idx, state=self.state)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")

        test_stats_run = defaultdict(list)
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y in self.test_data:  # sample batch
                self.run_evaluation_iter(x=x, y=y, stats=test_stats_run, experiment_key='test')
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description("loss: {:.4f}, accuracy: {:.4f}".format(test_stats_run['test_loss'],
                                                                                  test_stats_run['test_acc']))

        test_stats = {key: [np.mean(value)] for key, value in test_stats_run.items()}
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        stats_dict=test_stats, current_epoch=0, continue_from_mode=False)

        return train_stats, test_stats

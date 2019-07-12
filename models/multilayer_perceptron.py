from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_layers, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(MLP, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers

        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers
        out = x
        print("Building basic block of network using input shape {}".format(out.shape))
        if len(out.shape) > 2:
            out = out.permute([0, 2, 1])
            out = F.max_pool1d(out, out.shape[-1])
            out = out.view(out.shape[0], -1)

        for i in range(self.num_layers):
            if i > 0:
                out = F.leaky_relu(out)
            self.layer_dict['linear_{}'.format(i)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                                               out_features=self.num_output_classes,
                                                               bias=self.use_bias)
            out = self.layer_dict['linear_{}'.format(i)](out)
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        if len(out.shape) > 2:
            out = out.permute([0, 2, 1])
            out = F.max_pool1d(out, out.shape[-1])
            out = out.view(out.shape[0], -1)

        for i in range(self.num_layers):
            if i > 0:
                out = F.leaky_relu(out)
            out = self.layer_dict['linear_{}'.format(i)](out)
        return out

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass


def multi_layer_perceptron(input_shape):
    return MLP(num_output_classes=4,
               num_layers=3,
               input_shape=input_shape)

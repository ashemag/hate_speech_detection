from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_output_classes, num_layers, use_bias=False):
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
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers

        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()

    def build_fc_layer(self, input_shape):
        self.layer_dict['fc_layer'] = nn.Linear(in_features=input_shape[1],  # add a linear layer
                                                out_features=self.num_output_classes,
                                                bias=self.use_bias)

    def build_layers(self, input_shape, layer_key='first'):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros(input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = x
        if len(out.shape) > 2:
            out = F.max_pool1d(out, out.shape[-1])
            out = out.view(out.shape[0], -1)
        print("Building basic block of network using input shape {} and layer key {}".format(out.shape, layer_key))

        for i in range(self.num_layers):
            self.layer_dict['linear_{}_{}'.format(i, layer_key)] = nn.Linear(in_features=out.shape[1],
                                                                             # add a linear layer
                                                                             out_features=out.shape[1],
                                                                             bias=self.use_bias)
            out = self.layer_dict['linear_{}_{}'.format(i, layer_key)](out)
            out = F.leaky_relu(out)

        if len(out.shape) == 2:
            out_shape = out.reshape(out.shape[0], out.shape[1], 1).shape

        self.layer_dict['fc_layer_{}'.format(layer_key)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                                                     out_features=self.num_output_classes,
                                                                     bias=self.use_bias)
        out = self.layer_dict['fc_layer_{}'.format(layer_key)](out)
        print("Block is built, output volume is {}".format(out.shape))
        return out_shape

    def forward(self, x, layer_key='first', flatten_flag=True):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = x
        if len(out.shape) > 2:
            out = F.max_pool1d(out, out.shape[-1])
            out = out.view(out.shape[0], -1)

        for i in range(self.num_layers):
            out = self.layer_dict['linear_{}_{}'.format(i, layer_key)](out)
            out = F.leaky_relu(out)

        if not flatten_flag:
            out = out.reshape(out.shape[0], out.shape[1], 1)

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


def multi_layer_perceptron():
    return MLP(num_output_classes=4,
               num_layers=3)

from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F

DILATION_PARAM = 1.4
WEIGHT_DECAY = 1e-4


class CNN(nn.Module):
    def __init__(self,
                 num_output_classes,
                 num_filters,
                 num_layers,
                 dropout=.5,
                 input_shape_context=None,
                 use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(CNN, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape_context = input_shape_context
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout, inplace=False)

        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.output_shape = None

    @staticmethod
    def process(out):
        if len(out.shape) > 2:
            out = out.permute([0, 2, 1])
        else:
            out = out.reshape(out.shape[0], out.shape[1], 1)
        return out

    def build_fc_layer(self, input_shape):
        self.layer_dict['fc_layer'] = nn.Linear(in_features=input_shape[1],  # add a linear layer
                                                out_features=self.num_output_classes,
                                                bias=self.use_bias)

    def build_layers(self, input_shape, layer_key='first'):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros(input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = self.process(x)
        print("Building basic block of Convolutional Network using input shape {} for layer key {}".format(out.shape, layer_key))
        context_list = []

        for i in range(self.num_layers):  # for number of layers times
            if i > 0:
                # to give every layer access to all prev layers (DenseNet connectivity)
                out = torch.cat(context_list, dim=1)

            dilation = int(DILATION_PARAM**i)
            self.layer_dict['conv_{}_{}'.format(i, layer_key)] = nn.Conv1d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters[i],
                                                             padding=dilation,
                                                             bias=False,
                                                             dilation=dilation)

            out = self.layer_dict['conv_{}_{}'.format(i, layer_key)](out)  # use layer on inputs to get an output
            self.layer_dict['batch_norm_{}_{}'.format(i, layer_key)] = nn.BatchNorm1d(num_features=out.shape[1])
            out = self.layer_dict['batch_norm_{}_{}'.format(i, layer_key)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.max_pool1d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)
        self.layer_dict['fc_layer_{}'.format(layer_key)] = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                  out_features=self.num_output_classes,
                                  bias=self.use_bias)

        out = self.layer_dict['fc_layer_{}'.format(layer_key)](out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is {} for layer key {}".format(out.shape, layer_key))

    def forward(self, x, layer_key='first', flatten_flag=True):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = self.process(x)
        context_list = []
        for i in range(self.num_layers):  # for number of layers times
            if i > 0:
                out = torch.cat(context_list, dim=1)
            out = self.layer_dict['conv_{}_{}'.format(i, layer_key)](out)  # use layer on inputs to get an output
            out = self.layer_dict['batch_norm_{}_{}'.format(i, layer_key)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)

        if flatten_flag:
            out = F.max_pool1d(out, out.shape[-1])
            out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        else: #don't flatten but max pool
            out = F.max_pool1d(out, out.shape[-1])
            out = out.permute(0, 2, 1)

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


def word_cnn(dropout=.5):
    return CNN(num_output_classes=4,
               num_filters=[8, 8, 8],
               num_layers=3,
               dropout=dropout)



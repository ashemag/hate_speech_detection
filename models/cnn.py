from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F

DILATION_PARAM = 1.4
WEIGHT_DECAY = 1e-4


class CNN(nn.Module):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, dropout=.5, use_bias=False):
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
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=dropout, inplace=False)
        self.logit_linear_layer = None
        # initialize a module dict, which is effectively a dictionary that can collect layers and integrate them into pytorch
        self.layer_dict = nn.ModuleDict()

        # build the network
        self.build_module()

    @staticmethod
    def process(out):
        if len(out.shape) > 2:
            out = out.permute([0, 2, 1])
        else:
            out = out.reshape(out.shape[0], out.shape[1], 1)
        return out

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros(self.input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = self.process(x)
        print("Building basic block of Convolutional Network using input shape", out.shape)
        context_list = []

        for i in range(self.num_layers):  # for number of layers times
            if i > 0:
                # to give every layer access to all prev layers (DenseNet connectivity)
                out = torch.cat(context_list, dim=1)

            dilation = int(DILATION_PARAM**i)
            self.layer_dict['conv_{}'.format(i)] = nn.Conv1d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters[i],
                                                             padding=dilation,
                                                             bias=False,
                                                             dilation=dilation)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            self.layer_dict['batch_norm_{}'.format(i)] = nn.BatchNorm1d(num_features=out.shape[1])
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.max_pool1d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)

        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)

        out = self.logit_linear_layer(out)  # apply linear layer on flattened inputs
        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
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
            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.max_pool1d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.logit_linear_layer(out)  # pass through a linear layer to get logits/preds

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

        self.logit_linear_layer.reset_parameters()


def word_cnn(input_shape, dropout=.5):
    return CNN(num_output_classes=4,
               num_filters=[8, 8, 8],
               num_layers=3,
               input_shape=input_shape,
               dropout=dropout)



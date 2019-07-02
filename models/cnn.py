import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import Network

DILATION_PARAM = 1.4
CHARACTER_LAYERS = 2
WEIGHT_DECAY = 1e-4


class CNN(Network):
    def __init__(self, input_shape, dim_reduction_type, num_output_classes, num_filters, num_layers, use_bias=False):
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
        self.dim_reduction_type = dim_reduction_type
        self.drop = nn.Dropout(p=0.5, inplace=False)

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
        out = out.permute([0, 2, 1])
        print("Building basic block of ConvolutionalNetwork using input shape", out.shape)

        context_list = []
        for i in range(5):  # for number of layers times
            if i > 0:
                # to give every layer access to all prev layers (DenseNet connectivity)
                out = torch.cat(context_list, dim=1)
            dilation = int(DILATION_PARAM**i)
            # dilation = 1
            self.layer_dict['conv_{}'.format(i)] = nn.Conv1d(in_channels=out.shape[1],
                                                             # add a conv layer in the module dict
                                                             kernel_size=3,
                                                             out_channels=self.num_filters, padding=dilation,
                                                             bias=False, dilation=dilation)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            self.layer_dict['batch_norm_{}'.format(i)] = nn.BatchNorm1d(num_features=out.shape[1])
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)  # apply relu
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.avg_pool1d(out, out.shape[-1])
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
        out = x
        out = out.permute([0, 2, 1])
        context_list = []
        for i in range(5):  # for number of layers times
            if i > 0:
                out = torch.cat(context_list, dim=1)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.avg_pool1d(out, out.shape[-1])
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

class CharacterCNN(Network):
    def __init__(self, input_shape, num_output_classes, num_filters, num_layers, kernel_size=3, use_bias=False):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(CharacterCNN, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.drop = nn.Dropout(p=0.5, inplace=False)
        self.kernel_size = kernel_size
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
        print("build module input {}".format(out.shape))
        out = out.permute([0, 1, 3, 2])
        out = out.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))

        ### CHARACTER EMBEDDING: embedded doc shape (58358, 17, 10, 69)
        for i in range(CHARACTER_LAYERS):
            self.layer_dict['conv_{}_{}'.format('char', i)] = nn.Conv1d(in_channels=out.shape[1],
                                                                  kernel_size=self.kernel_size,
                                                                  out_channels=self.num_filters,
                                                                  padding=1,
                                                                  bias=False,
                                                                  dilation=1)

            out = self.layer_dict['conv_{}_{}'.format('char', i)](out)
            out = F.leaky_relu(out)  # apply relu

        out = F.avg_pool1d(out, out.shape[-1])
        print(out.shape)
        # torch.Size([634916, 64, 1])
        # out shape (58358, 17, 10)

        out = out.reshape((x.shape[0], x.shape[1], self.num_filters))
        out = out.permute([0, 2, 1])
        print("Building basic block of ConvolutionalNetwork using input shape", out.shape)
        context_list = []
        for i in range(5):  # for number of layers times
            if i > 0:
                # to give every layer access to all prev layers (DenseNet connectivity)
                out = torch.cat(context_list, dim=1)

            dilation = int(DILATION_PARAM**i)
            #dilation = 1
            self.layer_dict['conv_{}'.format(i)] = nn.Conv1d(in_channels=out.shape[1],
                                                             kernel_size=self.kernel_size,
                                                             out_channels=self.num_filters,
                                                             padding=dilation,
                                                             bias=False,
                                                             dilation=dilation)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            self.layer_dict['batch_norm_{}'.format(i)] = nn.BatchNorm1d(num_features=out.shape[1])
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)  # apply relu
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.avg_pool1d(out, out.shape[-1])
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
        out = x
        out = out.permute([0, 1, 3, 2])
        out = out.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))

        for i in range(CHARACTER_LAYERS):
            out = self.layer_dict['conv_{}_{}'.format('char',i)](out)
            out = F.leaky_relu(out)

        out = F.avg_pool1d(out, out.shape[-1])
        out = out.reshape((x.shape[0], x.shape[1], self.num_filters))
        out = out.permute([0, 2, 1])

        context_list = []
        for i in range(5):  # for number of layers times
            if i > 0:
                out = torch.cat(context_list, dim=1)

            out = self.layer_dict['conv_{}'.format(i)](out)  # use layer on inputs to get an output
            out = self.layer_dict['batch_norm_{}'.format(i)](out)
            out = F.leaky_relu(out)
            out = self.drop(out)
            context_list.append(out)

        out = torch.cat(context_list, dim=1)
        out = F.avg_pool1d(out, out.shape[-1])
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


def word_cnn(input_shape):
    model = CNN(num_output_classes=4,
                num_filters=64,
                num_layers=3,
                dim_reduction_type='max_pooling',
                input_shape=input_shape)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    return model, criterion, optimizer


def character_cnn(input_shape):
    model = CharacterCNN(num_output_classes=4,
                         num_filters=8,
                         num_layers=3,
                         kernel_size=3,
                         input_shape=input_shape)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    return model, criterion, optimizer

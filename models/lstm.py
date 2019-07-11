from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F

DILATION_PARAM = 1.4

class LSTM(nn.Module):
    def __init__(self, input_shape, num_hidden_layers, dropout=.5, use_bias=False, num_output_classes=4):
        """
        Initializes a convolutional network module object.
        :param input_shape: The shape of the inputs going in to the network.
        :param dim_reduction_type: The type of dimensionality reduction to apply after each convolutional stage, should be one of ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']
        :param num_output_classes: The number of outputs the network should have (for classification those would be the number of classes)
        :param num_filters: Number of filters used in every conv layer, except dim reduction stages, where those are automatically infered.
        :param num_layers: Number of conv layers (excluding dim reduction stages)
        :param use_bias: Whether our convolutions will use a bias.
        """
        super(LSTM, self).__init__()
        # set up class attributes useful in building the network and inference
        self.input_shape = input_shape
        self.num_hidden_layers = num_hidden_layers
        self.num_output_classes = num_output_classes
        self.use_bias = use_bias
        self.logit_linear_layer = None
        self.layer_dict = nn.ModuleDict()
        # build the network
        self.build_module()

    @staticmethod
    def process(out):
        if len(out.shape) == 2:
            out = out.reshape(out.shape[0], out.shape[1], 1)
        out = out.permute([2, 0, 1])
        return out

    def build_module(self):
        """
        Builds network whilst automatically inferring shapes of layers.
        """
        x = torch.zeros(self.input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = self.process(x)
        print("Building basic block of LSTM using input shape", out.shape)
        print(out.shape)
        # expects (seq_len, batch, input_size)
        self.layer_dict['lstm'] = nn.LSTM(input_size=out.shape[-1],
                                          hidden_size=self.num_hidden_layers,
                                          bias=self.use_bias,
                                          num_layers=1,
                                          dropout=.5)

        out, _ = self.layer_dict['lstm'](out)
        out = out.permute(1, 0, 2)
        out = out.contiguous().view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        self.logit_linear_layer = nn.Linear(in_features=out.shape[1],  # add a linear layer
                                            out_features=self.num_output_classes,
                                            bias=self.use_bias)

        print("Block is built, output volume is", out.shape)
        return out

    def forward(self, x):
        """
        Forward propages the network given an input batch
        :param x: Inputs x (b, c, h, w)
        :return: preds (b, num_classes)
        """
        out = self.process(x)
        out, _ = self.layer_dict['lstm'](out)
        out = out.permute(1, 0, 2)
        out = out.contiguous().view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
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


def lstm(input_shape):
    return LSTM(num_output_classes=4,
                num_hidden_layers=10,
                input_shape=input_shape)

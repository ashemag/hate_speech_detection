import torch
import torch.nn as nn
from models.base import Network
import torch.nn.functional as F


class FCNetworkTDIDF(Network):
    def __init__(self, input_shape, num_output_classes, num_layers):
        super(FCNetworkTDIDF, self).__init__()
        self.num_output_classes = num_output_classes
        self.layer_dict = nn.ModuleDict()
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.build_module()

    def build_module(self):
        x = torch.zeros(self.input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = x
        print('[Build module] Initial input shape is {}'.format(out.shape))
        print(out.shape)
        for i in range(self.num_layers):
            if i != 0:
                out = F.leaky_relu(out)
            self.layer_dict['linear_{}'.format(i)] = nn.Linear(out.shape[1], self.num_output_classes)
            out = self.layer_dict['linear_{}'.format(i)](out)

        print("[Build module] Final output shape is {}".format(out.shape))
        return out

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            if i != 0:
                out = F.leaky_relu(out)
            out = self.layer_dict['linear_{}'.format(i)](out)

        return out


def fc_linear_tdidf(input_shape, num_output_classes=4):
    model = FCNetworkTDIDF(input_shape, num_output_classes, num_layers=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)
    return model, criterion, optimizer

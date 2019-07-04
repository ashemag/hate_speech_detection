import torch
import torch.nn as nn
from models.base import Network
import torch.nn.functional as F


class LogisticRegression(Network):
    def __init__(self, input_shape, num_output_classes, num_layers):
        super(LogisticRegression, self).__init__()
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
        # for i in range(self.num_layers):
        #     self.layer_dict['linear_{}'.format(i)] = nn.Linear(out.shape[1], self.num_output_classes)
        #     out = self.layer_dict['linear_{}'.format(i)](out)
        #     out = F.leaky_relu(out)

        # # make linear
        self.layer_dict['linear'] = nn.Linear(out.shape[1], self.num_output_classes)
        out = self.layer_dict['linear'](out)
        out = F.log_softmax(out)

        # print("[Build module] Final output shape is {}".format(out.shape))
        return out

    def forward(self, x):
        out = x
        # for i in range(self.num_layers):
        #     out = self.layer_dict['linear_{}'.format(i)](out)
        #     out = F.leaky_relu(out)

        # make linear
        out = self.layer_dict['linear'](out)
        out = F.log_softmax(out)
        return out


def logistic_regression(input_shape, num_output_classes):
    model = LogisticRegression(input_shape, num_output_classes, num_layers=1)
    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.NLLLoss()  # same as CELoss except it does the log softmax for you.

    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=WEIGHT_DECAY)
    return model, criterion, optimizer


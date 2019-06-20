import torch
import torch.nn as nn
from models.base import Network
import torch.nn.functional as F

NUM_LAYERS = 1


class LogisticRegression(Network):
    def __init__(self, input_shape, num_output_classes):
        super(LogisticRegression, self).__init__()
        self.num_output_classes = num_output_classes
        self.layer_dict = nn.ModuleDict()
        self.input_shape = input_shape
        self.build_module()

    def build_module(self):
        x = torch.zeros(self.input_shape)  # create dummy inputs to be used to infer shapes of layers
        out = x
        print('input shape is ', out.shape)
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        for i in range(NUM_LAYERS):
            self.layer_dict['linear_{}'.format(i)] = nn.Linear(out.shape[1], self.num_output_classes)
            out = self.layer_dict['linear_{}'.format(i)](out)

        out = F.relu(out)
        return out

    def forward(self, x):
        out = x
        out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        for i in range(NUM_LAYERS):
            out = self.layer_dict['linear_{}'.format(i)](out)

        out = F.relu(out)
        return out

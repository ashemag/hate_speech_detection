import torch
import torch.nn as nn
from models.base import Network
import torch.nn.functional as F

class LogisticRegression(Network):
    def __init__(self, input_shape, n_class):
        super(LogisticRegression, self).__init__()
        self.num_output_classes = n_class
        self.layer_dict = nn.ModuleDict()
        self.input_shape = input_shape
        self.build_module()

    def build_module(self):
        x = torch.zeros((self.input_shape))  # create dummy inputs to be used to infer shapes of layers
        out = x
        self.layer_dict['linear'] = nn.Linear(out.shape[1], 1)

        return out

    def forward(self, x):
        out = x
        # out = out.view(out.shape[0], -1)  # flatten outputs from (b, c, h, w) to (b, c*h*w)
        out = self.layer_dict['linear'](out)
        # out = F.relu(out)
        return out

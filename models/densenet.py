'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import Network
import numpy as np

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(4*growth_rate)
        self.conv2 = nn.Conv1d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_planes)
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool1d(out, out.shape[-1])
        return out


class DenseNet(Network):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_output_classes=4):
        super(DenseNet, self).__init__()

        self.growth_rate = growth_rate
        self.num_output_classes = num_output_classes
        self.nblocks = nblocks
        self.reduction=reduction
        self.block=block
        self.layer_dict = nn.ModuleDict()

    def build_layers(self, input_shape, layer_key='first'):
        x = torch.zeros(input_shape)
        out = self.process(x)

        num_planes = 2 * self.growth_rate
        self.layer_dict['conv_1_{}'.format(layer_key)] = nn.Conv1d(out.shape[1], num_planes, kernel_size=3, padding=1, bias=False)

        self.layer_dict['dense_1_{}'.format(layer_key)] = self._make_dense_layers(self.block, num_planes, self.nblocks[0])
        num_planes += self.nblocks[0] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.layer_dict['trans_1_{}'.format(layer_key)] = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.layer_dict['dense_2_{}'.format(layer_key)] = self._make_dense_layers(self.block, num_planes, self.nblocks[1])
        num_planes += self.nblocks[1] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.layer_dict['trans_2_{}'.format(layer_key)] = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.layer_dict['dense_3_{}'.format(layer_key)] = self._make_dense_layers(self.block, num_planes, self.nblocks[2])
        num_planes += self.nblocks[2] * self.growth_rate
        out_planes = int(math.floor(num_planes * self.reduction))
        self.layer_dict['trans_3_{}'.format(layer_key)] = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.layer_dict['dense_4_{}'.format(layer_key)] = self._make_dense_layers(self.block, num_planes, self.nblocks[3])
        num_planes += self.nblocks[3] * self.growth_rate

        self.layer_dict['bn_{}'.format(layer_key)] = nn.BatchNorm1d(num_planes)
        self.layer_dict['fc_layer_{}'.format(layer_key)] = nn.Linear(num_planes, self.num_output_classes)

        #forward pass
        out = self.layer_dict['conv_1_{}'.format(layer_key)](out)
        out = self.layer_dict['trans_1_{}'.format(layer_key)](self.layer_dict['dense_1_{}'.format(layer_key)](out))
        out = self.layer_dict['trans_2_{}'.format(layer_key)](self.layer_dict['dense_2_{}'.format(layer_key)](out))
        out = self.layer_dict['trans_3_{}'.format(layer_key)](self.layer_dict['dense_3_{}'.format(layer_key)](out))
        out = self.layer_dict['dense_4_{}'.format(layer_key)](out)
        out = F.avg_pool1d(F.relu(self.layer_dict['bn_{}'.format(layer_key)](out)), out.shape[-1])

        out = out.permute([0, 2, 1])

        return out.shape

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    @staticmethod
    def process(out):
        if len(out.shape) > 2:
            out = out.permute([0, 2, 1])
        else:
            out = out.reshape(out.shape[0], out.shape[1], 1)
        return out

    def forward(self, x, layer_key='first', flatten_flag=True):
        out = self.process(x)
        out = self.layer_dict['conv_1_{}'.format(layer_key)](out)
        out = self.layer_dict['trans_1_{}'.format(layer_key)](self.layer_dict['dense_1_{}'.format(layer_key)](out))
        out = self.layer_dict['trans_2_{}'.format(layer_key)](self.layer_dict['dense_2_{}'.format(layer_key)](out))
        out = self.layer_dict['trans_3_{}'.format(layer_key)](self.layer_dict['dense_3_{}'.format(layer_key)](out))
        out = self.layer_dict['dense_4_{}'.format(layer_key)](out)
        out = F.avg_pool1d(F.relu(self.layer_dict['bn_{}'.format(layer_key)](out)), out.shape[-1])
        if flatten_flag:
            out = out.view(out.size(0), -1)
        else:
            out = out.permute([0, 2, 1])
        return out

    def reset_parameters(self):
        pass

# def DenseNet121():
#     return DenseNet(Bottleneck, [4,4,4,4], growth_rate=16)
#
# def DenseNet169():
#     return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)
#
# def DenseNet201():
#     return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)
#
# def DenseNet161():
#     return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)


def densenet():
    return DenseNet(Bottleneck, [4,4,4,4], growth_rate=12)

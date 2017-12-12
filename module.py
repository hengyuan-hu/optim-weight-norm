import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WnLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(WnLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.g = nn.Parameter(torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.g.data[0] = torch.norm(self.weight).data[0]
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def get_param_v(self):
        return self.weight

    def get_param_g(self):
        return self.g

    def forward(self, input):
        normed_weight = self.weight / torch.norm(self.weight)
        if self.g is not None:
            weight = normed_weight * self.g
        else:
            weight = normed_weight
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


import collections
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)



class WnConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(WnConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.output_padding = _pair(0)
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *self.kernel_size))
        self.g = nn.Parameter(torch.ones(1))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        # self.conv_ = nn.Conv2d(
        #     in_channels, out_channels, kernel_size, stride,
        #     padding, dilation, groups, bias)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        self.g.data[0] = torch.norm(self.weight).data[0]

    def get_param_v(self):
        return self.weight

    def get_param_g(self):
        return self.g

    def forward(self, input):
        normed_weight = self.weight / torch.norm(self.weight)
        if self.g is not None:
            weight = normed_weight * self.g
        else:
            weight = normed_weight
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)



class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def get_param_v(self):
        ret = []
        for module in self._modules.values():
            if type(module) == WnLinear or type(module) == WnConv2d:
                ret.append(module.get_param_v())
        return ret

    def get_param_g(self):
        ret = []
        for module in self._modules.values():
            if type(module) == WnLinear or type(module) == WnConv2d:
                ret.append(module.get_param_g())
        return ret

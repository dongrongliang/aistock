# -*- coding: utf-8 -*-
# !/usr/bin/env python


import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import init
import math
""""""""""""""""""""""""""""""""" half conv """""""""""""""""""""""""""""""""


class _halfConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: int
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_halfConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.half_kernel = kernel_size // 2 + 1
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * 1
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(1 - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, self.half_kernel), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, self.half_kernel), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.weights_padding = torch.zeros((out_channels, in_channels // groups, self.half_kernel - 1))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
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
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_halfConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class halfConv1d(_halfConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super(halfConv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride_, padding_, dilation_,
            False, _single(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        full_weight = torch.cat((self.weight, self.weights_padding.to(self.weight.device)), dim=-1)
        return self._conv_forward(input, full_weight, self.bias)

class Conv1DHalf(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, stride=1, dilation=1, groups=1):
        super().__init__()
        assert kernel >= 2
        self.filters = out_chan
        self.full_kernel = (kernel - 1) * 2 + 1
        self.stride = stride
        self.dilation = dilation
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.groups = 1
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel)))
        self.weights_padding = torch.zeros((out_chan, in_chan, kernel - 1))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x):
        b, c, l = x.shape

        weights = torch.cat((self.weight, self.weights_padding), dim=-1)

        padding = self._get_same_padding(l, self.full_kernel, self.dilation, self.stride)
        x = F.conv1d(x, weights, padding=padding, groups=self.groups, dilation=self.dilation)
        return x

def make_halfconv(in_dim, out_dim, kernel_size=3, stride=1,
              padding=1, activation=None, norm_type='', groups=1, use_bias=True):
    layer = []
    if norm_type == 'SN':
        layer += [nn.utils.spectral_norm(
            halfConv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias))]
    elif norm_type in ('BN'):
        layer += [halfConv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.BatchNorm1d(out_dim, momentum=0.8, eps=1e-3)]
    elif norm_type == 'ABN':
        layer += [halfConv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.BatchNorm1d(out_dim, momentum=0.01)]
    elif norm_type == 'IN':
        layer += [halfConv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.InstanceNorm1d(out_dim, affine=False, track_running_stats=False)]
    elif norm_type == 'WN':
        layer += [nn.utils.weight_norm(
            halfConv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias))]
    else:
        layer += [halfConv1d(in_dim, out_dim, kernel_size, stride, padding, bias=use_bias)]

    if activation is not None:
        layer += [activation]

    return nn.Sequential(*layer)

""""""""""""""""""""""""""""""""" act """""""""""""""""""""""""""""""""
class upsampling1d(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super(upsampling1d, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, input: Tensor):
        input = self.upsample(input.unsqueeze(dim=3)).mean(dim=3)
        return input


def _get_conv_act(activation, _inplace=True):
    """Return an activation function given a string"""

    # inplace only change placeholder synchronously in this situation:
    # res = x
    # x = act(x)
    # res == x
    # will not change follow by operations:
    # res = x
    # x = conv2D(x)
    # x = act(x)
    # res != x

    if activation == 'selu':
        activation = nn.SELU()
    elif activation == 'prelu':
        activation = nn.PReLU()
    elif activation == 'lrelu':
        activation = nn.LeakyReLU(negative_slope=0.2, inplace=_inplace)
    elif activation == 'relu':
        activation = nn.ReLU(inplace=_inplace)
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        activation = None

    return activation


""""""""""""""""""""""""""""""""" linear """""""""""""""""""""""""""""""""


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, drop_out=0.1,
                 norm_flag=True, transit_flag=False, activation='lrelu'):
        super(FFN, self).__init__()

        self.transit_flag = input_dim != output_dim or transit_flag
        self.norm_flag = norm_flag
        self.drop_flag = drop_out > 0

        if self.transit_flag:
            self.transit_linear = nn.Linear(input_dim, output_dim)
            self.activation_transit = nn.ReLU()
            if norm_flag:
                self.transit_norm = nn.LayerNorm(output_dim)

        self.linear1 = nn.Linear(output_dim, output_dim * 2)
        self.activation_ffn = _get_conv_act(activation)
        self.linear2 = nn.Linear(output_dim * 2, output_dim)

        if self.drop_flag:
            self.dropout1 = nn.Dropout(drop_out)
            self.dropout2 = nn.Dropout(drop_out)

        if norm_flag:
            self.ffn_norm = nn.LayerNorm(output_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):

        if self.transit_flag:
            x = self.activation_transit(self.transit_linear(x))
            if self.norm_flag:
                x = self.transit_norm(x)

        if self.drop_flag:
            ffn = self.linear2(self.dropout1(self.activation_ffn(self.linear1(x))))
            ffn = x + self.dropout2(ffn)
        else:
            ffn = self.linear2(self.activation_ffn(self.linear1(x)))
            ffn = x + ffn

        if self.norm_flag:
            ffn = self.ffn_norm(ffn)

        return ffn


""""""""""""""""""""""""""""""""" conv """""""""""""""""""""""""""""""""


def make_conv(in_dim, out_dim, kernel_size=3, stride=1,
              padding=1, activation=None, norm_type='', groups=1, use_bias=True):
    layer = []
    if norm_type == 'SN':
        layer += [nn.utils.spectral_norm(
            nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias))]
    elif norm_type in ('BN'):
        layer += [nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.BatchNorm1d(out_dim, momentum=0.8, eps=1e-3)]
    elif norm_type == 'ABN':
        layer += [nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.BatchNorm1d(out_dim, momentum=0.01)]
    elif norm_type == 'IN':
        layer += [nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias),
                  nn.InstanceNorm1d(out_dim, affine=False, track_running_stats=False)]
    elif norm_type == 'WN':
        layer += [nn.utils.weight_norm(
            nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, groups=groups, bias=use_bias))]
    else:
        layer += [nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias=use_bias)]

    if activation is not None:
        layer += [activation]

    return nn.Sequential(*layer)


""""""""""""""""""""""""""""""""" Resnet """""""""""""""""""""""""""""""""


class HrBaseBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU(True), use_bias=False):
        super(HrBaseBlock, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=use_bias)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=use_bias)

        self.activation = activation

        if inplanes != planes:
            self.residual_transition = nn.Conv1d(inplanes, planes, 1, stride, bias=use_bias)
            self.transition_flag = True
        else:
            self.transition_flag = False

    def forward(self, x):
        # use inplace activation which will not change residual after conv
        residual = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)

        if self.transition_flag:
            residual = self.residual_transition(x)

        out += residual
        out = self.activation(out)

        return out


class HrBottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, activation=nn.ReLU(True), expansion=4, use_bias=False):
        super(HrBottleneck, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=use_bias)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=use_bias)

        self.conv3 = nn.Conv1d(planes, planes * expansion, kernel_size=1,
                               bias=use_bias)

        self.activation = activation

        if inplanes != planes * expansion:
            self.residual_transition = nn.Conv1d(inplanes, planes * expansion, 1, stride, bias=use_bias)
            self.transition_flag = True
        else:
            self.transition_flag = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.activation(out)

        out = self.conv3(out)

        if self.transition_flag:
            residual = self.residual_transition(x)

        out += residual
        out = self.activation(out)

        return out


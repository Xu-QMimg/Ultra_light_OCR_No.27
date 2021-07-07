# 加入CTCHead2


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F


def get_para_bias_attr(l2_decay, k, name):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [weight_attr, bias_attr]


class CTCHead(nn.Layer):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead, self).__init__()
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc')
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='ctc_fc')
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
        return predicts

class CTCHead_kd(nn.Layer):
    def __init__(self, in_channels, out_channels, fc_decay=0.0004, **kwargs):
        super(CTCHead_kd, self).__init__()
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc_kd')
        self.fc = nn.Linear(
            in_channels,
            out_channels,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='ctc_fc_kd')
        self.out_channels = out_channels

    def forward(self, x, labels=None):
        predicts = self.fc(x)
        if not self.training:
            predicts = F.softmax(predicts, axis=2)
        return predicts
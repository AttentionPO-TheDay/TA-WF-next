"""Explicit PyTorch port of the official Var-CNN direction-only dilated model.

Source: sanjit-bhat/Var-CNN, commit e5db76a86fcbf8839f764b9aca241749d1c1f700,
var_cnn.py (frozen in the preceding multimodel audit run). No old project imports.
ResNet18 [2,2,2,2], causal dilations [(1,2),(4,8)] at each stage,
64/128/256/512 channels, direct global average -> Dense. No metadata/time,
extra FC, dropout, masking or normalization of the input. Returns logits because
PyTorch cross_entropy includes the official final softmax mathematically.

Framework limitations: Keras BN momentum=.99 maps to PyTorch momentum=.01;
PyTorch running variance uses the unbiased batch estimator, unlike some old
Keras backends. Initializers match distributions, not random number streams:
residual kernels use variance-corrected +/-2 sigma truncated He normal, stem
and classifier Glorot uniform. No TensorFlow numerical parity is claimed.

MIT License
Copyright (c) 2019 Sanjit Bhat, David Lu, Albert Kwon, Srinivas Devadas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def keras_he_normal_(weight):
    # std of a unit normal truncated to [-2, 2], as in Keras VarianceScaling.
    fan_in = weight.shape[1] * math.prod(weight.shape[2:])
    std = math.sqrt(2.0 / fan_in) / 0.87962566103423978
    return nn.init.trunc_normal_(weight, std=std, a=-2 * std, b=2 * std)


class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0)))


class SameMaxPool1d(nn.Module):
    """TensorFlow SAME padding: extra element belongs on the right."""
    def __init__(self, kernel_size=3, stride=2):
        super().__init__()
        self.kernel_size, self.stride = kernel_size, stride

    def forward(self, x):
        output_length = (x.shape[-1] + self.stride - 1) // self.stride
        total = max(0, (output_length - 1) * self.stride + self.kernel_size - x.shape[-1])
        return F.max_pool1d(F.pad(x, (total // 2, total - total // 2), value=-float('inf')),
                            self.kernel_size, self.stride)


def bn(channels):
    return nn.BatchNorm1d(channels, eps=1e-5, momentum=0.01)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, stage, block):
        super().__init__()
        stride = 2 if stage > 0 and block == 0 else 1
        d1, d2 = (1, 2) if block == 0 else (4, 8)
        self.conv1 = CausalConv1d(in_channels, channels, 3, stride=stride, dilation=d1, bias=False)
        self.bn1 = bn(channels)
        self.conv2 = CausalConv1d(channels, channels, 3, dilation=d2, bias=False)
        self.bn2 = bn(channels)
        # The official first block projects even at stage 0, with unchanged width.
        self.shortcut = (nn.Sequential(nn.Conv1d(in_channels, channels, 1, stride=stride, bias=False),
                                       bn(channels)) if block == 0 else nn.Identity())
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                keras_he_normal_(module.weight)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x))


class VarCNNNative(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.stem = nn.Sequential(nn.ConstantPad1d(3, 0), nn.Conv1d(1, 64, 7, stride=2, bias=False),
                                  bn(64), nn.ReLU(), SameMaxPool1d())
        self.stages = nn.ModuleList()
        in_channels = 64
        for stage, channels in enumerate((64, 128, 256, 512)):
            self.stages.append(nn.Sequential(ResidualBlock(in_channels, channels, stage, 0),
                                             ResidualBlock(channels, channels, stage, 1)))
            in_channels = channels
        self.classifier = nn.Linear(512, num_classes)
        nn.init.xavier_uniform_(self.stem[1].weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (1, 5000):
            raise ValueError('VarCNNNative requires B x 1 x 5000 direction input')
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.classifier(x.mean(dim=-1))

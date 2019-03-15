#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class CNN(nn.Module):
    def __init__(self, e_char, filters, kernel_size=5):
        super(CNN, self).__init__()

        self.e_char = e_char
        self.kernel_size = kernel_size
        self.filters = filters

        self.conv_layer = nn.Conv1d(e_char, filters, kernel_size, bias=True)

    def forward(self, x_reshaped) -> torch.Tensor:
        x_conv = self.conv_layer(x_reshaped)
        x_conv_out = torch.max(F.relu(x_conv), 2)[0]
        return x_conv_out

### END YOUR CODE


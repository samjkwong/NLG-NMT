#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self).__init__()
        
        self.e_word = e_word

        self.proj = nn.Linear(self.e_word, self.e_word, bias=True)
        self.gate = nn.Linear(self.e_word, self.e_word, bias=True)

    def forward(self, x_conv_out) -> torch.Tensor:
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_proj = F.relu(self.proj(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_proj
        return x_highway

### END YOUR CODE 


from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class NLG(nn.Module):
	""" Natural Language Generation model 
		using a Neural Machine Translation context
		https://arxiv.org/abs/1702.07826
	"""

	def __init__(self):
	""" Need: NMT Model for each character (ex: translate from "person speaking to Michael" to "Michael")
			  Model for determining who speaks after who
	"""
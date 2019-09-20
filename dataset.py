import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchnlp.nn import Attention
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm
import os

class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.x = data[0].float()
        self.y = data[1].long()

    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    
    def __len__(self):
        return len(self.y)
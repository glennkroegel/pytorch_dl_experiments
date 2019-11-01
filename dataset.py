import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.x = data[0].float()/255
        self.y = data[1].long()

    def __getitem__(self, i):
        return (self.x[i], self.y[i])
    
    def __len__(self):
        return len(self.y)

class ProcessedDataLoaderFactory():
    def __init__(self):
        self.train_data = torch.load('processed/training.pt')
        self.cv_data = torch.load('processed/cv.pt')

    def gen(self, batch_size=256):
        train_set = ProcessedDataset(self.train_data)
        cv_set = ProcessedDataset(self.cv_data)
        train_loader = DataLoader(train_set, batch_size=batch_size)
        cv_loader = DataLoader(cv_set, batch_size=batch_size)
        torch.save(train_loader, 'train_loader.pt')
        torch.save(cv_loader, 'cv_loader.pt')
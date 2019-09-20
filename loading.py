'''
created_by: Glenn Kroegel
date: 3 August 2019

description: Create dataloaders to feed for training

'''
import pandas as pd
import numpy as np
from config import TRAIN_DATA, CV_DATA
from dataset import ProcessedDataset

import torch
import os
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader

class DataLoaderFactory():
    '''Standard dataloaders with regular collating/sampling from padded dataset'''
    def __init__(self, train_path=TRAIN_DATA, cv_path=CV_DATA):
        self.train_data = torch.load(TRAIN_DATA)
        self.cv_data = torch.load(CV_DATA)
        self.train_sampler = RandomSampler(self.train_data)
        self.cv_sampler = RandomSampler(self.cv_data)
        self.train_set = ProcessedDataset(self.train_data)
        self.cv_set = ProcessedDataset(self.cv_data)

    def gen_loaders(self, batch_size=1024):
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, collate_fn=self.collate_fn)
        self.cv_loader = DataLoader(self.cv_set, batch_size=batch_size, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        xs = torch.stack([x[0] for x in batch])
        ys = torch.stack([x[1] for x in batch])
        return xs, ys

    def save_loaders(self):
        torch.save(self.train_loader, 'train_loader.pt')
        torch.save(self.cv_loader, 'cv_loader.pt')
'''
created_by: Glenn Kroegel
date: 2 August 2019

'''

import pandas as pd
import numpy as np
from collections import defaultdict, Counter

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torchnlp.nn import Attention
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from tqdm import tqdm

from dataset import ProcessedDataset
from utils import count_parameters, accuracy

from config import NUM_EPOCHS

status_properties = ['loss', 'accuracy']

class Dense(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(Dense, self).__init__()
        self.fc = nn.Linear(in_size, out_size, bias=bias)
        self.bn = nn.BatchNorm1d(out_size)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.in_size = in_size
        self.out_size = out_size

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.bn(self.drop(self.act(self.fc(x))))
        return x

class SamePad(nn.Module):
    def __init__(self, in_dim, stride, ks):
        super(SamePad, self).__init__()
        h = in_dim
        w = in_dim
        self.h = h
        self.w = w
        self.stride = stride
        self.out_h = int(np.ceil(float(h) / float(stride)))
        self.out_w = int(np.ceil(float(w) / float(stride)))
        # self.pad = nn.ZeroPad2d(padding=(self.out_w//2, self.out_w//2, self.out_h//2, self.out_h//2))
        self.p = (ks-1)//2
        self.pad = nn.ZeroPad2d(self.p)

    def forward(self, x):
        x = self.pad(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_c, out_c, in_dim, ks=3, stride=1, bias=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ks, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_c)
        self.drop = nn.Dropout(0.1)
        self.act = nn.LeakyReLU()
        self.pad = SamePad(in_dim, stride, ks)
        self.in_size = in_c
        self.out_size = out_c

    def forward(self, x):
        x = self.pad(x)
        x = self.drop(self.act(self.bn(self.conv(x))))
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, out_dim=14):
        super(ConvBlock, self).__init__()
        self.c1 = Conv(in_c=in_c, out_c=out_c, ks=ks)
        self.c2 = Conv(in_c=out_c, out_c=out_c, ks=ks)
        self.out_dim = out_dim
        self.pool = nn.AdaptiveMaxPool2d(self.out_dim)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.pool(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, ks=3, in_dim=28, out_dim=14):
        super(ResidualBlock, self).__init__()
        self.c1 = Conv(in_c=in_c, out_c=in_c, in_dim=in_dim, ks=ks)
        self.c2 = Conv(in_c=in_c, out_c=in_c, in_dim=in_dim, ks=ks)
        self.c3 = Conv(in_c=in_c, out_c=out_c, in_dim=in_dim, ks=ks)
        self.out_dim = out_dim
        self.pool = nn.AdaptiveMaxPool2d(self.out_dim)

    def forward(self, x):
        x = self.c1(x)
        f = self.c2(x)
        x2 = f + x
        x3 = self.c3(x2)
        x3 = self.pool(x3)
        return x3

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.szs = [28, 14, 7, 7, 5, 2]
        self.conv1 = Conv(in_c = 1, out_c=20, in_dim=self.szs[0], ks=7)
        self.pool1 = nn.AdaptiveMaxPool2d(self.szs[1])
        self.conv_block = ResidualBlock(in_c=20, out_c=40, ks=3, out_dim=self.szs[2])
        self.conv_block2 = ResidualBlock(in_c=40, out_c=80, ks=3, out_dim=self.szs[3])
        self.conv_block3 = ResidualBlock(in_c=80, out_c=80, ks=3, out_dim=self.szs[4])
        self.conv_block4 = ResidualBlock(in_c=80, out_c=80, ks=3, out_dim=self.szs[5])
        self.fc = Dense(160, 10)

    def forward(self, x):
        x = x/255
        bs = x.size(0)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv_block(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class BaseLearner():
    '''Training loop'''
    def __init__(self, epochs=NUM_EPOCHS):
        self.model = Encoder()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-2, weight_decay=1e-6)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-3)
        self.epochs = epochs

        self.train_loader = torch.load('train_loader.pt')
        self.cv_loader = torch.load('cv_loader.pt')

        self.train_loss = []
        self.cv_loss = []

        self.best_loss = 1e3
        print('Model Parameters: ', count_parameters(self.model))

    def train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        props = {k:0 for k in status_properties}
        for _, data in enumerate(train_loader):
            x, targets = data
            logits = model(x)
            loss = criterion(logits, targets)
            props['loss'] += loss.item()
            props['accuracy'] += accuracy(logits, targets).item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(self.model.parameters(), 0.25)
        L = len(train_loader)
        props = {k:v/L for k,v in props.items()}
        return props

    def step(self):
        '''Actual training loop.'''
        for epoch in tqdm(range(self.epochs)):
            train_props = self.train(self.train_loader, self.model, self.criterion, self.optimizer, epoch)
            self.scheduler.step(epoch)
            lr = self.scheduler.get_lr()[0]
            self.train_loss.append(train_props['loss'])
            # cross validation
            props = {k:0 for k in status_properties}
            with torch.no_grad():
                for _, data in enumerate(self.cv_loader):
                    self.model.eval()
                    x, targets = data
                    logits = self.model(x)
                    val_loss = self.criterion(logits, targets)
                    props['loss'] += val_loss.item()
                    props['accuracy'] += accuracy(logits, targets).item()
                L = len(self.cv_loader)
                self.cv_loss.append(props['loss'])
                props = {k:v/L for k,v in props.items()}
                if epoch % 1 == 0:
                    self.status(epoch, train_props, props)
                if props['loss'] < self.best_loss:
                    print('dumping model...')
                    path = 'model' + '.pt'
                    torch.save(self.model, path)
                    self.best_loss = props['loss']

    def status(self, epoch, train_props, cv_props):
        s0 = 'epoch {0}/{1}\n'.format(epoch, self.epochs)
        s1, s2 = '',''
        for k,v in train_props.items():
            s1 = s1 + 'train_'+ k + ': ' + str(v) + ' '
        for k,v in cv_props.items():
            s2 = s2 + 'cv_'+ k + ': ' + str(v) + ' '
        print(s0 + s1 + s2)

if __name__ == "__main__":
    try:
        mdl = BaseLearner()
        mdl.step()
    except KeyboardInterrupt:
        pd.to_pickle(mdl.train_loss, 'train_loss.pkl')
        pd.to_pickle(mdl.cv_loss, 'cv_loss.pkl')
        print('Stopping')
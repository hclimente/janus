#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import trange, tqdm

from datasets import Boyd2019
from losses import ContrastiveLoss
from networks import SiameseNet
from viz import plot_cell, tsne, umap


data_path = 'data/boyd_2019'
metadata_file = 'data/boyd_2019_PlateMap-KPP_MOA.xlsx'

metadata = Boyd2019.read_metadata(metadata_file)

# filter by 2 moas and make train test
metadata = metadata.loc[metadata.moa.isin(['Neutral', 'PKC Inhibitor'])]
train_metadata = metadata.sample(frac = .7)
test_metadata = metadata.drop(train_metadata.index)


boyd2019 = Boyd2019('data/boyd_2019/', train_metadata)

# training params
train_batch_size = 64
train_number_epochs = 100

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
np.random.seed(42)

train_dataloader = DataLoader(boyd2019,
                              shuffle=True,
                              num_workers=8,
                              batch_size=train_batch_size)

# get test set: different wells than training
test_boyd2019 = Boyd2019('data/boyd_2019/', test_metadata)

test_dataloader = DataLoader(test_boyd2019,
                             shuffle=True,
                             num_workers=8,
                             batch_size=64)

net = SiameseNet().to(device)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = 0.0005)
    
with trange(train_number_epochs) as epochs:
    for epoch in epochs:
        test_data = iter(test_dataloader)
        with tqdm(train_dataloader, total = int(len(boyd2019)/train_batch_size)) as tepoch:
            for data in tepoch:
                img0, moa0, img1, moa1, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    
                optimizer.zero_grad()
                output1,output2 = net(img0,img1)
                loss = criterion(output1,output2,label)
                loss.backward()
                optimizer.step()

                img0_test, _, img1_test, _, label_test = next(test_data)
                img0_test, img1_test, label_test = img0_test.to(device), img1_test.to(device), label_test.to(device)
                output1,output2 = net(img0_test,img1_test)
                loss_test = criterion(output1,output2,label_test)
                    
                tepoch.set_postfix(tr_loss=loss.item(), te_loss=loss_test.item())

        torch.save(net.state_dict(), 'siamese_%04d.torch' % epoch)


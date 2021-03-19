#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from datasets import Boyd2019
from losses import ContrastiveLoss
from networks import SiameseNet

# parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', default='../../data/boyd_2019', help="Data folder.")
parser.add_argument('--metadata', default='../../data/boyd_2019_PlateMap-KPP_MOA.xlsx', help="Metadata path.")
parser.add_argument('--batch', default=64, help="Batch size.")
parser.add_argument('--epochs', default=100, help="Number of epochs.")
parser.add_argument('--dropout', default=.05, help="Dropout probability.")
parser.add_argument('--margin', default=2, help="Contrastive loss margin.")
parser.add_argument('--seed', default=42, help="Random seed.")
args = vars(parser.parse_args())

# prepare data
metadata = Boyd2019.read_metadata(args['metadata'])
np.random.seed(args['seed'])

## filter by 2 moas and make train test
metadata = metadata.loc[metadata.moa.isin(['Neutral', 'PKC Inhibitor'])]

## train
tr_metadata = metadata.sample(frac=0.7)
tr_data = Boyd2019(args['data'], tr_metadata)
tr_loader = DataLoader(tr_data,
                       shuffle=True,
                       num_workers=8,
                       batch_size=args['batch'])

## test on different wells
te_metadata = metadata.drop(tr_metadata.index)
te_boyd2019 = Boyd2019(args['data'], te_metadata)
te_loader = DataLoader(te_boyd2019,
                       shuffle=True,
                       num_workers=8,
                       batch_size=64)

# training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

net = SiameseNet(p_dropout=args['dropout']).to(device)
criterion = ContrastiveLoss(margin=args['margin'])
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

te_losses = tr_losses = list()
    
with trange(args['epochs']) as epochs:
    for epoch in epochs:
        test_data = iter(te_loader)
        with tqdm(tr_loader, total=int(len(tr_data)/args['batch'])) as tepoch:
            for data in tepoch:
                img0, _, img1, _, label = data
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    
                optimizer.zero_grad()
                out0, out1 = net(img0, img1)
                tr_loss = criterion(out0, out1, label)
                tr_loss.backward()
                optimizer.step()

                img0_test, _, img1_test, _, label_test = next(test_data)
                img0_test, img1_test, label_test = img0_test.to(device), img1_test.to(device), label_test.to(device)
                out0, out1 = net(img0_test, img1_test)
                te_loss = criterion(out0, out1, label_test)
                    
                tepoch.set_postfix(tr_loss=tr_loss.item(), te_loss=te_loss.item())
                tr_losses.append(tr_loss.item())
                te_losses.append(te_loss.item())

        torch.save(net.state_dict(), 'siamese_%04d.torch' % epoch)

pd.DataFrame({'train_loss': tr_losses,
              'test_loss': te_losses}).\
    to_csv('sn_dropout_%s_margin_%s_seed_%s.tsv' % (args['dropout'], args['margin'], args['seed']), sep='\t')

#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms
from tqdm import trange, tqdm

from janus.datasets import Boyd2019, MultiCellDataset
from janus.losses import ContrastiveLoss
from janus.networks import SiameseNet
from janus.transforms import RandomRot90, RGB


# parser
parser = argparse.ArgumentParser()

parser.add_argument('--data', default='../../data/boyd_2019', help="Data folder.")
parser.add_argument('--metadata', default='../../data/boyd_2019_PlateMap-KPP_MOA.xlsx', help="Metadata path.")
parser.add_argument('--batch', default=64, type=int, help="Batch size.")
parser.add_argument('--epochs', default=101, type=int, help="Number of epochs.")
parser.add_argument('--dropout', default=.05, type=float, help="Dropout probability.")
parser.add_argument('--margin', default=2, type=float, help="Contrastive loss margin.")
parser.add_argument('--seed', default=42, type=int, help="Random seed.")
parser.add_argument('--split', default='crop', type=str, help="Train/test split by crop or by well.")
parser.add_argument('--csize', default=64, type=int, help="Crop size (pixels).")

args = vars(parser.parse_args())

# prepare data
metadata = Boyd2019.read_metadata(args['metadata'])
## filter by 2 moas and make train test
metadata = metadata.loc[metadata.moa.isin(['Neutral', 'EGF Receptor Kinase Inhibitor', 'Cysteine Protease Inhibitor', 'PKC Inhibitor', 'Tyrosine Kinase Inhibitor', 'Protein Tyrosine Phosphatase Inhibitor'])]

np.random.seed(args['seed'])

padding = int(args['csize']/2)
scale = 64/float(args['csize'])
print(scale)

if args['split'] == 'crop':

    tr_data = Boyd2019(args['data'], metadata, padding=padding,
                       scale=scale, train_test=True,
                       transform=transforms.Compose([
                           transforms.RandomHorizontalFlip(),
                           transforms.RandomVerticalFlip(),
                           RandomRot90(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]))

    te_1 = torch.load('test_1.pkl')
    te_2 = torch.load('test_2.pkl')

    te_data = MultiCellDataset(te_1, te_2, metadata,
        transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                RandomRot90(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]))

elif args['split'] == 'well':

    tr_metadata = metadata.sample(frac=0.7, weights=metadata.groupby('moa')['moa'].transform('count'))
    tr_metadata.to_csv('tr_seed_%s.tsv' % args['seed'], sep='\t', index=False)

    tr_data = Boyd2019(args['data'], tr_metadata, padding=padding, scale=scale)

    te_metadata = metadata.drop(tr_metadata.index)
    te_metadata.to_csv('te_seed_%s.tsv' % args['seed'], sep='\t', index=False)

    te_data = Boyd2019(args['data'], te_metadata, padding=padding, scale=scale)

tr_loader = DataLoader(tr_data, shuffle=True, num_workers=8, batch_size=args['batch'])
te_loader = DataLoader(te_data, shuffle=True, num_workers=8, batch_size=args['batch'])

# training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

vgg19 = models.vgg19(pretrained=True)


net = SiameseNet(feature_extractor=vgg19, p_dropout=args['dropout']).to(device)
criterion = ContrastiveLoss(margin=args['margin'])
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

te_losses = list()
tr_losses = list()
    
with trange(args['epochs']) as epochs:
    for epoch in epochs:
        test_data = iter(te_loader)
        with tqdm(tr_loader, total=int(len(tr_data)/args['batch'])) as tepoch:
            for img0, _, img1, _, label in tepoch:
                # train
                net.train()

                img0, img1, label = img0.to(device), img1.to(device), label.to(device)
                    
                optimizer.zero_grad()
                out0, out1 = net(img0, img1)
                tr_loss = criterion(out0, out1, label)
                tr_loss.backward()
                optimizer.step()

                tr_losses.append(tr_loss.item())

                # test
                net.eval()

                img0_test, _, img1_test, _, label_test = next(test_data)
                img0_test, img1_test, label_test = img0_test.to(device), img1_test.to(device), label_test.to(device)
                out0, out1 = net(img0_test, img1_test)
                te_loss = criterion(out0, out1, label_test)
                te_losses.append(te_loss.item())

                tepoch.set_postfix(tr_loss=tr_loss.item(), te_loss=te_loss.item())

        if epoch % 10 == 0:
            torch.save(net.state_dict(), 'sn_dropout_%s_margin_%s_seed_%s_epoch_%03d.torch' %
                       (args['dropout'],args['margin'], args['seed'], epoch))

pd.DataFrame({'train_loss': tr_losses,
              'test_loss': te_losses}).\
    to_csv('sn_dropout_%s_margin_%s_seed_%s.tsv' %
           (args['dropout'], args['margin'], args['seed']), sep='\t', index=False)

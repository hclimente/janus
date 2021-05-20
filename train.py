#!/usr/bin/env python

import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

from janus.datasets import Boyd2019, MultiCellDataset
from janus.losses import ContrastiveLoss
from janus.networks import SiameseNet
from janus.transforms import RandomRot90


class Janus(SiameseNet, pl.LightningModule):
    def __init__(
        self,
        p_dropout=0,
        embedding_dim=256,
        margin=2.0,
        feature_extractor=None,
        idx_cutoff=19,
    ):
        super(Janus, self).__init__(
            p_dropout=p_dropout,
            embedding_dim=embedding_dim,
            feature_extractor=feature_extractor,
            idx_cutoff=idx_cutoff,
        )

        self.criterion = ContrastiveLoss(margin=margin)

    def training_step(self, batch, batch_idx):
        x1, _, x2, _, y = batch
        output1, output2 = self.forward(x1, x2)
        loss = self.criterion(output1, output2, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1, _, x2, _, y = batch
        output1, output2 = self.forward(x1, x2)
        loss = self.criterion(output1, output2, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer


# parser
parser = argparse.ArgumentParser()

parser.add_argument("--data", default="data/boyd_2019", help="Data folder.")
parser.add_argument(
    "--metadata",
    default="data/boyd_2019_PlateMap-KPP_MOA.xlsx",
    help="Metadata path.",
)
parser.add_argument("--batch", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=101, type=int, help="Number of epochs.")
parser.add_argument("--dropout", default=0.05, type=float, help="Dropout probability.")
parser.add_argument("--margin", default=2, type=float, help="Contrastive loss margin.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--split",
    default="crop",
    type=str,
    choices=["well", "crop"],
    help="Train/test split by crop or by well.",
)
parser.add_argument("--size", default=64, type=int, help="Crop size (pixels).")
parser.add_argument(
    "--pretrain", default=False, type=bool, help="Use pre-trained network (vgg19)."
)

if __name__ == "__main__":

    args = vars(parser.parse_args())

    # prepare data
    metadata = Boyd2019.read_metadata(args["metadata"])
    # filter by 2 moas and make train test
    metadata = metadata.loc[metadata.moa.isin(["Neutral", "PKC Inhibitor"])]

    np.random.seed(args["seed"])

    padding = int(args["size"] / 2)
    scale = 64 / float(args["size"])
    transform = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRot90(),
    ]

    if args["pretrain"]:
        transform.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

    transform = transforms.Compose(transform)

    print(scale)

    if args["split"] == "crop":

        tr_data = Boyd2019(
            args["data"],
            metadata,
            padding=padding,
            scale=scale,
            train_test=True,
            transform=transform,
        )

        te_1 = torch.load("test_1.pkl")
        te_2 = torch.load("test_2.pkl")

        te_data = MultiCellDataset(te_1, te_2, metadata, transform=transform)

    elif args["split"] == "well":

        tr_metadata = metadata.sample(
            frac=0.7, weights=metadata.groupby("moa")["moa"].transform("count")
        )
        tr_metadata.to_csv("tr_seed_%s.tsv" % args["seed"], sep="\t", index=False)

        tr_data = Boyd2019(
            args["data"], tr_metadata, padding=padding, scale=scale, transform=transform
        )

        te_metadata = metadata.drop(tr_metadata.index)
        te_metadata.to_csv("te_seed_%s.tsv" % args["seed"], sep="\t", index=False)

        te_data = Boyd2019(
            args["data"], te_metadata, padding=padding, scale=scale, transform=transform
        )

    tr_loader = DataLoader(tr_data, shuffle=True, batch_size=args["batch"])
    te_loader = DataLoader(te_data, batch_size=args["batch"])

    if args["pretrain"]:
        vgg19 = models.vgg19(pretrained=True)
        model = Janus(
            margin=args["margin"], feature_extractor=vgg19, p_dropout=args["dropout"]
        )
    else:
        model = Janus(margin=args["margin"], p_dropout=args["dropout"])

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(model, tr_loader, te_loader)

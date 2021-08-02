#!/usr/bin/env python
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms

from janus.datasets import Boyd2019
from janus.losses import ContrastiveLoss
from janus.networks import SiameseNet
from janus.transforms import RandomRot90
from janus.utils import split_by_crop, split_by_well


class Janus(SiameseNet, pl.LightningModule):
    def __init__(
        self,
        p_dropout=0,
        embedding_dim=256,
        margin=2.0,
        pretrain=False,
        idx_cutoff=19,
        split_by="well",
        lr=0.005,
    ):
        if pretrain:
            vgg19 = models.vgg19(pretrained=True)
            super(Janus, self).__init__(
                p_dropout=p_dropout,
                embedding_dim=embedding_dim,
                feature_extractor=vgg19,
                idx_cutoff=idx_cutoff,
            )
        else:
            super(Janus, self).__init__(
                p_dropout=p_dropout,
                embedding_dim=embedding_dim,
            )

        self.criterion = ContrastiveLoss(margin=margin)
        self.split_by = split_by
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x1, _, x2, _, y = batch
        output1, output2 = self.forward(x1, x2)
        loss = self.criterion(output1, output2, y)
        self.log("train_loss", loss)
        return loss

    def train_dataloader(self):
        return DataLoader(
            self.tr_data,
            shuffle=True,
            batch_size=args.batch,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def validation_step(self, batch, batch_idx):
        x1, moas1, x2, moas2, y = batch
        output1, output2 = self.forward(x1, x2)
        loss = self.criterion(output1, output2, y)
        self.log("val_loss", loss)
        val_dict = {
            "val_loss": loss,
            "emb1": output1,
            "emb2": output2,
            "moas1": moas1,
            "moas2": moas2,
        }
        return val_dict

    def validation_epoch_end(self, validation_step_outputs):
        embedding = torch.empty((0,))
        labels = []

        if self.current_epoch % 5 == 1:
            for val_dict in validation_step_outputs[:4]:
                embedding = torch.cat([embedding, val_dict["emb1"], val_dict["emb2"]])
                labels.extend(val_dict["moas1"] + val_dict["moas2"])

            self.logger.experiment.add_embedding(
                embedding, metadata=labels, global_step=self.current_epoch
            )

        loss = torch.mean(torch.stack([val_dict["val_loss"]
                          for val_dict in validation_step_outputs]))
        return loss

    def val_dataloader(self):
        return DataLoader(
            self.te_data,
            batch_size=args.batch,
            num_workers=os.cpu_count(),
            pin_memory=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def train_test_split(self, data, metadata, crop_size, seed):

        padding = int(crop_size / 2)
        scale = 64 / float(crop_size)
        trfm = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
        ]

        if self.hparams.pretrain:
            vgg_norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            trfm.append(vgg_norm)

        trfm = transforms.Compose(trfm)

        # prepare data
        metadata = Boyd2019.read_metadata(metadata)
        metadata = metadata.loc[metadata.moa.isin(["Neutral", "PKC Inhibitor"])]

        if self.split_by == "crop":
            self.tr_data, self.te_data = split_by_crop(
                data, metadata, seed, padding, scale, trfm
            )
        elif self.split_by == "well":
            self.tr_data, self.te_data = split_by_well(
                data, metadata, seed, padding, scale, trfm
            )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Janus")
        parser.add_argument("--batch", default=64, type=int, help="Batch size.")
        parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
        parser.add_argument(
            "--dropout", default=0.05, type=float, help="Dropout probability."
        )
        parser.add_argument(
            "--margin", default=2, type=float, help="Contrastive loss margin."
        )
        parser.add_argument(
            "--pretrain",
            default=False,
            type=bool,
            help="Use pre-trained network (vgg19).",
        )
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument(
            "--split",
            default="well",
            type=str,
            choices=["well", "crop"],
            help="Train/test split by crop or by well.",
        )
        parser.add_argument("--size", default=64, type=int, help="Crop size (pixels).")
        return parent_parser


parser = argparse.ArgumentParser()

parser.add_argument("--data", default="data/boyd_2019", help="Data folder.")
parser.add_argument(
    "--metadata",
    default="data/boyd_2019_PlateMap-KPP_MOA.xlsx",
    help="Metadata path.",
)
parser = Janus.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

if __name__ == "__main__":

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    logger = TensorBoardLogger("runs", name="janus", default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger
    )

    model = Janus(
        margin=args.margin,
        pretrain=args.pretrain,
        p_dropout=args.dropout,
        split_by=args.split,
    )

    dummy_inputs = torch.randn((1, 3, args.size, args.size))
    logger.experiment.add_graph(model, [dummy_inputs, dummy_inputs])

    model.train_test_split(args.data, args.metadata, args.size, args.seed)
    trainer.fit(model)

#!/usr/bin/env python
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.multiprocessing

# address https://github.com/facebookresearch/maskrcnn-benchmark/issues/103
torch.multiprocessing.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader

from janus.datasets import Boyd2019, CellCognition
from janus.losses import ContrastiveLoss
from janus.networks import FCSN
from janus.utils import split_features_by_well


class Janus(FCSN, pl.LightningModule):
    def __init__(
        self,
        args,
        p_dropout=0,
        embedding_dim=256,
        margin=2.0,
        split_by="well",
        lr=0.005,
    ):
        super(Janus, self).__init__(
            p_dropout=p_dropout,
            embedding_dim=embedding_dim,
        )

        self.args = args
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
            batch_size=self.args.batch,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )

    def validation_step(self, batch, batch_idx):
        x1, moas1, x2, moas2, y = batch
        output1, output2 = self.forward(x1, x2)
        loss = self.criterion(output1, output2, y)
        self.log("val_loss", loss)
        return loss

    def val_dataloader(self):
        return DataLoader(
            self.te_data,
            batch_size=self.args.batch,
            num_workers=os.cpu_count(),
            pin_memory=True,
            persistent_workers=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)
        return optimizer

    def train_test_split(self, data, metadata, seed):

        # prepare data
        metadata = Boyd2019.read_metadata(metadata)
        metadata = metadata.loc[metadata.moa.isin(["Neutral", "PKC Inhibitor"])]

        if self.split_by == "crop":
            pass
        elif self.split_by == "well":
            self.tr_data, self.te_data = split_features_by_well(data, metadata, seed)

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
        parser.add_argument("--seed", default=42, type=int, help="Random seed.")
        parser.add_argument(
            "--split",
            default="well",
            type=str,
            choices=["well", "crop"],
            help="Train/test split by crop or by well.",
        )
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

    logger = TensorBoardLogger("{}/logs".format(args.data), name="fc", default_hp_metric=False)

    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], accelerator="dp", logger=logger
    )

    model = Janus(
        args,
        margin=args.margin,
        p_dropout=args.dropout,
        split_by=args.split,
    )

    dummy_inputs = torch.randn((1, 3, 517))
    logger.experiment.add_graph(model, [dummy_inputs, dummy_inputs])

    model.train_test_split(args.data, args.metadata, args.seed)
    trainer.fit(model)
    trainer.save_checkpoint(
        "fc_{}_split_{}_dropout_{}_margin_{}.ckpt".format(
            args.seed, args.split, args.dropout, args.margin
        )
    )
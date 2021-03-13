import numpy as np
import pandas as pd
import torch

from datasets import Boyd2019
from readers import HDF5Reader


def test_boyd2019():

    d = Boyd2019('data/boyd_2019_PlateMap-KPP_MOA.xlsx')

    assert torch.all(d.mda231 == d.dataset_1)
    assert d.mda231.shape[1:] == (3, 64, 64)
    assert d.avg_mda231.shape == (3, 64, 64)
    assert d.std_mda231.shape == (3, 64, 64)

    assert torch.all(d.mda468 == d.dataset_2)
    assert d.mda468.shape[1:] == (3, 64, 64)
    assert d.avg_mda468.shape == (3, 64, 64)
    assert d.std_mda468.shape == (3, 64, 64)


def test_read_metadata():

    df = Boyd2019.read_metadata('data/boyd_2019_PlateMap-KPP_MOA.xlsx')

    assert df.shape == (384, 9)


def test_get_normalization_params():
    # basic functionality
    row = torch.linspace(0, 4, 5)
    channel = torch.vstack([row, row, row, row, row])
    img = torch.stack([channel, channel, channel])
    imgs = torch.stack([img, img, img, img])

    mean, std = Boyd2019.get_normalization_params(imgs)

    assert type(mean) is torch.Tensor
    assert mean.shape == (3, 5, 5)
    assert torch.all(mean == channel)

    assert type(std) is torch.Tensor
    assert std.shape == (3, 5, 5)
    assert torch.all(std == 0)

    # actual crops
    metadata = pd.DataFrame({'well': ['A01', 'B02']})
    padding = 32

    crops = HDF5Reader.get_crops('test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231', metadata,
                                 padding)
    crops = torch.stack([x for x, _ in crops])

    mean, std = Boyd2019.get_normalization_params(crops)

    assert type(mean) is torch.Tensor
    assert mean.shape == (3, 64, 64)

    assert type(std) is torch.Tensor
    assert std.shape == (3, 64, 64)


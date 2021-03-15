import os
import pandas as pd
import torch

from datasets import Boyd2019
from readers import HDF5Reader


def test_boyd2019():

    assert not os.path.isfile('test/data/mda231_params.pkl')
    assert not os.path.isfile('test/data/mda468_params.pkl')

    metadata = pd.DataFrame({'well': ['A01', 'B02']})
    d = Boyd2019('test/data', metadata)

    assert torch.all(d.mda231 == d.dataset_1)
    assert d.mda231.shape == (891, 3, 64, 64)
    assert d.avg_mda231.shape == (3, 64, 64)
    assert d.std_mda231.shape == (3, 64, 64)
    assert os.path.isfile('test/data/mda231_params.pkl')

    assert torch.all(d.mda468 == d.dataset_2)
    assert d.mda468.shape == (609, 3, 64, 64)
    assert d.avg_mda468.shape == (3, 64, 64)
    assert d.std_mda468.shape == (3, 64, 64)
    assert os.path.isfile('test/data/mda468_params.pkl')

    # test parameter retrieval
    new = Boyd2019('test/data', metadata)
    assert torch.all(new.avg_mda231 == d.avg_mda231)
    assert torch.all(new.std_mda231 == d.std_mda231)
    assert torch.all(new.avg_mda468 == d.avg_mda468)
    assert torch.all(new.std_mda468 == d.std_mda468)

    os.remove('test/data/mda231_params.pkl')
    os.remove('test/data/mda468_params.pkl')


def test_read_metadata():

    df = Boyd2019.read_metadata('data/boyd_2019_PlateMap-KPP_MOA.xlsx')

    assert df.shape == (384, 9)


def test_get_normalization_params():
    # basic functionality
    row = torch.linspace(0, 4, 5)
    imgs = row.repeat(4, 3, 5, 1)

    mean, std = Boyd2019.get_normalization_params(imgs)

    assert type(mean) is torch.Tensor
    assert mean.shape == (3, 5, 5)
    assert torch.all(mean == row.repeat(5,1))

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


def test_load_parameters():

    assert not os.path.isfile('test/params.pkl')

    row = torch.linspace(0, 4, 5)
    crops = row.repeat(4, 3, 5, 1)
    avg, std = Boyd2019.load_parameters('test/params.pkl', crops)

    assert os.path.isfile('test/params.pkl')
    assert type(avg) is torch.Tensor
    assert avg.shape == (3, 5, 5)
    assert torch.all(avg == row.repeat(5,1))

    assert type(std) is torch.Tensor
    assert std.shape == (3, 5, 5)
    assert torch.all(std == 0)

    new_avg, new_std = Boyd2019.load_parameters('test/params.pkl', None)

    assert torch.all(avg == new_avg)
    assert torch.all(std == new_std)

    os.remove('test/params.pkl')

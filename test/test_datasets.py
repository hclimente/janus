import os
import pandas as pd
import torch

from datasets import Boyd2019, MultiCellDataset
from readers import HDF5Reader


def clean_pickles():

    for f in ['test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231/norm_params.pkl',
              'test/data/22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468/norm_params.pkl',
              'test/params.pkl']:
        if os.path.isfile(f):
            os.remove(f)


def test_boyd2019():

    clean_pickles()

    metadata = pd.DataFrame({'well': ['A01', 'B02'],
                             'moa': ['dmso', 'tp53']})
    d = Boyd2019('test/data', metadata)

    assert len(d.dataset_1) == 891
    assert all([x[0].shape == (3, 64, 64) for x in d.dataset_1])
    assert all([type(x[1]) is dict for x in d.dataset_1])
    assert os.path.isfile('test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231/norm_params.pkl')

    assert len(d.dataset_2) == 609
    assert all([x[0].shape == (3, 64, 64) for x in d.dataset_2])
    assert all([type(x[1]) is dict for x in d.dataset_2])
    assert os.path.isfile('test/data/22_384_20X-hNA_D_F_C3_C5_20160032_2016.01.25.16.27.22_MDA468/norm_params.pkl')

    clean_pickles()


def test_read_metadata():

    df = Boyd2019.read_metadata('data/boyd_2019_PlateMap-KPP_MOA.xlsx')

    assert df.shape == (204, 9)


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
    metadata = pd.DataFrame({'well': ['A01', 'B02'],
                             'moa': ['dmso', 'tp53']})
    padding = 32

    crops = HDF5Reader.get_crops('test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231', metadata,
                                 padding)
    crops = torch.stack([x for x, _ in crops])

    mean, std = Boyd2019.get_normalization_params(crops)

    assert type(mean) is torch.Tensor
    assert mean.shape == (3, 64, 64)

    assert type(std) is torch.Tensor
    assert std.shape == (3, 64, 64)

    # layers are correct
    img = torch.stack([torch.tensor([-42]).repeat((5, 5)),
                       torch.tensor([3]).repeat((5, 5)),
                       torch.tensor([57]).repeat((5, 5))])
    imgs = img.repeat((15, 1, 1, 1))

    mean, std = Boyd2019.get_normalization_params(imgs)

    assert type(mean) is torch.Tensor
    assert mean.shape == (3, 5, 5)
    assert torch.all(mean == img)
    assert torch.all((imgs - mean) == 0)
    assert torch.all(std == 0)


def test_load_parameters():

    clean_pickles()

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

    clean_pickles()


def test_multicelldataset():

    p = MultiCellDataset(None, None, None, None)

    assert len(p) == 10000


def test_sample_crops():

    dataset = [('moa1', {'moa': 1}),
               ('moa2', {'moa': 2}),
               ('moa3', {'moa': 3}),
               ('moa4', {'moa': 4}),
               ('moa5', {'moa': 5})]

    for i in range(10):
        _, moa = MultiCellDataset.sample_crops(1, True, dataset)
        assert moa == 1

    for i in range(10):
        _, moa = MultiCellDataset.sample_crops(1, False, dataset)
        assert moa != 1


def test_next():

    # test data
    dataset_1 = [('A1', {'moa': 1}),
                 ('A2', {'moa': 2}),
                 ('A3', {'moa': 3}),
                 ('A4', {'moa': 4}),
                 ('A5', {'moa': 5})]
    dataset_2 = [('B1', {'moa': 1}),
                 ('B2', {'moa': 2}),
                 ('B3', {'moa': 3}),
                 ('B4', {'moa': 4}),
                 ('B5', {'moa': 5})]
    metadata = pd.DataFrame({'moa': [x for x in range(1, 6)]})

    p = MultiCellDataset(dataset_1, dataset_2, metadata, None)

    for i in range(20):
        cell1, moa1, cell2, moa2, same_moa = p[i]
        assert cell1 in [x[0] for x in dataset_1]
        assert moa1 == [x[1]['moa'] for x in dataset_1 if x[0] == cell1][0]
        assert cell2 in [x[0] for x in dataset_2]
        assert moa2 == [x[1]['moa'] for x in dataset_2 if x[0] == cell2][0]
        assert same_moa == (moa1 == moa2)

    # real data
    metadata = pd.DataFrame({'well': ['A01', 'B02'],
                             'moa': ['dmso', 'tp53']})
    d = Boyd2019('test/data', metadata, transform=None)

    for i in range(20):
        cell1, moa1, cell2, moa2, same_moa = d[i]
        assert any([torch.all(x[0] == cell1) for x in d.dataset_1])
        assert moa1 == [x[1]['moa'] for x in d.dataset_1 if torch.all(x[0] == cell1)][0]
        assert any([torch.all(x[0] == cell2) for x in d.dataset_2])
        assert moa2 == [x[1]['moa'] for x in d.dataset_2 if torch.all(x[0] == cell2)][0]
        assert same_moa == (moa1 == moa2)
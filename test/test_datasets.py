import torch

from datasets import Boyd2019


def test_read_metadata():

    df = Boyd2019.read_metadata('../data/boyd_2019_PlateMap-KPP_MOA.xlsx')

    assert df.shape == (384, 9)


def test_get_normalization_params():
    row = torch.linspace(0, 4, 5)
    channel = torch.vstack([row, row, row, row, row])
    img = torch.stack([channel, channel, channel])

    mean, std = Boyd2019.get_normalization_params(img)

    assert mean.shape == (5, 5)
    assert std.shape == (5, 5)
    assert torch.all(mean == channel)
    assert torch.all(std == 0)

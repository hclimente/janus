import torch

from datasets import Boyd2019


def test_read_metadata():

    df = Boyd2019.read_metadata('../multi-cell-line/PlateMap-KPP_MOA.xlsx')

    assert df.shape == (384, 9)

def test_get_normalization_params():
    a = torch.randn(4, 4, 10)

    Boyd2019.get_normalization_params(a)

    Boyd2019('../multi-cell-line/PlateMap-KPP_MOA.xlsx')
    assert a.shape == (4, 4)

import pytest
import torch

from losses import ContrastiveLoss


def test_forward():

    # same target
    same = torch.Tensor([1])
    diff = torch.Tensor([0])

    ones = torch.ones(1,10)

    loss = ContrastiveLoss()

    assert loss(ones, ones, same) == 0
    assert loss(ones, ones, diff) == pytest.approx(2, 0.01)

    assert loss(ones, 2*ones, same) == 5
    assert loss(2*ones, 2*ones, diff) == pytest.approx(2, 0.01)
    assert loss(ones, 2*ones, diff) == 0

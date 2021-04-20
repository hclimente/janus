import numpy as np
import torch

from janus.transforms import RandomRot90


def test_random_rot90():
    img0 = torch.linspace(0, 4, 5).repeat(3, 5, 1)
    img90 = torch.tensor(np.linspace(np.repeat(0, 5), np.repeat(4, 5), 5)).repeat(3, 1, 1)
    img180 = torch.linspace(4, 0, 5).repeat(3, 5, 1)
    img270 = torch.tensor(np.linspace(np.repeat(4, 5), np.repeat(0, 5), 5)).repeat(3, 1, 1)
    imgs = [img0, img90, img180, img270]

    rot90 = RandomRot90()

    for _ in range(10):
        for i in imgs:
            r = rot90(i)
            assert sum([torch.all(r == x) for x in imgs])

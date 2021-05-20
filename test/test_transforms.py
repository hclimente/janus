import numpy as np
import torch

from janus.transforms import Normalize, RandomRot90, RGB


def test_random_rot90():
    img0 = torch.linspace(0, 4, 5).repeat(3, 5, 1)
    img90 = torch.tensor(np.linspace(np.repeat(0, 5), np.repeat(4, 5), 5)).repeat(
        3, 1, 1
    )
    img180 = torch.linspace(4, 0, 5).repeat(3, 5, 1)
    img270 = torch.tensor(np.linspace(np.repeat(4, 5), np.repeat(0, 5), 5)).repeat(
        3, 1, 1
    )
    imgs = [img0, img90, img180, img270]

    rot90 = RandomRot90()

    for _ in range(10):
        for i in imgs:
            r = rot90(i)
            assert sum([torch.all(r == x) for x in imgs])


def test_normalize():

    ch1 = torch.linspace(0, 4, 5).repeat(5, 1)
    ch2 = torch.linspace(40, 10, 5).repeat(5, 1)
    ch3 = torch.linspace(100, 400, 5).repeat(5, 1)
    img = torch.stack((ch1, ch2, ch3))

    norm = Normalize()
    norm_img = norm(img)

    ch1_n = torch.linspace(0, 1, 5).repeat(5, 1)
    ch2_n = torch.linspace(1, 0, 5).repeat(5, 1)
    good_img = torch.stack((ch1_n, ch2_n, ch1_n))

    assert torch.all(good_img == norm_img)


def test_rgb():

    ch1 = torch.linspace(0, 4, 5).repeat(5, 1)
    ch2 = torch.linspace(40, 10, 5).repeat(5, 1)
    ch3 = torch.linspace(100, 400, 5).repeat(5, 1)
    img = torch.stack((ch1, ch2, ch3))
    good_img = torch.stack((ch2, ch3, ch1))
    good_img = good_img.permute(1, 2, 0)

    rgb = RGB()
    img = rgb(img)

    assert torch.all(img == good_img)

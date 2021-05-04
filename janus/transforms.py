from numpy import random
import numpy as np
import torch


class RandomRot90(object):

    def __init__(self):
        pass

    def __call__(self, img):
        return torch.rot90(img, k=random.randint(0, 3), dims=(1, 2))


class Normalize(object):

    def __init__(self):
        pass

    def __call__(self, img):

        max = torch.amax(img, dim=(1, 2)).reshape(3, 1, 1) # torch.quantile(channel, 0.999)
        min = torch.amin(img, dim=(1, 2)).reshape(3, 1, 1)
        # clip image
        img = torch.max(torch.min(img, max), min)
        img = (img - min) / (max - min)

        return img


class RGB(object):

    def __init__(self):
        pass

    def __call__(self, img):
        return torch.stack((img[1, ...], img[2, ...], img[0, ...]), dim=2)

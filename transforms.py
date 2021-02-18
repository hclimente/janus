from numpy import random
import torch


class RandomRot90(object):

    def __init__(self):
        pass

    def __call__(self, img):
        return torch.rot90(img, k=random.randint(0, 3), dims=(1, 2))

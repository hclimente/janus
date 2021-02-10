import numpy as np


class RandomHorizontalFlip(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        return img


class RandomVerticalFlip(object):

    def __init__(self):
        pass

    def __call__(self, img):
        if np.random.rand() > 0.5:
            img = np.flipud(img)

        return img


class RandomRotation(object):

    def __init__(self):
        pass

    def __call__(self, img):
        img = np.rot90(img, k=np.randint(0, 3))

        return img

import numpy as np
import pandas as pd

from janus.datasets import Boyd2019
from janus.networks import SiameseNet
from src.lococv import LocoCV

metadata = pd.DataFrame({"well": ["A01", "B02"], "moa": ["dmso", "tp53"]})
metadata.index = metadata.well
d = Boyd2019("test/data", metadata)
net = SiameseNet()

loco = LocoCV(d, net)
df_profiles = loco.construct_profiles()


def test_LocoCV():

    assert loco.dataset == d
    assert loco.net == net


def test_construct_profiles():

    assert type(df_profiles) == pd.DataFrame
    assert np.all(df_profiles.index.isin(metadata.well))
    assert df_profiles.shape[0] == metadata.shape[0]
    assert df_profiles.shape[1] == net.fc[-1].out_features


def test_lococv():

    confusion = loco.lococv(df_profiles)

    assert type(confusion) == np.ndarray
    assert confusion.shape == (2, 2)

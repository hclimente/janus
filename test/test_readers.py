import pandas as pd
import torch

from janus.readers import HDF5Reader


def test_get_fields():

    metadata = pd.DataFrame({"well": ["A01", "B02"], "moa": ["dmso", "tp53"]})

    fields = HDF5Reader.get_fields(
        "test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231",
        metadata,
    )

    for field, info in fields:
        assert len(info) == 5
        assert info["well"] in metadata.well.to_list()
        assert info["field_no"] in [1, 2, 3, 4]
        assert info["moa"] in ["dmso", "tp53"]

        assert type(field) is torch.Tensor
        assert len(field.shape) == 3
        assert field.shape[0] == 3
        # verify the other dimensions


def test_get_crops():

    metadata = pd.DataFrame({"well": ["A01", "B02"], "moa": ["dmso", "tp53"]})
    padding = 32

    crops = HDF5Reader.get_crops(
        "test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231",
        metadata,
        padding,
    )

    for crop, info in crops:
        assert len(info) == 5
        assert info["well"] in metadata.well.to_list()
        assert info["field_no"] in [1, 2, 3, 4]
        assert info["moa"] in ["dmso", "tp53"]

        assert type(crop) is torch.Tensor
        assert len(crop.shape) == 3
        assert crop.shape[0] == 3
        assert crop.shape[1] == padding * 2


def test_get_features():

    metadata = pd.DataFrame({"well": ["A01", "B02"], "moa": ["dmso", "tp53"]})

    features = HDF5Reader.get_features(
        "test/data/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231",
        metadata,
    )

    for feats, info in features:
        assert len(info) == 5
        assert info["well"] in metadata.well.to_list()
        assert info["field_no"] in [1, 2, 3, 4]
        assert info["moa"] in ["dmso", "tp53"]

        assert type(feats) is torch.Tensor
        assert len(feats) == 517

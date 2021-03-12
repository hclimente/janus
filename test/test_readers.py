import numpy as np
import pandas as pd

from readers import HDF5Reader


def test_get_fields():

    metadata = pd.DataFrame({'well': ['A01','B02']})

    fields = HDF5Reader.get_fields('../multi-cell-line/cecog_out_propagate_0.5/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231', metadata)

    for field, info in fields:
        assert len(info) == 7
        assert info['well'] in metadata.well.to_list()
        assert info['field_no'] in [1, 2, 3, 4]

        assert type(field) is np.ndarray
        assert field.shape[0] == 3
        # verify the other dimensions

def test_get_crops():

    metadata = pd.DataFrame({'well': ['A01','B02']})

    crops = HDF5Reader.get_crops('../multi-cell-line/cecog_out_propagate_0.5/22_384_20X-hNA_D_F_C3_C5_20160031_2016.01.25.17.23.13_MDA231', metadata, 32)

    for crop, info in crops:
        assert len(info) == 7
        assert info['well'] in metadata.well.to_list()
        assert info['field_no'] in [1, 2, 3, 4]

        assert type(field) is np.ndarray
        assert field.shape[0] == 3
        # verify the other dimensions

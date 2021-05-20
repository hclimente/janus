import glob
import os

import h5py
import numpy as np
import torch
from torch.nn import functional as F


class HDF5Reader:
    def __init__(self):
        pass

    @staticmethod
    def get_fields(plate, metadata, channel="primary__primary4"):

        wells = metadata["well"].to_list()

        for well in wells:

            files = glob.glob("%s/hdf5/%s_*.ch5" % (plate, well))

            for file in files:

                with h5py.File(file, "r") as f:
                    file = os.path.splitext(file)[0]
                    file = os.path.basename(file)
                    _, field_no = file.split("_")
                    field_no = int(field_no)

                    base_path = "sample/0/plate/%s/experiment/%s/position/%s" % (
                        os.path.basename(plate),
                        well,
                        field_no,
                    )

                    field = f["%s/image/channel" % base_path][()]
                    info = {
                        "center": f["%s/feature/%s/center" % (base_path, channel)][()],
                        "well": well,
                        "field_no": field_no,
                        "moa": metadata.moa.values[metadata.well == well][0],
                    }

                    yield torch.tensor(np.squeeze(field)), info

    @staticmethod
    def get_crops(plate, metadata, padding, scale=1.0, channel="primary__primary4"):

        for field, info in HDF5Reader.get_fields(plate, metadata, channel):
            _, height, width = field.shape
            centers = info["center"][()]

            for center_x, center_y in centers:
                center_y = np.clip(center_y, padding, height - padding)
                center_x = np.clip(center_x, padding, width - padding)

                left = center_x - padding
                right = center_x + padding
                top = center_y - padding
                bottom = center_y + padding

                crop = field[:, top:bottom, left:right]

                # resize crops
                if not scale == 1.0:
                    c, h, w = crop.shape
                    re_h, re_w = int(scale * h), int(scale * w)
                    crop = F.interpolate(crop[None], size=(re_h, re_w))[0]

                yield crop, info

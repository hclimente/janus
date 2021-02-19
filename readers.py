import glob
import os

import h5py
import numpy as np


class HDF5Reader:

    def __init__(self):
        pass

    @staticmethod
    def get_fields(plate, metadata, channel='primary__primary4', shuffle=False):

        wells = metadata['well'].to_list()
        if shuffle:
            np.random.shuffle(wells)

        for well in wells:

            files = glob.glob("%s/hdf5/%s_*.ch5" % (plate, well))
            if shuffle:
                np.random.shuffle(files)

            for file in files:

                f = h5py.File(file, 'r')
                file = os.path.splitext(file)[0]
                file = os.path.basename(file)
                _, field = file.strsplit('_')

                field = f["sample/0/plate/%s/experiment/%s/position/%s/image/channel" % (plate, well, field)][:]
                info = f["definition/feature/%s" % channel] + metadata[well]

                yield field, info

    @staticmethod
    def get_crops(plate, metadata, padding, channel='primary__primary4', shuffle=False):

        for field, info in HDF5Reader.get_fields(plate, metadata, channel, shuffle):
            height, width, _ = field.shape
            centers = info['center'][()]
            if shuffle:
                np.random.shuffle(centers)
                
            for center_x, center_y in centers:
                center_y = np.clip(center_y, padding, height - padding)
                center_x = np.clip(center_x, padding, width - padding)

                left = center_x - padding
                right = center_x + padding
                top = center_y - padding
                bottom = center_y + padding

                yield field[top:bottom, left:right, :], info

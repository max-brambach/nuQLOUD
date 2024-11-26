import tifffile as tif
import h5py
import numpy as np
import os

"""
Some functions to read images.
"""


def get_number_from_filename(filename, key='TM', delimiter='_'):
    """
    Extract a number from a filename.

    The filename is split into substrings (pieces) based on the delimiter. These
    substrings are checked if they begin with the specified key. If the key is found,
    the rest of the substring will be returned as an integer number.

    Example:
    filename = 'test-file_001_TM0003'
    get_number_from_filename(filename)
    >> 3
    :param filename: str, name of or path to the file.
    :return: int, number extracted from the filename
    """
    filename = os.path.splitext(os.path.basename(filename))[0]
    pieces = filename.split(delimiter)
    for piece in pieces:
        if piece[0:len(key)] == key:
            return int(piece[len(key):])

def universal_image_reader(file):
    """
    Read a 3D image as a numpy array.

    Automatically identify the image file format and use appropriate reader.
    The following formats are supported:
    * .tif/.tiff: using tifffile
    * .ims: imaris file using h5py, highest resolution and first timepoint is loaded
    :param file: str, path to image file, can be relative or absolute.
    :return: np.array, image data, shape: (x, y, (z))
    """
    file_path, file_extension = os.path.splitext(file)
    if file_extension == '.tif' or file_extension == '.tiff':
        img =tif.imread(file)
    elif file_extension == '.ims':
        img = h5py.File(file, 'r')['DataSet']['ResolutionLevel 0']['TimePoint 0']['Channel 0']['Data'][()]
    return np.swapaxes(img, 0, 2)

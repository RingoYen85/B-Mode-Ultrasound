import numpy as np
from read_data import *
import logging
import warnings

data = []


def center_data(data):
    """given RF data in vector form, recenter the data around average

    :param data: 1D array of RF raw data
    :type data: numpy array
    :returns: centered_data: array of recentered rf data (np.array)
    """
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

    avg = np.mean(data)
    logging.debug(avg)
    centered_data = []
    for i in range(0, len(data)):
        newData = (float(data[i]) - avg)
        centered_data.append(newData)

    return centered_data


def rectify_data(centered_data):
    """given RF data in vector form, rectify the data so there's no negative values

    :param centered_data: 1D array of RF raw data
    :type centered_data: numpy array
    :returns: rectify_data_set: array of rectified rf data (np.array)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    rectify_data_set = [abs(x) for x in centered_data]
    logging.debug(len(rectify_data_set))
    return rectify_data_set


def low_pass_filter(rectify_data_set, window_size=15):
    """given RF data in vector form, convolve the data with kernel
    of pre-defined window size to achieve low pass filter effect

    :param rectify_data_set: 1D array of RF raw data
    :param window_size: user input window size, default 15
    :type rectify_data_set: numpy array
    :type window_size: int
    :returns: filtered_data: array of filtered rf data (np.array)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    kernel = np.ones(window_size)
    filtered_data = np.convolve(rectify_data_set, kernel, 'same')
    filtered_data = np.divide(filtered_data, len(kernel))

    return filtered_data


def log_compression(filtered_data):
    """given RF data in vector form, compress the data with log10

    :param filtered_data: 1D array of RF raw data
    :type filtered_data: numpy array
    :returns: data_compress: array of compressed rf data (np.array)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    warnings.filterwarnings('error')
    try:
        data_compress = np.log10(filtered_data)
    except Warning:
        logging.error(" Runtimewarning: divide by zero in log10")
        sys.exit()
    logging.info(data_compress)
    return data_compress


def reshape_process(data_compress, axial_samples, num_beams):
    """given RF data in vector form, return reshaped data in 2D matrix form

    :param data_compress: 1D array of RF raw data
    :param axial_samples: Number of axial samples - given by JSON file
    :param num_beams: Number of ultrasound beams - given by JSON file
    :type data_compress: numpy array
    :type axial_samples: int
    :type num_beams: int
    :returns: reshape_data: matrix of reorginized RF data
    """
    reshape_data = np.reshape(data_compress,
                              (axial_samples, num_beams), order='F')
    return reshape_data


if __name__ == '__main__':

    fs, c, axial_samples, num_beams, beam_spacing = readJSON("bmode.json")
    data_out = readBinary("rfdat.bin")

    centered_data = center_data(data_out)
    rectified_data = rectify_data(centered_data)

    filtered_data = low_pass_filter(rectified_data, 15)
    data_compress = log_compression(filtered_data)

    reshape_data = reshape_process(data_compress, axial_samples, num_beams)

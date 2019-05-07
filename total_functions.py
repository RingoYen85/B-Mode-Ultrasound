import numpy as np
import matplotlib.pyplot as plt
from argparse_func import *
import logging
import warnings
import sys


data = []


def readBinary(filename):
    """given binary filename, read the RF data

    :param filename: text string representing the name of RF file
    :type filename: string
    :returns: rf_data: array of rf data (np.array)
    """

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    import numpy as np
    try:
        rf_data = np.fromfile(filename, dtype='int16', count=-1, sep='')
    except FileNotFoundError:
        logging.error(" FileNotFoundError: no such file or directory")
        sys.exit()
    return rf_data


def readJSON(filename):
    """given JSON filename, read the metadata and return all metadata parameters

    :param filename: text string representing the name of JSON file
    :type filename: string
    :returns: fs, c, axial_samples, num_beams, beam_spacing
    """
    import json

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    try:
        with open(filename) as data_file:
            data = json.load(data_file)
    except FileNotFoundError:
        logging.error(" FileNotFoundError: no such file or directory")
        sys.exit()
    logging.debug(data)
    c = data['c']
    num_beams = data['num_beams']
    fs = data['fs']
    beam_spacing = data['beam_spacing']
    axial_samples = data['axial_samples']
    logging.debug("c " + str(c) + " num_beams " + str(num_beams))
    logging.debug("fs " + str(fs))
    logging.debug(" spacing " + str(beam_spacing))
    logging.debug("axial_samples " + str(axial_samples))

    return fs, c, axial_samples, num_beams, beam_spacing


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
    logging.basicConfig(filename='example.log',
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(center_data)

    return centered_data


def rectify_data(centered_data):
    """given RF data in vector form, recenter the data around average

    :param data: 1D array of RF raw data
    :type data: numpy array
    :returns: centered_data: array of recentered rf data (np.array)
    """
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

    rectify_data_set = [abs(x) for x in centered_data]
    logging.debug(len(rectify_data_set))
    logging.info(rectify_data_set)
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
    logging.info(filtered_data)

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

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    reshape_data = np.reshape(data_compress, (axial_samples, num_beams),
                              order='F')
    logging.info(reshape_data)
    return reshape_data


def Display(data, num_beams, beam_spacing, axial_samples, fs, c):
    """given processed RF data and JSON metadata, display 2D matrix using imshow

    :param data: 2D matrix of processed RF data
    :param num_beams: Number of beams of ultrasound beams
    :param beam_spacing: Spacing between ultrasound beams
    :param axial_samples: Number of axial samples taken during ultrasound
    :param fs: Sampling Frequency in Hertz
    :param c: Sound Speed
    :type data: numpy array
    :type num_beams: int
    :type beam_spacing: float
    :type axial_samples: int
    :type fs: int
    :type c: int
    :returns: null
    """

    plt.imshow(data, cmap='Greys_r',
               extent=[0, num_beams * beam_spacing,
                       axial_samples / fs * (c / 2), 0])
    plt.xlabel('Lateral Distance (meters)')
    plt.ylabel('Axial Distance (meters)')
    plt.title('B-Mode Ultrasound Image')
    plt.show()


def Save(data, num_beams, beam_spacing, axial_samples, fs, c):
    """given processed RF data and JSON metadata, save PNG img file generated

    :param data: 2D matrix of processed RF data
    :param num_beams: Number of beams of ultrasound beams
    :param beam_spacing: Spacing between ultrasound beams
    :param axial_samples: Number of axial samples taken during ultrasound
    :param fs: Sampling Frequency in Hertz
    :param c: Sound Speed
    :type data: numpy array
    :type num_beams: int
    :type beam_spacing: float
    :type axial_samples: int
    :type fs: int
    :type c: int
    :returns: null
    """

    logging.debug(data)
    plt.imshow(data, cmap='Greys_r',
               extent=[0, num_beams * beam_spacing,
                       axial_samples / fs * (c / 2), 0])
    plt.savefig('bmode.png')


if __name__ == '__main__':

    fs, c, axial_samples, num_beams, beam_spacing = readJSON("bmode.json")
    data_out = readBinary("rfdat.bin")

    centered_data = center_data(data_out)
    rectified_data = rectify_data(centered_data)

    filtered_data = low_pass_filter(rectified_data, 15)
    data_compress = log_compression(filtered_data)

    reshape_data = reshape_process(data_compress, axial_samples, num_beams)

    logging.info(reshape_data)

    Display(reshape_data, num_beams, beam_spacing, axial_samples, fs, c)
    Save(reshape_data, num_beams, beam_spacing, axial_samples, fs, c)

import logging
import sys


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

    logging.basicConfig(filename='log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')
    import json
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

    return fs, c, axial_samples, num_beams, beam_spacing


if __name__ == '__main__':

    data = readBinary("rfdat.bin")
    readJSON("bmode.json")

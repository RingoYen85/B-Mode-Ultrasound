import matplotlib.pyplot as plt
from process_data import *
from read_data import *


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

    Display(reshape_data, num_beams, beam_spacing, axial_samples, fs, c)
    Save(reshape_data, num_beams, beam_spacing, axial_samples, fs, c)

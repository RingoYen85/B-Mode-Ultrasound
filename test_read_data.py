from read_data import readBinary
from read_data import readJSON

import numpy as np


class TestClass:
    def test_reading_data(self):
        """ This function tests the reader function. A .bin file is run through and tested to see if the reader can properly
            detect it and read it using the proper reader method in the function.

        """
        values_actual = np.array([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Define output, with the input.bin file

        values = readBinary("unit_test.bin")

        for i in range(0,len(values_actual)):
            assert values[i] == values_actual[i]

    def test_reading_json(self):
        """ This function tests if the json reader can properly read the file
        """

        # actual values

        fs_actual = 40000000
        c_actual = 1540
        axial_samples_actual = 1556
        num_beams_actual = 256
        beam_spacing_actual = 0.00011746274509803921

        fs, c, axial_samples, num_beams, beam_spacing = readJSON("bmode.json")

        assert fs_actual == fs
        assert c_actual == c
        assert axial_samples_actual == axial_samples
        assert num_beams_actual == num_beams
        assert beam_spacing_actual == beam_spacing













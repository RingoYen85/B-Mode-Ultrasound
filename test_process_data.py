import numpy as np
from process_data import *
import pytest


class TestClass:

    def test_center_data(self):

        input_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        input_2 = [1, 1.5, -1.5, 0, 1, 1, 1]
        input_3 = [-2, -1, -2, -3, -4, -10]
        input_4 = [-5, -5, -5, -5, -5]

        center_actual_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        center_actual_2 = [1 - (4/7), 1.5 - (4/7), -1.5 - (4/7), 0 - (4/7), 1 - (4/7), 1-(4/7), 1-(4/7)]
        center_actual_3 = [-2 + (22/6), -1 + (22/6.0), -2 + (22/6), -3 + (22/6), -4 + (22/6), -10+ (22/6)]
        center_actual_4 =[0, 0, 0, 0, 0]

        center_1 = center_data(input_1)
        center_2 = center_data(input_2)
        center_3 = center_data(input_3)
        center_4 = center_data(input_4)

        assert center_actual_1 == center_1
        assert center_actual_2 == center_2
        assert center_actual_3 == center_3
        assert center_actual_4 == center_4


    def test_rectify_data(self):

        input_1 = [0,0,0,0,0]
        input_2 = [1,1,1,1,1]
        input_3 = [-1, -1, -1, -1, -1]
        input_4 = [-10, 0, -10, 1.1, -1.4]

        rectify_actual_1 = [0,0,0,0,0]
        rectify_actual_2 = [1,1,1,1,1]
        rectify_actual_3 = [1, 1, 1, 1, 1]
        rectify_actual_4 = [10, 0, 10, 1.1, 1.4]

        rectify_1 = rectify_data(input_1)
        rectify_2 = rectify_data(input_2)
        rectify_3 = rectify_data(input_3)
        rectify_4 = rectify_data(input_4)

        assert rectify_actual_1 == rectify_data(input_1)
        assert rectify_actual_2 == rectify_data(input_2)
        assert rectify_actual_3 == rectify_data(input_3)
        assert rectify_actual_4 == rectify_data(input_4)

    def test_low_pass_filter(self):

        input_1 = np.linspace(1,1,20)
        input_2 = np.linspace(0,0,20)

        a = [8, 9, 10, 11, 12, 13, 14, 15, 15, 15, 15, 15, 15, 14, 13, 12, 11, 10, 9, 8.]
        low_pass_actual_1 = [x/15 for x in a]

        b = np.linspace(0,0,20)
        low_pass_actual_2 = np.divide(b,15)


        low_pass_1 = low_pass_filter(input_1,15)
        low_pass_2 = low_pass_filter(input_2, 15)

        for i in range(0,len(low_pass_actual_1)):
            assert low_pass_1[i] == low_pass_actual_1[i]

        for k in range(0,len(low_pass_actual_2)):
            assert low_pass_2[k] == low_pass_actual_2[k]

        assert len(low_pass_1) == len(low_pass_actual_1)
        assert len(low_pass_2) == len(low_pass_actual_2)

    def test_log_compression(self):

        input_1 = [100, 100, 100]
        input_2 = [1000, 1000, 1000]

        log_compress_actual_1 = [2.0, 2.0, 2]
        log_compress_actual_2 = [3, 3, 3]

        log_compress_1 = log_compression(input_1)
        log_compress_2 = log_compression(input_2)

        np.testing.assert_equal(log_compress_1,log_compress_actual_1)
        np.testing.assert_equal(log_compress_2, log_compress_actual_2)

    def test_reshape_data(self):

        # first set of inputs
        values_1 = np.linspace(1, 1, 9)
        axial_samples_1 = 3
        num_beams_1 = 3

        # second set of inputs
        values_2 = np.linspace(2, 2, 12)
        axial_samples_2 = 4
        num_beams_2 = 3

        # third set of inputs
        values_3 = np.linspace(5, 30, 6)
        axial_samples_3 = 3
        num_beams_3 = 2

        # actual output
        reshape_data_actual_1 = np.matrix('1 1 1; 1 1 1; 1 1 1')
        reshape_data_actual_2 = np.matrix('2 2 2; 2 2 2; 2 2 2; 2 2 2')
        reshape_data_actual_3 = np.matrix('5 20; 10 25 ; 15 30')

        # Define functions
        reshape_process_1 = reshape_process(values_1,axial_samples_1, num_beams_1)
        reshape_process_2 = reshape_process(values_2,axial_samples_2, num_beams_2)
        reshape_process_3 = reshape_process(values_3, axial_samples_3, num_beams_3)

        np.testing.assert_equal(reshape_process_1, reshape_data_actual_1)
        np.testing.assert_equal(reshape_process_2, reshape_data_actual_2)
        np.testing.assert_equal(reshape_process_3, reshape_data_actual_3)

































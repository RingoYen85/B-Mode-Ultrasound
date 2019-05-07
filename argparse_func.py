from output_data import *
from process_data import *
from read_data import *
import argparse

metadata = ()


def main():
    """Main function to use user inputs and run all functions

    """
    args = parser_cli()

    RFbinaryFilename = args.RFbinaryFilename
    JSONFilename = args.JSONFilename
    display = args.display
    save = args.save

    data = readBinary(RFbinaryFilename)
    fs, c, axial_samples, num_beams, beam_spacing = readJSON(JSONFilename)

    centered_data = center_data(data)
    rectified_data = rectify_data(centered_data)

    filtered_data = low_pass_filter(rectified_data, 15)
    data_compress = log_compression(filtered_data)

    processed_data = reshape_process(data_compress, axial_samples, num_beams)

    if display:
        Display(processed_data, num_beams, beam_spacing, axial_samples, fs, c)
    if save:
        Save(processed_data, num_beams, beam_spacing, axial_samples, fs, c)


def parser_cli():
    """Argparser to take user input arguments

    :param argument 0: RF binary filename
    :param argument 1: JSON binary filename
    :param argument 2: display boolean option
    :param argument 3: save boolean option
    :returns: RFbinaryFilename(string), JSONFilename(string)
    """
    parser = argparse.ArgumentParser(description='B-mode Ultrasound Imaging.')
    parser.add_argument('--RFbinaryFilename',
                        dest="RFbinaryFilename",
                        default='rfdat.bin')

    parser.add_argument('--JSONFilename',
                        dest="JSONFilename",
                        default='bmode.json')

    parser.add_argument('--display', default=False,
                        type=bool, dest="display",
                        help='Boolean input argument to render B-mode image')

    parser.add_argument('--save', default=True,
                        type=bool, dest="save",
                        help='Boolean input argument to save PNG B-mode image')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()

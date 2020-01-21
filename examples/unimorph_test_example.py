"""
Example script
"""

import argparse
import os

#from unimorph_inflect import download
from unimorph_inflect.utils.resources import download, DEFAULT_MODEL_DIR
from unimorph_inflect import inflect
from unimorph_inflect.src.myutil import simple_read_data


if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument('-d', '--models-dir', help='location of models files | default: ~/unimorph_inflect_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--language', help='language of text | default: en', default='eng')
    parser.add_argument('-t', '--testfile', help='file to test on (in Unimorph format) | default: <empty>', default='')
    parser.add_argument('-o', '--output', help='output file path', default=None)
    # misc arguments
    parser.add_argument('--force-download', help='force download of models', action='store_true')

    args = parser.parse_args()
    # set output file path
    if args.output is None:
        output_file_path = 'text_file.out'
    else:
        output_file_path = args.output

    language = args.language

    try:
        inputs, outputs, tags = simple_read_data(args.testfile)
        curr_out = inflect(inputs, tags, language=language)
        correct = [o==c for o,c in zip(outputs,curr_out)]
        print(f"Accuracy: {float(sum(correct))/len(correct)}")
    except:
        print("dangit")


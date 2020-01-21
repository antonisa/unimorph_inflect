"""
Example script
"""

import argparse
import os

#from unimorph_inflect import download
from unimorph_inflect.utils.resources import download, DEFAULT_MODEL_DIR
from unimorph_inflect import inflect

if __name__ == '__main__':
    # get arguments
    parser = argparse.ArgumentParser()
    # main arguments
    parser.add_argument('-d', '--models-dir', help='location of models files | default: ~/unimorph_inflect_resources',
                        default=DEFAULT_MODEL_DIR)
    parser.add_argument('-l', '--language', help='language of text | default: en', default='eng')
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
    print(inflect("love", "V;3;SG", language=language))
    print(inflect("love", "V;NFIN", language=language))
    print(inflect("love", "V;V.PTCP;PRS", language=language))
    print(inflect("drink", "V;3;SG", language=language))
    print(inflect("drink", "V;NFIN", language=language))
    print(inflect("drink", "V;V.PTCP;PRS", language=language))
    print(inflect("drink", "N;3;SG", language=language))
    print(inflect("αντίο", "V;3;SG", language=language))

    language = 'deu'
    print(inflect("lieben", "V.PTCP;PST", language=language))
    print(inflect("sortieren", "V.PTCP;PST", language=language))
    
    language = 'ell'
    print(inflect("Βέλγιο", "N;NOM;PL", language=language))
    print(inflect("Βέλγιο", "N;NEUT;GEN;SG", language=language))
    print(inflect("βέλγικη", "ADJ;FEM;GEN;SG", language=language))
    print(inflect("ανταγωνιστικότητα", "N;ACC;PL", language=language))
    print(inflect(["βλέπω", "ακούω"], ["V;3;SG;IPFV;PRS","V;PFV;PST;3;PL"], language=language))
    print(inflect("ψυχή", "N;FEM;ACC;PL", language=language))
    print(inflect("ψυχή", "N;ACC;PL", language=language))


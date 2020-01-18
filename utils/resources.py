"""
utilities for getting resources
"""

import os
import requests
import zipfile

from tqdm import tqdm
from pathlib import Path

# set home dir for default
HOME_DIR = str(Path.home())
DEFAULT_MODEL_DIR = os.path.join(HOME_DIR, 'unimorph_inflect_resources')
DEFAULT_MODELS_URL = 'http://www.cs.cmu.edu/~aanastas/software/inflection_models'
DEFAULT_DOWNLOAD_VERSION = 'latest'

# list of supported language names
languages = ['ell', 'ell2', 'eng']
supported_tagset = {}
supported_tagset['eng'] = ["V;3;SG;PRS", "V;NFIN", "V;PST", "V;V.PTCP;PRS", "V;V.PTCP;PST"]


# download a dynet model zip file
def download_dynet_model(lang_name, resource_dir=None, should_unzip=True, confirm_if_exists=False, force=False,
                      version=DEFAULT_DOWNLOAD_VERSION):
    # ask if user wants to download
    if resource_dir is not None and os.path.exists(os.path.join(resource_dir, f"{lang_name}")):
        if confirm_if_exists:
            print("")
            print(f"The model directory already exists at \"{resource_dir}/{lang_name}\". Do you want to download the models again? [y/N]")
            should_download = 'y' if force else input()
            should_download = should_download.strip().lower() in ['yes', 'y']
        else:
            should_download = False
    else:
        print('Would you like to download the models for: '+lang_name+' now? (Y/n)')
        should_download = 'y' if force else input()
        should_download = should_download.strip().lower() in ['yes', 'y', '']
    if should_download:
        # set up data directory
        if resource_dir is None:
            print('')
            print('Default download directory: ' + DEFAULT_MODEL_DIR)
            print('Hit enter to continue or type an alternate directory.')
            where_to_download = '' if force else input()
            if where_to_download != '':
                download_dir = where_to_download
            else:
                download_dir = DEFAULT_MODEL_DIR
        else:
            download_dir = resource_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        print('')
        print('Downloading models for: '+lang_name)
        model_zip_file_name = f'{lang_name}.zip'
        download_url = f'{DEFAULT_MODELS_URL}/{version}/{model_zip_file_name}'
        download_file_path = os.path.join(download_dir, model_zip_file_name)
        print('Download location: '+download_file_path)

        # initiate download
        r = requests.get(download_url, stream=True)
        with open(download_file_path, 'wb') as f:
            file_size = int(r.headers.get('content-length'))
            default_chunk_size = 67108864
            with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=default_chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
        # unzip models file
        print('')
        print('Download complete.  Models saved to: '+download_file_path)
        if should_unzip:
            unzip_dynet_model(lang_name, download_file_path, download_dir)
        # remove the zipe file
        print("Cleaning up...", end="")
        os.remove(download_file_path)
        print('Done.')


# unzip a dynet models zip file
def unzip_dynet_model(lang_name, zip_file_src, zip_file_target):
    print('Extracting models file for: '+lang_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def download(download_language, resource_dir=None, confirm_if_exists=False, force=False, version=DEFAULT_DOWNLOAD_VERSION):
    if download_language in languages:
        download_dynet_model(download_language, resource_dir=resource_dir, confirm_if_exists=confirm_if_exists, force=force,
                          version=version)
    else:
        raise ValueError(f'The language "{download_language}" is not currently supported by this function. Please try again with other languages.')


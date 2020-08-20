#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 10:16:17 2020

the main to run with arguments

@author: Farruh Kushnazarov
"""


import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

# from src.data.make_dataset import CombineAndCreateDF, CreateAccentData
# from src.features.mfcc_train_test_data import WavToMFCC
# from src.models.train_model import Modeling

sys.path.append('../../src/')
from data.make_dataset import CombineAndCreateDF, CreateAccentData
from features.mfcc_train_test_data import WavToMFCC
from models.train_model import Modeling


def main(project_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dir_external = project_dir / "data/external"
    dir_interim = project_dir / "data/interim"
    logger = logging.getLogger(__name__)

    logger.info('Download and convert media files.')
    path_df_accent = dir_interim / 'df_accent_gmu.pkl'
    if not os.path.exists(path_df_accent):
        root_url = 'http://accent.gmu.edu/'
        browse_language_url = 'browse_language.php?function=find&language={}'
        create_accent_gmu = CreateAccentData(dir_external, dir_interim, root_url, browse_language_url)
        media_url = 'http://chnm.gmu.edu/accent/soundtracks/{}.mp3'
        accent_source = "audio/accent_gmu_edu"
        create_accent_gmu.GetMedia(accent_source, media_url)
    path_df_accent = dir_interim / 'df_accent.pkl'
    if not os.path.exists(path_df_accent):
        accent_df = CombineAndCreateDF(dir_external, dir_interim)
        dir_in_media = 'audio/DS_10283_3443/VCTK-Corpus-0.92/wav48_silence_trimmed'
        dir_out_wav = 'audio/selected_VCTK-Corpus-0.92'
        accent = 'native'
        native_selected = ['p248', 'p251', 'p376']
        name_df_accent = 'df_accent_VCTK_1.pkl'
        accent_df.CreateVCTK_DF(name_df_accent, dir_in_media, dir_out_wav, native_selected, accent)
        accent = 'non_native'
        non_native_selected = ['p294', 'p334', 'p339']
        name_df_accent = 'df_accent_VCTK_2.pkl'
        accent_df.CreateVCTK_DF(name_df_accent, dir_in_media, dir_out_wav, non_native_selected, accent)

        dir_in_media = Path('/mnt/projects/speech_recognition/DeepSpeech_Mozilla/datasets/audio_en_mozilla/en/clips')
        dir_out_wav = 'audio/selected_mozilla'
        name_df_accent = 'df_accent.pkl'
        accent_df.CreateMozillaDF(name_df_accent, dir_in_media, dir_out_wav)

    logger.info('Change wav files to MFCC.')
    path_to_result = project_dir / "data/processed"
    if not (path_to_result / 'data.pkl').exists() or not (path_to_result / 'labels.pkl').exists():
        path_to_df = dir_interim / 'df_accent.pkl'
        to_mfcc = WavToMFCC(path_to_df, path_to_result)
        to_mfcc.ToMFCC(1, 19)

    logger.info('Modeling and Print results.')

    # Network can be:
    # -> FFNN - Feed Forward Neural Network
    # -> CNN - Convolutional Neural Network
    # -> LSTM - Long Short-Term Memory
    networks = ['FFNN', 'CNN', 'LSTM']
    epochs = 2
    num_classes = 2
    batch_size = 128
    need_to_train = False

    dir_data = project_dir / "data/processed"
    dir_results = project_dir / "reports"
    models = Modeling(project_dir, dir_data, dir_results, num_classes, epochs, batch_size)
    for network in networks:
        models.MultiModel(network, need_to_train)
    models.PrintResult()
    models.SaveResults()

    logger.info('Finished.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[1]
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(project_dir)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 10:16:17 2020

Defined three classes:
1. CreateAccentDF - since we have three source of the speech audio files, need to collect audio from those sources
   and create a dataframe with information of audio.
2. DownloadMedia - download the audio file from three sources
3. Preprocessing - different sources has different audio formats, like flac, mp3 and other,
   for the MFCC need to transfer those formats to wav, author recommend todo it with ffmpeg

@author: Farruh Kushnazarov
"""

import pickle
import librosa
import numpy as np
import sys

# from src import functions
sys.path.append('../../src/')
import functions

# Class to transfer audio file to MFCC (Mel-Frequency Cepstral Coefficients)
class WavToMFCC:
    X = []
    y = []

    def __init__(self, path_to_df, path_to_obj):
        self.path_to_df = path_to_df
        self.path_to_obj = path_to_obj

    def LoadDF(self):
        # Load metadata
        with open(self.path_to_df, 'rb') as input_:
            self.df = pickle.load(input_)
        class_name = ['non_native', 'native']
        for i in range(len(class_name)):
            self.df.loc[self.df['accent'] == class_name[i], 'accent'] = i
        self.df['accent'] = self.df['accent'].astype(int)

    def ToMFCC(self, start, end):
        self.LoadDF()
        n_files = self.df.shape[0]
        ind_files = 0
        for accent, media_path, file_name in zip(self.df['accent'], self.df['media_path'], self.df['file_name']):
            check_bit = 0
            try:
                a, s = librosa.load(media_path)
            except:
                check_bit = 1
                print('File {} can\'t open'.format(media_path))
            if check_bit == 0:
                mfcc = librosa.feature.mfcc(y=a, sr=s)
                temp = mfcc.T[1][start:end]
                for frame in range(10, 50):
                    temp = np.concatenate((temp, mfcc.T[frame][start:end]))
                self.X.append(temp)
                self.y.append(accent)
                functions.CoolPrint(ind_files, n_files)
                ind_files = ind_files + 1
            else:
                print(f'Can not open the audio file {media_path}.')
                df = df.drop(df['media_path'] == media_path)
                continue
        self.SaveObj()

    def SaveObj(self):
        functions.saveObject(self.X, self.path_to_obj / 'data.pkl')
        functions.saveObject(self.y, self.path_to_obj / 'labels.pkl')

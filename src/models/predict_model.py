#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:16:17 2020

@author: farruh
"""

import contextlib
import librosa
import numpy as np
import os
import pandas as pd
import pickle
import requests
import wave

from collections import Counter
from math import floor
from tensorflow.keras.models import load_model
from time import localtime, strftime


def CoolPrint(current, total, text=""):
    if text == "":
        if current < total:
            percent = int(floor(current / total * 100))
            print('{:<3}/{:3}: [{}>{}]'.format(current, total, '=' * percent, '.' * (100 - percent)), end='\r',
                  flush=True)
        else:
            print('{:3}/{:3}: [{}]'.format(current, total, '=' * 100))
    else:
        if current < total:
            percent = int(floor(current / total * 100))
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '{:<4}/{:4}. {}'.format(current, total, percent), text)
        else:
            print('{:4}/{:4}: [{}]'.format(current, total, '=' * 100))


def GetVideoLink(links, accents):
    path_video = project_path + 'input/video/test_v02/'
    if not os.path.exists(path_video):
        os.mkdir(path_video)

    n_links = len(links)
    i = 0
    for link, accent in zip(links, accents):
        i = i + 1
        # Checking is session_sn is not available value
        if link is None or accent is None:
            CoolPrint(i, n_links, f'-- The value of link is {link} and accent is {accent}. Go to the next iteration.')
            continue
        path_consultant_video = path_video + str(i) + '_' + accent + '.mp4'
        if os.path.exists(path_consultant_video):
            CoolPrint(i, n_links, f'-- The video for {accent} with number {i} exists.')
            continue
        with open(path_consultant_video, 'wb') as file:
            # get request
            CoolPrint(i, n_links, f'-- Start to download the video for accent: {accent}')
            response = requests.get(link)
            # write to file
            file.write(response.content)
            CoolPrint(i, n_links, f'-- Finished download the video for accent: {accent}')


def Preprocessing(in_folder, out_folder):
    if not os.path.exists(in_folder):
        print('The folder {} does not exist.'.format(in_folder))
        return -1
    if not os.path.exists(out_folder + 'all/'):
        os.mkdir(out_folder + 'all/')
    n_files = len(next(os.walk(in_folder))[2])
    i = 0
    for file in os.scandir(in_folder):
        i = i + 1
        # Change grab audio from video
        f_in = in_folder + file.name
        out_format = '.mp4'
        in_format = '.wav'
        length = len(out_format)
        f_out = out_folder + 'all/' + file.name[:-length] + in_format
        if not file.name.endswith(out_format):
            CoolPrint(i, n_files, f'The file {file.name} not {out_format} format.')
            continue
        if os.path.exists(f_out):
            CoolPrint(i, n_files, f'Audio file {file.name[:-length]}{out_format} exist.')
            continue

        command = f'ffmpeg -i {f_in} -ar 48000 -ac 1 -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold=-40dB ' \
                  f'{f_out} &> /dev/null'
        os.system(command)
        f_in = out_folder + 'all/' + file.name[:-length] + in_format
        os.mkdir(out_folder + file.name[:-length])
        f_out = out_folder + file.name[:-length] + '/' + file.name[:-length] + '_%03d' + in_format
        duration = 60
        command = f'ffmpeg -i {f_in} -f segment -segment_time {duration} -c copy {f_out} &> /dev/null'
        os.system(command)

        CoolPrint(i, n_files, f'-- From video file {file.name} from extracted audio.')


def createTrainingData(audio_folder):
    start = 1
    end = 19

    X = []
    y = []
    if not os.path.exists(audio_folder):
        print('The folder {} does not exist.'.format(audio_folder))
        return -1
    # r = root, d = directories, f = files
    for r, d, f in os.walk(audio_folder):
        n_directories = len(d)
        ind_d = 1
        for directory in d:
            n_files = len(next(os.walk(r + directory))[2])
            ind_files = 1
            print('\nMFCC for audio file in a folder {}. Left {} folders with audio files\n'.format(directory,
                                                                                                    len(d) - ind_d))
            ind_d = ind_d + 1
            for file in os.listdir(r + directory):
                if not file.endswith('.wav'):
                    print('The file {} not wav format.'.format(file_path))
                    continue
                check_bit = 0
                audio_path = r + directory + '/' + file
                with contextlib.closing(wave.open(audio_path, 'r')) as temp_audio:
                    frames = temp_audio.getnframes()
                    rate = temp_audio.getframerate()
                    duration = frames / float(rate)
                if duration < 1.0:
                    os.system('rm {}'.format(audio_path))
                    print('The audio file {} duration is zero. Hence deleted this file'.format(file))
                    continue

                try:
                    a, s = librosa.load(audio_path)
                except:
                    check_bit = 1
                    print('File {} can\'t open'.format(audio_path))
                if check_bit == 0:
                    mfcc = librosa.feature.mfcc(y=a, sr=s)
                    temp = mfcc.T[1][start:end]
                    for frame in range(10, 50):
                        temp = np.concatenate((temp, mfcc.T[frame][start:end]))
                    answer_id = file[:-4]

                    y.append(answer_id)
                    X.append(temp)
                    CoolPrint(ind_files, n_files)
                    ind_files = ind_files + 1
                else:
                    print('Can not open the audio file {}.'.format(file))
                    continue
    print('\nX length is: {}. y length is: {}\n'.format(len(X), len(y)))
    return X, y

def saveObject(obj, file_name):
    with open(file_name, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


VERSION = 'v02'
project_path = '/mnt/projects/accent_recognition/native-non-native-classification/'
file_path = project_path + 'input/10_video_links_' + VERSION + '.csv'
df_session = pd.read_csv(file_path, sep='\t')

preprocess = False
if preprocess:
    GetVideoLink(df_session['Session recording'], df_session['Accent'])
    video_folder = project_path + 'input/video/test_' + VERSION + '/'
    audio_folder = project_path + 'input/audio/test/' + VERSION + '/'
    Preprocessing(video_folder, audio_folder)

    X_test, y_test = createTrainingData(audio_folder)

    file_path = project_path + 'output/X_test_' + VERSION + '.pkl'
    saveObject(X_test, file_path)
    file_path = project_path + 'output/y_test_' + VERSION + '.pkl'
    saveObject(y_test, file_path)
else:
    file_path = '../output/X_test_' + VERSION + '.pkl'
    with open(file_path, 'rb') as input_:
        X_test = pickle.load(input_)
    file_path = '../output/y_test_' + VERSION + '.pkl'
    with open(file_path, 'rb') as input_:
        y_test = pickle.load(input_)

print(sorted(Counter(y_test).items()))
X_test = np.array(X_test)
model = load_model(project_path + "models/best_CNN_v03.h5")  # rollback to best mod
cols = 18
rows = int(X_test.shape[1] / cols)
n_test_samples = X_test.shape[0]
X_test = X_test.reshape(n_test_samples, rows, cols, 1)
y_pred = model.predict_classes(X_test)

df_result = pd.DataFrame({
    'actual accent': y_test,
    'prediction': y_pred,
})

file_path = project_path + 'output/' + 'df_test_result_' + VERSION + '.csv'
df_result.to_csv(file_path, sep='\t', encoding='utf-8')






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

'''
Used 3 sources of open-data sources and iTutorGroup data-set
1. Using audio samples from [The Speech Accent Archive] (http://accent.gmu.edu/)
2. Using the CSTR VCTK Corpus dataset. Link for Dataset is: https://datashare.is.ed.ac.uk/handle/10283/3443
3. Common Voice is part of Mozilla's initiative to help teach machines how real people speak. 
   Link for dataset is: commonvoice.mozilla.org

1 and 2 datasets could be downloaded by code, and the 3d dataset should be downloaded manually, because need to put 
email address before download the dataset.

'''
import contextlib
import pandas as pd
import os
import pickle
import re
import requests
import time
import urllib.request
import sys
import wave

from bs4 import BeautifulSoup
from pathlib import PurePath
from pydub import AudioSegment
# from src import functions
sys.path.append('../../src/')
import functions


class CreateAccentData:
    files_name: []
    files_media_path: []
    wait = 1.5
    accents = []
    df_accent = pd.DataFrame(columns=["accent", "duration", "media_path", 'file_name'])

    def __init__(self, media_path, df_path, root_url, browse_language_url):
        self.media_path = media_path
        self.df_path = df_path
        self.browse_language_url = browse_language_url
        self.root_url = root_url

    def GetHTMLs(self, urls):
        '''
        Retrieves html in text form from ROOT_URL
        :param urls (list): List of urls from which to retrieve html
        :return (list): list of HTML strings
        '''
        htmls = []
        for url in urls:
            htmls.append(requests.get(url).text)
            time.sleep(self.wait)
        return htmls

    def BuildSearchURLs(self, languages):
        '''
        creates url from ROOT_URL and languages
        :param languages (list): List of languages
        :return (list): List of urls
        '''
        return ([self.root_url + self.browse_language_url.format_media(language) for language in languages])

    def ParsePage(self, p_tag):
        '''
        Extracts href property from HTML <p> tag string
        :param p_tag (str): HTML string
        :return (str): string of link
        '''
        text = p_tag.text.replace(' ', '').split(',')
        return ([self.root_url + p_tag.a['href'], text[0], text[1]])

    def GetAccentInfo(self, hrefs):
        '''
        Retrieves HTML from list of hrefs and returns bio information
        :param hrefs (list): list of hrefs
        :return (DataFrame): Pandas DataFrame with bio information
        '''

        htmls = self.GetHTMLs(hrefs)
        bss = [BeautifulSoup(html, 'html.parser') for html in htmls]
        rows = []
        bio_row = []
        for bs in bss:
            rows.append([li.text for li in bs.find('ul', 'bio').find_all('li')])
        for row in rows:
            bio_row.append(self.ParseAccentInfo(row))

        return (pd.DataFrame(bio_row))

    def GetAccents(self, url):
        hrefs = BeautifulSoup(requests.get(url).text, 'html.parser').find_all('a')
        for href in hrefs:
            if 'language=' in href['href']:
                self.accents.append(href.text)

    def ParseAccentInfo(self, row):
        '''
        Parse bio data from row string
        :param row (str): Unparsed bio string
        :return (list): Bio columns
        '''
        cols = []
        for col in row:
            try:
                tmp_col = re.search((r"\:(.+)", col.replace(' ', '')).group(1))
            except:
                tmp_col = col
            cols.append(tmp_col)
        return (cols)

    def CreateDF(self):
        '''

        :param languages (str): language from which you want to get html
        :return df (DataFrame): DataFrame that contains all audio metadata from searched language
        '''

        # Create a full list of audio from different sources
        browse_language_url = self.root_url + self.browse_language_url.split('?', 1)[0]
        self.GetAccents(browse_language_url)
        htmls = self.GetHTMLs(self.BuildSearchURLs(self.accents))
        bss = [BeautifulSoup(html, 'html.parser') for html in htmls]
        persons = []

        for bs in bss:
            for p in bs.find_all('p'):
                if p.a:
                    persons.append(self.ParsePage(p))

        self.df_bio_info = pd.DataFrame(persons, columns=['href', 'language_num', 'sex'])

        bio_rows = self.GetAccentInfo(self.df_bio_info['href'])

        self.df_bio_info['native_language'] = bio_rows.iloc[:, 1]
        self.df_bio_info['native_language'] = self.df_bio_info['native_language'].apply(lambda x: x.split(' ')[2])
        self.df_bio_info = self.df_bio_info.rename(columns={"href": "link", "language_num": "file_name"})
        self.SaveDF(self.df_bio_info, 'df_bio_info')

    def SaveDF(self, df, name_df):
        functions.saveObject(df, self.df_path / (name_df + ".pkl"))

    def AppendDF(self, path_wav, file_name):
        with contextlib.closing(wave.open(path_wav.as_posix(), 'r')) as temp_audio:
            frames = temp_audio.getnframes()
            rate = temp_audio.getframerate()
            duration = round(frames / float(rate), 2)
        if file_name.startswith('english'):
            accent = 'native'
        else:
            accent = 'non_native'
        self.df_accent = self.df_accent.append({
            'accent': accent,
            'duration': duration,
            'media_path': path_wav,
            'file_name': file_name
        },
            ignore_index=True)

    def GetMedia(self, accent_source, media_url):
        path_df_bio_info = self.df_path / 'df_bio_info.pkl'
        if os.path.exists(path_df_bio_info):
            with open(path_df_bio_info, 'rb') as input_:
                df_bio_info = pickle.load(input_)
        else:
            self.CreateDF()

        dir_media_path = self.media_path / accent_source
        if not os.path.exists(dir_media_path):
            os.mkdir(dir_media_path)
        n_links = len(df_bio_info["link"])
        i = 0
        for file_name in df_bio_info['file_name']:
            i = i + 1
            # Checking is session_sn is not available value
            if file_name is None:
                functions.CoolPrint(i, n_links,
                                    f'-- The value of accent {file_name} is none. Go to the next iteration.')
                continue
            link = media_url.format(file_name)
            path_wav = dir_media_path / (file_name + '.wav')
            if path_wav.exists():
                self.AppendDF(path_wav, file_name)
                functions.CoolPrint(i, n_links, f'-- The media file {file_name} exists.')
                continue
            functions.CoolPrint(i, n_links, f'-- downloading the media file for accent: {file_name}')
            try:
                (filename, headers) = urllib.request.urlretrieve(link)
            except urllib.error.HTTPError as e:
                continue
            audio_file = AudioSegment.from_mp3(filename)
            with open(path_wav, 'wb') as file:
                # get request
                audio_file.export(file, format="wav")
                self.AppendDF(path_wav, file_name)
        self.SaveDF(self.df_accent, 'df_accent_gmu')


class CombineAndCreateDF:
    df_accent = pd.DataFrame(columns=["accent", "duration", "media_path", 'file_name'])

    def __init__(self, dir_external, dir_processed):
        self.dir_external = dir_external
        self.dir_processed = dir_processed
        path_df_audio_accent = self.dir_processed / 'df_accent_gmu.pkl'
        with open(path_df_audio_accent, 'rb') as input_:
            self.df_accent = self.df_accent.append(pickle.load(input_))

    def AnyFormatToWav(self, f_in_record, f_out_record):
        command = 'ffmpeg -i {} {} &> /dev/null'.format(f_in_record, f_out_record)
        os.system(command)
    def AppendDF(self, path_wav, file_name, accent=None):
        with contextlib.closing(wave.open(path_wav, 'r')) as temp_audio:
            frames = temp_audio.getnframes()
            rate = temp_audio.getframerate()
            duration = round(frames / float(rate), 2)
        if duration <= 1.0:
            os.system('rm {}'.format(path_wav))
            print(
                f'The audio file {file_name} duration is less than {duration} seconds. Hence deleted this file')
            return

        self.df_accent = self.df_accent.append({
            'accent': accent,
            'duration': duration,
            'media_path': path_wav,
            'file_name': file_name
        },
            ignore_index=True)

    # Function to transfer audio file from other formats to wav and change rate to 48000
    def CreateVCTK_DF(self, name_df_out, dir_in_media, dir_out_wav, dirs, accent):
        if (self.dir_processed / name_df_out).exists():
            return
        format_media = 'flac'
        path_in_media = self.dir_external / dir_in_media
        path_out_wav = self.dir_external / dir_out_wav
        for dir in dirs:
            ind_files = 0
            n_files = len(next(os.walk(PurePath(path_in_media, dir)))[2])
            for file in os.listdir(PurePath(path_in_media, dir)):
                if not file.endswith(format_media):
                    print(f'The file {file} not {format_media} format.')
                    continue
                ind_files = ind_files + 1
                # Change audio format from flac to wav for Indian dir
                f_in_record = PurePath(path_in_media, dir, file).as_posix()
                length = len(format_media) + 1
                f_out_record = PurePath(path_out_wav, dir + '_wav', file[:-length] + ".wav").as_posix()
                if not os.path.exists(f_out_record):
                    self.AnyFormatToWav(f_in_record, f_out_record)
                    functions.CoolPrint(ind_files, n_files,
                                        f' - The file {file} from {format_media} to wav changed.')
                else:
                    functions.CoolPrint(ind_files, n_files, f'- Audio file {file[:-length]}.wav exist.')

                self.AppendDF(f_out_record, file[:length], accent)

        functions.saveObject(self.df_accent, self.dir_processed / name_df_out)

    # Function to transfer audio file from other formats to wav and change rate to 48000
    def CreateMozillaDF(self, name_df_out, dir_in_media, dir_out_wav):
        format_media = 'mp3'
        # path_in_media = self.dir_external / dir_in_media
        path_in_media = dir_in_media
        path_out_wav = self.dir_external / dir_out_wav
        path_df = path_in_media.as_posix().rsplit('/', 1)[0] + '/train.tsv'
        df = pd.read_csv(path_df, sep='\t')
        df = df.loc[df['accent'].dropna().index, ]
        df.loc[df['accent'] == 'us', 'accent'] = 'native'
        df.loc[df['accent'] == 'england', 'accent'] = 'native'
        df.loc[df['accent'] != 'native', 'accent'] = 'non_native'
        ind_files = 0
        n_files = df.shape[0]
        for file, accent in zip(df['path'], df['accent']):
            if not file.endswith(format_media):
                print(f'The file {file} not {format_media} format.')
                continue
            ind_files = ind_files + 1
            # Change audio format from flac to wav for Indian dir
            f_in_record = PurePath(path_in_media, file).as_posix()
            length = len(format_media) + 1
            f_out_record = PurePath(path_out_wav, file[:-length] + ".wav").as_posix()
            if not os.path.exists(f_out_record):
                self.AnyFormatToWav(f_in_record, f_out_record)
                functions.CoolPrint(ind_files, n_files,
                                    f'- The file {file} from {format_media} to wav changed.')
            else:
                functions.CoolPrint(ind_files, n_files, f'- Audio file {file[:-length]}.wav exist.')

            self.AppendDF(f_out_record, file[:length], accent)

        functions.saveObject(self.df_accent, self.dir_processed / name_df_out)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 10:16:17 2020

The file contains all function that need for project

@author: Farruh Kushnazarov
"""

import pickle

from collections import Counter
from math import floor
from time import localtime, strftime

def CoolPrint(current, total, text="", end='\r', flush=True):
    max_print = 2
    n_print_symbols = 50
    percent = floor(current / total * 100)
    percent_done = floor(current / total * 100 / max_print)
    percent_need = floor((100 - percent_done) / max_print)
    if text == "":
        if current < total:
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()),
                  '{:<4}/{:4}: [{}>{}]'.format(current, total, '=' * percent_done, '.' * percent_need), end=end,
                  flush=flush)
        else:
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '{:4}/{:4}: [{:4}]'.
                  format(current, total, '=' * n_print_symbols))
    else:
        if current < total:
            print(' ' * 120, end=end, flush=flush)
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '{:<4}/{:4}. Percent:{:<3}'.
                  format(current, total, percent), text, end=end, flush=flush)
        else:
            print(strftime("%Y-%m-%d %H:%M:%S", localtime()), '{:4}/{:4}: [{:3}]'.
                  format(current, total, '=' * n_print_symbols))


def saveObject(obj, file_path):
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def SameField(x):
    counter = Counter(x)
    if x.nunique() == 2 and len(x) > 2 or x.nunique() == 1:
        value = list(counter.most_common(1))[0][0]
        return value
    else:
        value = 'Delete'
        return value
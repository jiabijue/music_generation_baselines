#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/15 14:11
# @Github   : https://github.com/BerylJia 
# @File     : dir_to_npy.py

import os
import numpy as np
from pypianoroll import Multitrack

filepath = './lpd_5_cleansed'

num_songs = 21425
cut_len = 400  # min length in this dataset is 600
cut_pitch_range = 84
num_tracks = 5
dataset = np.empty([num_songs, cut_len, cut_pitch_range, num_tracks])

idx = 0
for root, dirs, files in os.walk(filepath):
    for name in files:
        if name.endswith('npz'):
            npz_file = os.path.join(root, name)
            print(npz_file)

            mt = Multitrack(npz_file, beat_resolution=24)
            mt.binarize()
            pr = mt.get_stacked_pianorolls()  # shape=(num_time_step, 128, num_track)

            dataset[idx, ...] = pr[200:600, 20:104, :]
            idx += 1

np.save('lpd_5_cleansed.npy', dataset.astype(np.bool_))



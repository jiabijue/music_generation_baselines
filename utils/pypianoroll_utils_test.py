#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/9 13:00
# @Github   : https://github.com/BerylJia 
# @File     : pypianoroll_utils_test.py

"""
Data sample is from ``lpd-5-cleansed`` (sub-dataset of Lakh Pianoroll Dataset).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pypianoroll import Multitrack
from utils.pypianoroll_utils import pianoroll_to_multitrack

rootpath = os.path.abspath('../')
sys.path.append(rootpath)

TRACK_NAMES = ['drums', 'piano', 'guitar', 'bass', 'strings']
NPZ_FILE = '../data/sample_data/508448c8e2c465d9237dfca9a4c9a265.npz'

# Get Multitrack object and its piano roll
mt = Multitrack(NPZ_FILE, beat_resolution=24)
mt.binarize()
pr = mt.get_stacked_pianorolls()  # shape=(num_time_step, 128, num_track)
# F.Y.I num_time_step = self.beat_resolution * num_beat   # pypianoroll/../multitrack.py line 744
print("Stacked piano-roll matrix: ", np.shape(pr))

# Get piano roll matrix of each track
track_list = mt.tracks
for track in track_list:
    _pr = track.pianoroll
    print("%s of shape: %s" % (track.name, str(list(np.shape(_pr)))))
    # print("  active length of this track: ", track.get_active_length())

# Plot the multi-track piano-roll
fig, axs = mt.plot()
plt.show()

# Convert piano-rolls to Multitrack object
program_nums = [118, 0, 25, 33, 49]  # ref: http://www.360doc.com/content/11/0401/13/3416571_106375312.shtml
is_drums = [True, False, False, False, False]
print(pr.dtype)
mt_out = pianoroll_to_multitrack(stacked_pianorolls=pr,
                                 program_nums=program_nums,
                                 is_drums=is_drums,
                                 track_names=TRACK_NAMES)

# Write Multitrack to midi file
file_path = '../data/sample_data/out_pypianoroll.mid'
mt_out.write(file_path)
print("Write to midi file done.")


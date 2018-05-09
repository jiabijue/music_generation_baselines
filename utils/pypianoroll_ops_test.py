#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/9 13:00
# @Github   : https://github.com/BerylJia 
# @File     : pypianoroll_ops_test.py

"""

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pypianoroll import Multitrack

root_path = os.path.abspath('..')
sys.path.append(root_path)
from utils.pypianoroll_ops import save_midi

# class Dataset:
TRACK_NAMES = ['drums', 'piano', 'guitar', 'bass', 'strings']
npz_file = '../data/sample_data/508448c8e2c465d9237dfca9a4c9a265.npz'

multi_track = Multitrack(npz_file, beat_resolution=24)
pr = multi_track.get_stacked_pianorolls()  # shape=(num_time_step, 128, num_track)
# F.Y.I num_time_step = self.beat_resolution * num_beat   # pypianoroll/../multitrack.py line 744
print("Stacked piano-roll matrix: ", np.shape(pr))

# Get piano roll matrix of each track
track_list = multi_track.tracks
for track in track_list:
    _pr = track.pianoroll
    print("%s of shape: %s" % (track.name, str(list(np.shape(_pr)))))
    # print("  active length of this track: ", track.get_active_length())

# Plot the multi-track piano-roll
fig, axs = multi_track.plot()
plt.show()

# Save piano-rolls to midi file
program_nums = [118, 0, 25, 33, 49]  # ref: http://www.360doc.com/content/11/0401/13/3416571_106375312.shtml
is_drums = [True, False, False, False, False]

save_midi(file_path='../data/sample_data/out_pypianoroll.mid',
          stacked_piano_rolls=pr,
          program_nums=program_nums,
          is_drums=is_drums,
          track_names=TRACK_NAMES)
print("Write to midi file done.")


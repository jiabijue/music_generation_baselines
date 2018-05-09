#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/9 17:03
# @Github   : https://github.com/BerylJia 
# @File     : pypianoroll_ops.py

from pypianoroll import Multitrack, Track


def save_midi(file_path, stacked_piano_rolls, program_nums=None, is_drums=None, track_names=None,
              tempo=80.0, beat_resolution=24):
    """ Write the given piano-roll(s) to a single MIDI file.

    :param file_path:
    :param stacked_piano_rolls: np.ndarray, shape=(num_time_step, 128, num_track)
    :param program_nums:
    :param is_drums:
    :param track_names:
    :param tempo:
    :param beat_resolution:
    :return:
    """

    # Check arguments
    # if not np.issubdtype(stacked_piano_rolls.dtype, np.bool_):
    #     raise TypeError("Support only binary-valued piano-rolls")
    if stacked_piano_rolls.shape[2] != len(program_nums):
        raise ValueError("`stacked_piano_rolls.shape[2]` and `program_nums` must have be the same")
    if stacked_piano_rolls.shape[2] != len(is_drums):
        raise ValueError("`stacked_piano_rolls.shape[2]` and `is_drums` must have be the same")
    if isinstance(program_nums, int):
        program_nums = [program_nums]
    if isinstance(is_drums, int):
        is_drums = [is_drums]
    if program_nums is None:
        program_nums = [0] * len(stacked_piano_rolls)
    if is_drums is None:
        is_drums = [False] * len(stacked_piano_rolls)

    # Write midi
    multitrack = Multitrack(tempo=tempo, beat_resolution=beat_resolution)
    for idx in range(stacked_piano_rolls.shape[2]):
        if track_names is None:
            track = Track(stacked_piano_rolls[..., idx], program_nums[idx],
                          is_drums[idx])
        else:
            track = Track(stacked_piano_rolls[..., idx], program_nums[idx],
                          is_drums[idx], track_names[idx])
        multitrack.append_track(track)
    multitrack.write(file_path)


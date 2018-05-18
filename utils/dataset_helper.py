#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/10 14:12
# @Github   : https://github.com/BerylJia
# @File     : dataset_helper.py

import collections
import numpy as np


class DataSet:
    def __init__(self, data):
        self._data = data
        self._num_examples = data.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end]


def read_data_sets(data):
    num_examples = data.shape[0]

    np.random.shuffle(data)

    # Split ratio 8:1:1
    idx1 = int(num_examples * 0.8)
    idx2 = int(num_examples * 0.9)

    train_data = data[:idx1, ...]
    develop_data = data[idx1:idx2, ...]
    test_data = data[idx2:, ...]

    train = DataSet(train_data)
    develop = DataSet(develop_data)
    test = DataSet(test_data)

    Datasets = collections.namedtuple('Datasets', ['train', 'develop', 'test'])

    return Datasets(train=train, develop=develop, test=test)



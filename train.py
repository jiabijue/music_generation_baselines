#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/16 21:23
# @Github   : https://github.com/BerylJia 
# @File     : train.py


import argparse
import gc
import numpy as np
from utils.dataset_helper import read_data_sets
from models.rnn import RNN
from utils.ops import plot_gif


def get_arguments():
    parser = argparse.ArgumentParser(description='Training procedure.')
    parser.add_argument('--model', type=str, default='rnn',
                        help='Specify the model to train. Optional args:rnn, xxx.  Default: rnn')
    return parser.parse_args()


def main_rnn():
    # Prepare Data
    data_file = './data/lpd_5_cleansed.npy'
    data = np.load(data_file)
    print(np.shape(data))

    data = np.reshape(data, [21425, 400, -1])
    print(np.shape(data))
    print(data.dtype)

    dataset = read_data_sets(data)
    train_set = dataset.train
    develop_set = dataset.develop
    test_set = dataset.test

    # release space
    del data
    del dataset
    gc.collect()

    # Create Model Object
    model = RNN()

    # Train
    log_tag = "20180518-1030"
    model.train(train_set, develop_set, log_tag)
    plot_gif(fig_dir='./logdir/%s/results' % log_tag)  # plot

    # Test
    model.test(test_set, log_tag)


# def main_xxx():
#     print()


if __name__ == '__main__':
    args = get_arguments()

    if args.model.lower() == 'rnn':
        main_rnn()
    # elif args.model.lower() == 'xxx':
    #     main_xxx()

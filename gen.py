#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/16 20:41
# @Github   : https://github.com/BerylJia 
# @File     : gen.py

import argparse
import gc
import numpy as np
import tensorflow as tf
from models.rnn import RNN
from utils.dataset_helper import DataSet


def get_arguments():
    parser = argparse.ArgumentParser(description='Training procedure.')
    parser.add_argument('--model', type=str, default='rnn',
                        help='Specify the model to generate. Optional args:rnn, rgan.  Default: rnn')
    parser.add_argument('--ckpt', type=str, default='./logdir/20180518-1030/train/model.ckpt-19',
                        help='Specify the checkpoint file.')
    parser.add_argument('--filename', type=str, default='./logdir/20180518-1030/results/gen.mid',
                        help='Specify the generated midi file name.')
    return parser.parse_args()


def main_rnn(ckpt, filename):
    # Load Data for Inference
    data = np.load('./data/lpd_5_cleansed.npy')
    data = np.reshape(data, [21425, 400, -1])
    dataset = DataSet(data)

    del data
    gc.collect()

    # Model Object
    model = RNN()

    # Restore Model Variables
    saver = tf.train.Saver()
    print('Restoring model from {} \n'.format(ckpt))
    saver.restore(model.sess, ckpt)

    # Generate
    model.generate(dataset, filename)
    print("Write to midi file %s done.\n" % filename)

    # Pass filename to web page


# def main_xxx(ckpt, filename):
#     print()


if __name__ == '__main__':
    args = get_arguments()

    if args.model.lower() == 'rnn':
        main_rnn(args.ckpt, args.filename)
    # elif args.model.lower() == 'xxx':
    #     main_rnn(args.ckpt, args.filename)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/4/18 15:20
# @Github   : https://github.com/BerylJia 
# @File     : ops.py

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import tensorflow as tf
from utils.pypianoroll_utils import pianoroll_to_multitrack


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def pad_pianoroll_to_multitrack(pr):
    pad_note = np.concatenate((np.zeros((pr.shape[0], 24, pr.shape[2])), pr,
                               np.zeros((pr.shape[0], 20, pr.shape[2]))), axis=1)
    bool_pr = np.around(pad_note).astype(np.bool_)
    mt = pianoroll_to_multitrack(stacked_pianorolls=bool_pr,
                                 program_nums=[118, 0, 24, 33, 49],
                                 is_drums=[True, False, False, False, False],
                                 track_names=['drums', 'piano', 'guitar', 'bass', 'strings'])
    return mt


# Plot&save Multitrack piano roll figure
def plot_save(output_dir, multi_track, epoch=None):
    check_path(output_dir)
    fig, axs = multi_track.plot()
    # plt.show()

    if epoch is None:
        title = 'Test'
    else:
        title = 'Epoch-%s' % str(epoch).zfill(3)
    fig.suptitle(title)
    plt.savefig(os.path.join(output_dir, title))
    # plt.close(fig)


def plot_gif(fig_dir):
    file_list = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
    file_list.sort()
    frames = [imageio.imread(os.path.join(fig_dir, file)) for file in file_list]
    imageio.mimsave(os.path.join(fig_dir, 'Train.gif'), frames, duration=0.5)


def plot_q_z(x, y, filename):
    from sklearn.manifold import TSNE
    colors = ["#2103c8", "#0e960e", "#e40402", "#05aaa8", "#ac02ab", "#aba808", "#151515", "#94a169", "#bec9cd",
              "#6a6551"]

    plt.clf()
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    if x.shape[1] != 2:
        x = TSNE().fit_transform(x)
    y = y[:, np.newaxis]
    xy = np.concatenate((x, y), axis=1)

    for l, c in zip(range(10), colors):
        ix = np.where(xy[:, 2] == l)
        ax.scatter(xy[ix, 0], xy[ix, 1], c=c, marker='o', label=l, s=10, linewidths=0)

    plt.savefig(filename)
    plt.close()


# Optimizers
def create_adam_optimizer(learning_rate, beta1=0.9, beta2=0.999):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  beta1=beta1,
                                  beta2=beta2)


def create_nadam_optimizer(learning_rate):
    return tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate,
                                         epsilon=1e-8)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum=0.0):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum)


optimizer_factory = {'adam': create_adam_optimizer,
                     'nadam': create_nadam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


# Universal operations
def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


# Tensorboard
def summary_setup(logdir, sess):
    """ Set up logging for TensorBoard. """
    check_path(logdir)

    summaries = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir, sess.graph)
    return summaries, writer


def save(saver, sess, logdir, epoch):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end='')
    sys.stdout.flush()

    check_path(logdir)
    saver.save(sess, checkpoint_path, global_step=epoch)
    print(' Done.')


def load(saver, sess, logdir):
    print('\nTrying to restore saved checkpoints from {} ...'.format(logdir), end='')

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print('  Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('  Global epoch was: {}'.format(global_epoch))
        print('  Restoring...', end='')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(' Done.\n')
        return global_epoch
    else:
        print(' No checkpoint found.\n')
        return None

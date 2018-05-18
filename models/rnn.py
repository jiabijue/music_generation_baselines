#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Bijue Jia
# @Datetime : 2018/5/16 20:53
# @Github   : https://github.com/BerylJia 
# @File     : rnn.py


import os
import sys
import numpy as np
import tensorflow as tf

rootpath = os.path.abspath('../..')
sys.path.append(rootpath)
from utils.ops import optimizer_factory, init_sess, summary_setup, save, load,\
    pad_pianoroll_to_multitrack, plot_save


class RNN:
    def __init__(self, hid_list=[128,128], in_dim=420, n_steps=400, rnn_cell='bilstm',
                 lr=1e-3, keep_prob=1.0, batch_size=16, num_epoch=20):
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.n_steps = n_steps
        self.hid_list = hid_list
        self.n_layers = len(hid_list)
        self.cell = rnn_cell
        if rnn_cell == 'lstm' or rnn_cell == 'gru':
            self.use_birnn = False
        elif rnn_cell == 'bilstm' or rnn_cell == 'bigru':
            self.use_birnn = True

        self.lr = lr
        self.keep_prob = keep_prob
        self.bz = batch_size
        self.n_epoch = num_epoch

        self._build_graph()
        self.sess = init_sess()

    def _build_graph(self):
        # Input
        self.x = tf.placeholder(tf.float32, [None, self.n_steps, self.in_dim])
        self.y = tf.placeholder(tf.float32, [None, self.n_steps, self.out_dim])

        # Forward compute
        self._forward()

        # Back propagation path
        self._backward()

        # add summaries to tensorboard
        self._build_summaries()

    def _define_rnn_cell(self, size, reuse=False):
        if self.cell == 'lstm' or self.cell == 'bilstm':
            return tf.nn.rnn_cell.LSTMCell(size, initializer=tf.orthogonal_initializer(), reuse=reuse)
        elif self.cell == 'gru' or self.cell == 'bigru':
            return tf.nn.rnn_cell.GRUCell(size, kernel_initializer=tf.orthogonal_initializer(), reuse=reuse)

    def _forward(self):
        with tf.name_scope('rnn_layer'):
            if self.use_birnn:
                rnn_layers_fw = [tf.nn.rnn_cell.DropoutWrapper(cell=self._define_rnn_cell(size),
                                                               output_keep_prob=self.keep_prob)
                                 for size in self.hid_list]
                rnn_layers_bw = [tf.nn.rnn_cell.DropoutWrapper(cell=self._define_rnn_cell(size),
                                                               output_keep_prob=self.keep_prob)
                                 for size in self.hid_list]
                m_rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_fw)
                m_rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell(rnn_layers_bw)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=m_rnn_cell_fw,
                    cell_bw=m_rnn_cell_bw,
                    inputs=self.x,
                    dtype=tf.float32)
                current = tf.concat(outputs, 2)  # (batch_size, n_steps, 2*hid_list[-1])
            else:
                rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(cell=self._define_rnn_cell(size),
                                                            output_keep_prob=self.keep_prob)
                              for size in self.hid_list]
                m_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
                outputs, final_states = tf.nn.dynamic_rnn(  # final_states is a tuple of len=num_layers
                    cell=m_rnn_cell,
                    inputs=self.x,
                    dtype=tf.float32)
                current = outputs  # (batch_size, n_steps, hid_list[-1])

        with tf.name_scope('output_layer'):
            # Reshape data so that it can multiply with W
            if self.use_birnn:
                current = tf.reshape(current, [-1, 2 * self.hid_list[-1]])
            else:
                current = tf.reshape(current, [-1, self.hid_list[-1]])
            # Compute output
            current = tf.layers.dense(current, units=self.out_dim)
            # Reshape data back
            self.logits = tf.reshape(current, [-1, self.n_steps, self.out_dim])
            self.out = tf.sigmoid(self.logits)

    def _backward(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.train_op = optimizer_factory['adam'](self.lr).minimize(self.loss)

    def _build_summaries(self):
        tf.summary.scalar('loss', self.loss)

    def train(self, train_set, develop_set, log_tag):
        logdir = './logdir/%s/train' % log_tag
        summaries, writer = summary_setup(logdir, self.sess)
        dev_writer = tf.summary.FileWriter('./logdir/%s/develop' % log_tag)

        # Prepare to save checkpoints
        saver = tf.train.Saver()
        try:
            saved_global_epoch = load(saver, self.sess, logdir)
            if saved_global_epoch is None:
                saved_global_epoch = 0
        except:
            print("Something went wrong while restoring checkpoint. "
                  "We will terminate training to avoid accidentally overwriting "
                  "the previous model.\n")
            raise

        epoch = None
        last_saved_epoch = saved_global_epoch
        step = 0
        try:
            for epoch in range(saved_global_epoch + 1, self.n_epoch + 1):
                for batch in range(1, train_set.num_examples // self.bz + 1):
                    next_batch = train_set.next_batch(self.bz)
                    x_data = np.reshape(next_batch, [self.bz, self.n_steps, self.out_dim])
                    y_data = np.roll(x_data, shift=-1, axis=1)
                    _, loss, sm = self.sess.run([self.train_op, self.loss, summaries], {self.x: x_data, self.y: y_data})
                    writer.add_summary(sm, step)

                    dev_batch = develop_set.next_batch(self.bz)
                    dev_x = np.reshape(dev_batch, [self.bz, self.n_steps, self.out_dim])
                    dev_y = np.roll(dev_x, -1, 1)
                    dev_loss, dev_sm, out = self.sess.run([self.loss, summaries, self.out], {self.x: dev_x, self.y: dev_y})
                    dev_writer.add_summary(dev_sm, step)

                    step += 1

                    if batch % 100 == 0:
                        print("Epoch %d [batch %d] > train loss: %.4f develop loss: %.4f " % (epoch, batch, loss, dev_loss))

                # Save middle output (figure and midi)
                pr = np.reshape(out[0], [self.n_steps, 84, 5])
                mt = pad_pianoroll_to_multitrack(pr)
                plot_save('./logdir/%s/results' % log_tag, mt, epoch)
                mt.write('./logdir/%s/results/Develop_%d.mid' % (log_tag, epoch))

                # Save checkpoint
                save(saver, self.sess, logdir, epoch)
                last_saved_epoch = epoch
        except KeyboardInterrupt:
            print()
        finally:
            if epoch > last_saved_epoch:
                save(saver, self.sess, logdir, epoch)
            print("Training done.\n")

    def test(self, test_set, log_tag):
        test_batch = test_set.next_batch(self.bz)
        test_x = np.reshape(test_batch, [self.bz, self.n_steps, self.out_dim])
        test_y = np.roll(test_x, -1, 1)
        loss, out = self.sess.run([self.loss, self.out], {self.x: test_x, self.y: test_y})
        print("Test loss: %.4f " % loss)

        pr = np.reshape(out[0], [self.n_steps, 84, 5])
        mt = pad_pianoroll_to_multitrack(pr)
        plot_save('./logdir/%s/results' % log_tag, mt)
        mt.write('./logdir/%s/results/Test.mid' % log_tag)
        print("Write to midi file './logdir/%s/results/Test.mid' done.\n" % log_tag)

    def generate(self, dataset, filename):
        next_batch = dataset.next_batch(1)
        x = np.reshape(next_batch, [1, self.n_steps, self.out_dim])
        out = self.sess.run(self.out, {self.x: x})

        pr = np.reshape(out, [self.n_steps, 84, 5])
        mt = pad_pianoroll_to_multitrack(pr)
        mt.write(filename)


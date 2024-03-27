# coding: utf-8
#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
BATCH_SIZE = 256
SLOT_SIZE =1024
EMB_DIM = 16
INPUT_SIZE = SLOT_SIZE * EMB_DIM
LEARNING_RATE = 0.01
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=[None, SLOT_SIZE, EMB_DIM], name="embeddings")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="label")
"""fm part"""
with tf.name_scope("fm_layers"):
    w = tf.placeholder(tf.float32,[None, 1],name="w")
    b = tf.Variable([0.0], name="b")
    fm_input = x
    sum = tf.reduce_sum(fm_input, axis=1)
    sum_square = tf.square(sum)
    sum2 = tf.reduce_sum(tf.square(fm_input), axis=1)
    fm_cross = tf.multiply(0.5, tf.reduce_sum(tf.subtract(sum_square,sum2), axis=1, keep_dims=True))
    z = b + w + fm_cross
sum = z
predict_out = tf.sigmoid(sum, name="predict_out")
labels = tf.identity(y, "labels")
loss = tf.identity(0.5 * tf.reduce_mean(tf.square(y - sum)), name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,beta1=0.9, beta2=0.999,epsilon=1e-8)
    train_step = optimizer.minimize(loss)

init = tf.group([tf.local_variables_initializer(), tf.global_variables_initializer()], name="weidl_init")
"""run"""
with tf.Session() as sess:
    sess.run(init)
    tf.train.write_graph(sess.graph_def, "./", "model_online.pb", as_text=False)


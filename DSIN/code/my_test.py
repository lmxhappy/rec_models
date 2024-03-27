# coding: utf-8
from deepctr.layers.sequence import (AttentionSequencePoolingLayer, BiasEncoding,
                                     BiLSTM, Transformer)
import tensorflow as tf

if __name__ == '__main__':
    sess_max_count = 5
    tr_input = [tf.random.uniform(shape=(16, 10, 8)) for _ in range(sess_max_count)]
    tr_input = BiasEncoding(sess_max_count)(tr_input)

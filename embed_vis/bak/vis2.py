# coding: utf-8
# from tensorflow.contrib.tensorboard.plugins import projector
from tensorboard.plugins import projector

import tensorflow as tf
import os

# Create randomly initialized embedding weights which will be trained.
N = 10000 # Number of items (vocab size).
D = 200 # Dimensionality of the embedding.
embedding_var = tf.Variable(tf.random.normal([N,D]), name='word_embedding')
LOG_DIR = './tb_log'

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), 1)

# Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
config = projector.ProjectorConfig()

# You can add multiple embeddings. Here we add only one.
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

# Use the same LOG_DIR where you stored your checkpoint.
summary_writer = tf.summary.FileWriter(LOG_DIR)

# The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
# read this file during startup.
projector.visualize_embeddings(summary_writer, config)
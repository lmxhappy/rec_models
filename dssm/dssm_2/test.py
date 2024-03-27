# coding: utf-8
import math

import tensorflow as tf

labels = [[0, 0, 1],
          [0, 1, 0]]
logits = [[4, 1, -2],
          [0.1, 1, 3]]

logits_scaled = tf.nn.softmax(logits)
result = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

with tf.Session() as sess:
    print(sess.run(logits_scaled))
    print(sess.run(result))

logits_scale = [[0.95033026, 0.04731416, 0.00235563], [0.04622407, 0.11369288, 0.84008306]]


def bce(logit, label):
    sum = 0
    for i, v in enumerate(logit):
        l = label[i]
        if v > 0:
            loss = l * math.log(v)
        else:
            loss = (1 - l) * math.log(1 - v)

        sum += loss

    return -sum


def logloss(logits, labels):
    batch_size = len(logits)
    # dimension = len(logits[0])

    for i in range(batch_size):
        logit = logits[i]
        label = labels[i]
        loss = bce(logit, label)


if __name__ == '__main__':
    logit = [0.95033026, 0.04731416, 0.00235563]
    label = [0, 0, 1]
    loss = bce(logit, label)
    print(loss)

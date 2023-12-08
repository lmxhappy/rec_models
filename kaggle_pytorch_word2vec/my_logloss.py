# coding: utf-8
# !/usr/bin/env python
# -*- coding: utf8 -*-
# author: klchang
# date: 2018.6.23

# y_true: list, the true labels of input instances
# y_pred: list, the probability when the predicted label of input instances equals to 1
def logloss(y_true, y_pred, eps=1e-15):
    import numpy as np

    # Prepare numpy array data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps
    p = np.clip(y_pred, eps, 1 - eps)
    first = y_true * np.log(p)
    second = (1 - y_true) * np.log(1 - p)
    loss = - np.sum(first + second)

    return loss / len(y_true)


def unitest():
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.2, 0.7, 0.99]

    print("Use self-defined logloss() in binary classification, the result is {}".format(logloss(y_true, y_pred)))

    from sklearn.metrics import log_loss
    print("Use log_loss() in scikit-learn, the result is {} ".format(log_loss(y_true, y_pred)))


if __name__ == '__main__':
    unitest()

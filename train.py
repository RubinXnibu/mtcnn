# -*- coding: utf-8 -*-

__author__ = 'smdsbz <https://www.github.com/smdsbz>'


import numpy as np
import tensorflow as tf

from dataloader import train_input_fn


mode = tf.estimator.ModeKeys.TRAIN


# create the only model instance
from mtcnn.mtcnn import MTCNN
model = MTCNN(trainable=True)


def model_fn(features, labels, mode):

    landmark_coord_pred = model.onet.get_layer('fc2-3')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode,
            predictions=landmark_coord_pred
        )

    # construct loss function
    loss = tf.losses.huber_loss(labels=labels, predictions=landmark_coord_pred)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss
        )

    # build train_op
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op
        )
    raise ValueError('unrecognized mode: {}'.format(mode))


# run
if __name__ == '__main__':

    landmark_106_model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir='./log'
    )

    landmark_106_model.train(
        input_fn=lambda: train_input_fn('./dataset/utils/example.record', 1),
        max_steps=int(1e5)
    )

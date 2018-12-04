# -*- coding: utf-8 -*-

import os

SAVE_DIR_BASE = './saves'
SAVE_NAME = 'test'

PB_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME)
TFLITE_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME + '.tflite')

if os.path.exists(PB_SAVE_PATH):
    os.remove(PB_SAVE_PATH)


# save as `.pb` file

from mtcnn.mtcnn import MTCNN
model = MTCNN(save_path=PB_SAVE_PATH)


# save as `.tflite` file

import tensorflow as tf

converter = tf.contrib.lite.TFLiteConverter.from_saved_model(PB_SAVE_PATH)
tflite_model = converter.convert()
open(TFLITE_SAVE_PATH, 'wb').write(tflite_model)

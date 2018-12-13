# -*- coding: utf-8 -*-

import os

SAVE_DIR_BASE = './saves'
SAVE_NAME = 'Rnet'

PB_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME)
TFLITE_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME + '.tflite')

import numpy as np
import pkg_resources

from mtcnn.mtcnn import PNet, RNet, ONet
from mtcnn.layer_factory import LayerFactory
from mtcnn.network import Network
import tensorflow as tf

with pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy') as weight_file:
    weights = np.load(weight_file).item()


class OriginalRNet(Network):
    """
    original RNet Model  copied from  mtcnn.py
    """

    def _config(self):

        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, 24, 24, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=28, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=48, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
        layer_factory.new_conv(name='conv3', kernel_size=(2, 2), channels_output=64, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_fully_connected(name='fc1', output_count=128, relu=False)  # shouldn't the name be "fc1"?
        layer_factory.new_prelu(name='prelu4')
        layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False)   # shouldn't the name be "fc2-1"?
        layer_factory.new_softmax(name='prob1', axis=1)

        layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, input_layer_name='prelu4')

    def _feed(self, image):
        return self._session.run(['rnet/fc2-2/fc2-2:0', 'rnet/prob1:0'], feed_dict={'rnet/input:0': image})
    
    
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:

        rnet = OriginalRNet(session=sess)
        rnet.set_weights(weights['RNet'], ignore_missing=True)

        converter = tf.contrib.lite.TFLiteConverter.from_session(
            sess=sess,
            input_tensors=[
                rnet.get_layer('data')
            ],
            output_tensors=[
                rnet.get_layer('fc2-2'),
                rnet.get_layer('prob1')
            ]

        )
        tflite_rnet = converter.convert()
        open('saves/rnet.tflite', 'wb').write(tflite_rnet)

print('All Done!')

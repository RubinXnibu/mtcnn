# -*- coding: utf-8 -*-

import os

SAVE_DIR_BASE = './saves'
SAVE_NAME = 'ONet'

PB_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME)
TFLITE_SAVE_PATH = os.path.join(SAVE_DIR_BASE, SAVE_NAME + '.tflite')

# if os.path.exists(PB_SAVE_PATH):
#     os.remove(PB_SAVE_PATH)


# save as `.pb` file

import numpy as np
import pkg_resources

from mtcnn.mtcnn import PNet, RNet, ONet
from mtcnn.layer_factory import LayerFactory
from mtcnn.network import Network
import tensorflow as tf

with pkg_resources.resource_stream('mtcnn', 'data/mtcnn_weights.npy') as weight_file:
    weights = np.load(weight_file).item()


# class OriginalONet(Network):
#      def _config(self):
#         layer_factory = LayerFactory(self)
#
#         layer_factory.new_feed(name='data', layer_shape=(None, 48, 48, 3))
#         layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
#                                padding='VALID', relu=False)
#         layer_factory.new_prelu(name='prelu1')
#         layer_factory.new_max_pool(name='pool1', kernel_size=(3, 3), stride_size=(2, 2))
#         layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
#                                padding='VALID', relu=False)
#         layer_factory.new_prelu(name='prelu2')
#         layer_factory.new_max_pool(name='pool2', kernel_size=(3, 3), stride_size=(2, 2), padding='VALID')
#         layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=64, stride_size=(1, 1),
#                                padding='VALID', relu=False)
#         layer_factory.new_prelu(name='prelu3')
#         layer_factory.new_max_pool(name='pool3', kernel_size=(2, 2), stride_size=(2, 2))
#         layer_factory.new_conv(name='conv4', kernel_size=(2, 2), channels_output=128, stride_size=(1, 1),
#                                padding='VALID', relu=False)
#         layer_factory.new_prelu(name='prelu4')
#
#         ######## layers before this line belone to feature extraction module ########
#
#         ######## layers added to ONet after this line are output / bottleneck layers ########
#
#         # commonly shared fully connected layer
#         # TODO: try larger output count
#         layer_factory.new_fully_connected(name='fc1', output_count=256, relu=False, trainable=self.is_trainable())
#         # layer_factory.new_fully_connected(name='fc1', output_count=1024, relu=False, trainable=self.is_trainable())
#         layer_factory.new_prelu(name='prelu5')
#
#         # (logistic based) confidence / probability output layer
#         layer_factory.new_fully_connected(name='fc2-1', output_count=2, relu=False, trainable=self.is_trainable())
#         layer_factory.new_softmax(name='prob1', axis=1)
#
#         # `( x_base, y_base, width, height )`-output layer
#         layer_factory.new_fully_connected(name='fc2-2', output_count=4, relu=False, trainable=self.is_trainable(),
#                                           input_layer_name='prelu5')
#
#         # landmark output layer
#         # TODO: to 106-point, i.e. 106 * 2 = 212 outputs
#         layer_factory.new_fully_connected(name='fc2-3', output_count=10, relu=False, input_layer_name='prelu5')
#         # layer_factory.new_fully_connected(name='fc2-3', output_count=68*2, relu=False, trainable=self.is_trainable(),
#         #                                   input_layer_name='prelu5')
#      def _feed(self, image):
#         return self._session.run(['onet/fc2-2/fc2-2:0', 'onet/fc2-3/fc2-3:0', 'onet/prob1:0'],feed_dict={'onet/input:0': image})



graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:

        onet = ONet(session=sess)
        onet.set_weights(weights['ONet'], ignore_missing=True)

        tsr_window_in = onet.get_layer('data')
        tsr_reg_out = onet.get_layer('fc2-2')
        tsr_landmark_out = onet.get_layer('fc2-3')  # 5 points
        tsr_prob_out = onet.get_layer('prob1')

        print('window_in.shape:', tsr_window_in.shape)
        print('reg_out.shape:', tsr_reg_out.shape)
        print('landmark_out.shape:', tsr_landmark_out.shape)
        print('prob_out.shape:', tsr_prob_out.shape)

        converter = tf.lite.TFLiteConverter.from_session(
            sess=sess,
            input_tensors=[
                tsr_window_in
            ],
            output_tensors=[
                tsr_reg_out,
                tsr_landmark_out,
                tsr_prob_out
            ]
        )
        tflite_onet = converter.convert()
        open('saves/onet.tflite', 'wb').write(tflite_onet)

print('All Done!')

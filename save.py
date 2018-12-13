# -*- coding: utf-8 -*-

import os


INPUT_BOX_SIZE = 1024


SAVE_DIR_BASE = './saves'
SAVE_NAME = 'test'

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


class PNetFixed(Network):
    '''
    Proposal network with fixed size, enabling compatibility with TFLite
    '''
    def _config(self):
        layer_factory = LayerFactory(self)

        layer_factory.new_feed(name='data', layer_shape=(None, INPUT_BOX_SIZE, INPUT_BOX_SIZE, 3))
        layer_factory.new_conv(name='conv1', kernel_size=(3, 3), channels_output=10, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu1')
        layer_factory.new_max_pool(name='pool1', kernel_size=(2, 2), stride_size=(2, 2))
        layer_factory.new_conv(name='conv2', kernel_size=(3, 3), channels_output=16, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu2')
        layer_factory.new_conv(name='conv3', kernel_size=(3, 3), channels_output=32, stride_size=(1, 1),
                               padding='VALID', relu=False)
        layer_factory.new_prelu(name='prelu3')
        layer_factory.new_conv(name='conv4-1', kernel_size=(1, 1), channels_output=2, stride_size=(1, 1), relu=False)
        layer_factory.new_softmax(name='prob1', axis=3)

        layer_factory.new_conv(name='conv4-2', kernel_size=(1, 1), channels_output=4, stride_size=(1, 1),
                               input_layer_name='prelu3', relu=False)

    def _feed(self, image):
        return self._session.run(['pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'], feed_dict={'pnet/input:0': image})



graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:

        pnet = PNetFixed(session=sess)
        pnet.set_weights(weights['PNet'], ignore_missing=True)

        tsr_img_in = pnet.get_layer('data')
        tsr_raw_reg_out = pnet.get_layer('conv4-2')
        tsr_prob_out = pnet.get_layer('prob1')

        print('img_in.shape:', tsr_img_in.shape)
        print('raw_reg_out.shape:', tsr_raw_reg_out.shape)
        print('prob_out.shape:', tsr_prob_out.shape)

        converter = tf.contrib.lite.TFLiteConverter.from_session(
            sess=sess,
            input_tensors=[
                tsr_img_in
            ],
            output_tensors=[
                tsr_raw_reg_out,
                tsr_prob_out
            ]
        )
        tflite_pnet = converter.convert()
        open('saves/pnet.tflite', 'wb').write(tflite_pnet)

print('All Done!')

# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 12 Mar, 2021

Author : chenlangscu@163.com
"""

import os
import sys

import tensorflow as tf

src_path = os.path.abspath("..")
sys.path.append(src_path)


class SoftmaxWeightedSum(tf.keras.layers.Layer):
    """
    :param align:           [batch_size, 1, T]
    :param value:           [batch_size, T, units]
    :param key_masks:       [batch_size, 1, T]
                            2nd dim size with align
    :param drop_out:
    :param future_binding:
    :return:                after weighted sum vector: [batch_size, 1, units]
                            before weighted sum vector: [batch_size, T, units]  (目前使用)
    """

    def __init__(self, dropout_rate=0.2, seed=2020, **kwargs):
        self.seed = seed
        self.dropout = tf.keras.layers.Dropout(dropout_rate, seed=self.seed)
        super(SoftmaxWeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError('A `SoftmaxWeightedSum` layer should be called '
                             'on a list of 3 tensors')
        if input_shape[0][-1] != input_shape[2][-1]:
            raise ValueError('query_size should keep the same dim with key_mask_size')

        super(SoftmaxWeightedSum, self).build(input_shape)

    def call(self, inputs, mask=None, training=None, **kwargs):
        """

        :param inputs:
        :param mask:
        :param training:
        :param kwargs:
        :return: [batch_size, T, units]
        """
        # align: [batch_size, 1, T]
        # value: [batch_size, T, units]
        # key_masks: [batch_size, 1, T]
        align, value, key_masks = inputs
        paddings = tf.ones_like(align) * (-2 ** 32 + 1)  # [batch_size, 1, T]
        align = tf.where(key_masks, align, paddings)     # [batch_size, 1, T]
        align = tf.nn.softmax(align)                     # [batch_size, 1, T]
        align = self.dropout(align, training=training)   # [batch_size, 1, T]
        # output = tf.matmul(align, value)

        align = tf.transpose(align, perm=[0, 2, 1])      # [batch_size, T, 1]
        output = value * align                           # [batch_size, T, units]
        return output

    def compute_output_shape(self, input_shape):
        return None, 1, input_shape[1][1]

    def get_config(self, ):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(SoftmaxWeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return mask

# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 12 Mar, 2021

Author : chenlangscu@163.com
"""

import tensorflow as tf
import sys
import os
import numpy as np

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.concat_attention import ConcatAttention
from layers.softmax_weight_sum import SoftmaxWeightedSum


class Item2ItemLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, pos_size, vocab_dim, pos_dim, dropout_rate=0, **kwargs):
        super(Item2ItemLayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.vocab_dim = vocab_dim
        self.pos_dim = pos_dim
        self.concat_att = ConcatAttention()
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=dropout_rate)

        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.vocab_dim)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=self.pos_size, output_dim=self.pos_dim)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError('A `Item2ItemLayer` layer should be called '
                             'on a list of 4 tensors')

        super(Item2ItemLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        """
        :param mask: zero_padding
        :param inputs:
        i2i_layer: ([batch_size,T],[batch_size,T],[batch_size,1],[batch_size,1])
        :param kwargs:
        :return:
        """
        items, position, target, keys_length = inputs

        items = self.item_embedding(items)       # [batch_size,T,item_embedding]
        position = self.pos_embedding(position)  # [batch_size,T,position_embedding]
        target = self.item_embedding(target)     # [batch_size,1,target_embedding]

        hist_len = items.get_shape()[1]  # T

        target_tile = tf.tile(target, [1, hist_len, 1])  # [batch_size, T, target_embedding]
        attention_score = self.concat_att([items, position, target_tile])  # [batch_size, 1, T]

        key_masks = tf.sequence_mask(keys_length, hist_len)  # [batch_size,1,T]
        sum_pooling_before = self.softmax_weight_sum(
            [attention_score, items, key_masks])  # [batch_size,T,item_embedding]
        sum_pooling = tf.reduce_sum(sum_pooling_before, axis=1, keepdims=True)  # [batch_size,1,item_embedding]

        attention_score = tf.reduce_sum(attention_score, axis=-1, keepdims=True)   # [batch_size, 1, 1]

        output = tf.concat([sum_pooling, attention_score, target], axis=-1)   # [batch_size,1,1+item_emb+target_emb]
        output = tf.squeeze(output, axis=1)  # [batch_size,1+item_embedding+target_embedding]
        return output


def debug():
    batch_size = 32
    vocab_size = 100
    pos_size = 80
    his_length = 16
    item_dim = 12
    pos_dim = 6

    model = Item2ItemLayer(vocab_size, pos_size, item_dim, pos_dim)

    items = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    position = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    target = np.random.randint(low=0, high=his_length, size=(batch_size, 1))

    # items = tf.random.normal([batch_size, his_length, item_dim])
    # position = tf.random.normal([batch_size, his_length, pos_dim])
    # target = tf.random.normal([batch_size, 1, item_dim])
    keys_length = np.random.randint(low=1, high=his_length+1, size=(batch_size, 1))
    keys_length = tf.convert_to_tensor(keys_length)

    print("items:", items.shape)
    print("position:", position.shape)
    print("target:", target.shape)
    print("keys_length:", keys_length.shape)

    m = model([items, position, target, keys_length])
    print("model: ", m)
    print("model shape: ", m.shape)


if __name__ == "__main__":
    print(tf.__version__)
    debug()

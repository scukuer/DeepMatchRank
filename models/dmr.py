# !/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 11 Mar, 2021

Author : chenlangscu@163.com
"""

import tensorflow as tf
import numpy as np
import sys
import os

src_path = os.path.abspath("..")
sys.path.append(src_path)

from layers.item2item import Item2ItemLayer
from layers.user2item import User2ItemLayer


class DeepMatchRank(tf.keras.Model):
    """
    定义DMR网络
    """
    def __init__(self, vocab_size, pos_size, vocab_dim, pos_dim, num_samples=5, dropout_rate=0, **kwargs):
        super(DeepMatchRank, self).__init__(**kwargs)
        self.i2i_layer = Item2ItemLayer(vocab_size, pos_size, vocab_dim, pos_dim, dropout_rate)
        self.u2i_layer = User2ItemLayer(vocab_size, pos_size, vocab_dim, pos_dim, num_samples, dropout_rate)

        self.layer_1 = tf.keras.layers.Dense(64)
        self.prelu_1 = tf.keras.layers.PReLU()
        self.layer_2 = tf.keras.layers.Dense(32)
        self.prelu_2 = tf.keras.layers.PReLU()

        self.sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')

        self.aux_loss = 0.0

    def call(self, inputs, training=None, mask=None):
        """
        :param inputs:([batch_size,T],[batch_size,T],[batch_size,1],[batch_size,1])
        :param training
        :param mask
        """

        # items_id, position_id, target_id, keys_length = inputs

        # print("@@@item:", items_id.shape)
        # print("@@@position:", position_id.shape)
        # print("@@@target:", target_id.shape)
        # print("@@@keys_length:", keys_length.shape)
        #
        # print("@@@item:", items_id)
        # print("@@@position:", position_id)
        # print("@@@target:", target_id)
        # print("@@@keys_length:", keys_length)

        i2i_layer_out = self.i2i_layer(inputs)              # [batch_size,1+item_embedding+target_embedding]
        u2i_layer_out, aux_loss = self.u2i_layer(inputs)    # [batch_size,1]

        self.aux_loss = float(aux_loss)                     # scala

        # [batch_size,2+item_emb+target_emb]
        concat_layer = tf.keras.layers.concatenate([i2i_layer_out, u2i_layer_out], axis=-1)
        layer_1 = self.layer_1(concat_layer)
        prelu_layer_1 = self.prelu_1(layer_1)
        layer_2 = self.layer_2(prelu_layer_1)
        prelu_layer_2 = self.prelu_2(layer_2)
        output_layer = self.sigmoid_layer(prelu_layer_2)   # [batch_size,1]

        return output_layer

    def aux_loss_fn(self):
        """
        batch auxiliary loss
        """
        return float(self.aux_loss)


def debug():
    batch_size = 32
    vocab_size = 100
    pos_size = 80
    his_length = 16
    item_dim = 12
    pos_dim = 6
    num_sample = 60

    dmr = DeepMatchRank(vocab_size, pos_size, item_dim, pos_dim, num_sample)

    item = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    position = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    target = np.random.randint(low=0, high=his_length, size=(batch_size, 1))

    # items = tf.random.normal([batch_size, his_length, item_dim])
    # position = tf.random.normal([batch_size, his_length, pos_dim])
    # target = tf.random.normal([batch_size, 1, item_dim])
    keys_length = np.random.randint(low=1, high=his_length + 1, size=(batch_size, 1))
    keys_length = tf.convert_to_tensor(keys_length)

    print("$$$item:", item.shape)
    print("$$$position:", position.shape)
    print("$$$target:", target.shape)
    print("$$$keys_length:", keys_length.shape)

    print("$$$item:", item)
    print("$$$position:", position)
    print("$$$target:", target)
    print("$$$keys_length:", keys_length)

    # print("==========================")
    # print("item value:", item)
    # print("keys_length value:", keys_length)
    # print("==========================")

    m = dmr([item, position, target, keys_length])
    print("model: ", m)
    print("model shape: ", m.shape)
    print("model aux loss: ", dmr.aux_loss_fn())


if __name__ == "__main__":
    print(tf.__version__)
    debug()





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
from tensorflow.keras.initializers import Zeros


class User2ItemLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, pos_size, vocab_dim, pos_dim, num_samples=5, dropout_rate=0, **kwargs):
        super(User2ItemLayer, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.vocab_dim = vocab_dim
        self.pos_dim = pos_dim
        self.num_sampled = num_samples
        self.concat_att = ConcatAttention()
        self.softmax_weight_sum = SoftmaxWeightedSum(dropout_rate=dropout_rate)

        self.item_embedding = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.vocab_dim)
        self.pos_embedding = tf.keras.layers.Embedding(input_dim=self.pos_size, output_dim=self.pos_dim)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError('A `User2ItemLayer` layer should be called '
                             'on a list of 4 tensors')

        self.size = self.vocab_size
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")

        super(User2ItemLayer, self).build(input_shape)

    def call(self, inputs, aux_mask=True, mask=None, **kwargs):
        """
        :param aux_mask: Auxiliary loss 环节 padding
        :param mask: zero_padding
        :param inputs:
        u2i_layer: ([batch_size,T],[batch_size,T],[batch_size,1],[batch_size,1])
        :param kwargs:
        :return: output: [batch_size,1], Auxiliary loss: ()
        """

        items_id, position_id, target_id, keys_length = inputs
        # print("@@@item:", items_id.shape)
        # print("@@@position:", position_id.shape)
        # print("@@@target:", target_id.shape)
        # print("@@@keys_length:", keys_length.shape)
        #
        # print("@@@item:", items_id)
        # print("@@@position:", position_id)
        # print("@@@target:", target_id)
        # print("@@@keys_length:", keys_length)

        items = self.item_embedding(items_id)  # [batch_size,T,item_embedding]
        position = self.pos_embedding(position_id)  # [batch_size,T,position_embedding]
        target = self.item_embedding(target_id)  # [batch_size,1,item_embedding]

        attention_score = self.concat_att([items, position])  # [batch_size, 1, T]

        hist_len = items.get_shape()[1]  # T

        # tf.sequence_mask([[1],[2],[6]],5)
        # < tf.Tensor: shape = (3, 1, 5), dtype = bool, numpy =
        # array([[[True, False, False, False, False]],
        #        [[True, True, False, False, False]],
        #        [[True, True, True, True, True]]]) >
        key_masks = tf.sequence_mask(keys_length, hist_len)  # [batch_size,1,T]
        # print("key_masks: ", key_masks)

        sum_pooling_before = self.softmax_weight_sum(
            [attention_score, items, key_masks])  # [batch_size,T,item_embedding]
        sum_pooling = tf.reduce_sum(sum_pooling_before, axis=1, keepdims=True)  # [batch_size,1,item_embedding]
        sum_pooling = tf.keras.layers.PReLU()(sum_pooling)  # [batch_size,1,item_embedding]

        assert sum_pooling.get_shape().as_list()[-1] == target.get_shape().as_list()[-1]

        target = tf.transpose(target, perm=[0, 2, 1])  # [[batch_size,target_embedding,1]]
        output = tf.matmul(sum_pooling, target)  # [batch_size,1,1]
        output = tf.squeeze(output, axis=1)  # [batch_size,1]

        # Auxiliary loss
        result = items_id[:, -1]    # [batch_size,]

        if aux_mask:
            # 0的0，非0的为1
            pad_id = 0
            padding = tf.cast(tf.logical_not(tf.equal(items_id, pad_id)), dtype=tf.float32)  # [batch_size,T]
            padding = padding[..., tf.newaxis]                  # [batch_size,T,1]
            sum_pooling_before = sum_pooling_before * padding   # [batch_size,T,item_embedding]

            result = get_diff_col(items_id, keys_length)        # [batch_size,]

        ut_1 = sum_pooling_before[:, :-1, :]  # [batch_size,t-1,item_dim]
        ut_1 = tf.reduce_sum(ut_1, axis=1)  # [batch_size,item_dim]

        label_idx = result
        label_idx = label_idx[:, tf.newaxis]  # [batch_size,1]

        # print("label_idx: ", label_idx.shape)
        # print("label_idx: ", label_idx)

        item_index = EmbeddingIndex(list(range(self.vocab_size)))(0)  # [vocab_size,]
        # print("item_index: ", item_index)
        item_emb = self.item_embedding(item_index)  # [vocab_size, dim]

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=item_emb,
                                                         biases=self.zero_bias,
                                                         labels=label_idx,
                                                         inputs=ut_1,
                                                         num_sampled=self.num_sampled,
                                                         num_classes=self.size,  # item的词典大小
                                                         ))

        return output, loss


class EmbeddingIndex(tf.keras.layers.Layer):

    def __init__(self, index, **kwargs):
        self.index = index
        super(EmbeddingIndex, self).__init__(**kwargs)

    def build(self, input_shape):
        super(EmbeddingIndex, self).build(
            input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        return tf.constant(self.index)

    def get_config(self, ):
        config = {'index': self.index, }
        base_config = super(EmbeddingIndex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def debug():
    batch_size = 32
    vocab_size = 100
    pos_size = 80
    his_length = 16
    item_dim = 12
    pos_dim = 6
    num_sample = 60

    model = User2ItemLayer(vocab_size, pos_size, item_dim, pos_dim, num_sample)

    item = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    position = np.random.randint(low=0, high=his_length, size=(batch_size, his_length))
    target = np.random.randint(low=0, high=his_length, size=(batch_size, 1))

    # items = tf.random.normal([batch_size, his_length, item_dim])
    # position = tf.random.normal([batch_size, his_length, pos_dim])
    # target = tf.random.normal([batch_size, 1, item_dim])
    keys_length = np.random.randint(low=1, high=his_length + 1, size=(batch_size, 1))
    keys_length = tf.convert_to_tensor(keys_length)

    print("item:", item.shape)
    print("position:", position.shape)
    print("target:", target.shape)
    print("keys_length:", keys_length.shape)  # (32, 1)

    # print("==========================")
    # print("item value:", item)
    # print("keys_length value:", keys_length)
    # print("==========================")

    m, loss = model([item, position, target, keys_length])
    print("model: ", m)
    print("model shape: ", m.shape)
    print("loss: ", float(loss))

    # print("**************")
    # x = list(range(100))  # [0-99]
    # print("x: ", x)
    #
    # ei = EmbeddingIndex(x)
    # print("ei: ", ei)
    # y = ei(0)
    # print("y: ", y)


# [batch_size, T],[batch_size,1]
def get_diff_col(items_id, keys_length):
    r_size = items_id.get_shape().as_list()[0]         # batch_size
    print("@@@keys_length: ", keys_length)
    print("@@@keys_length shape: ", keys_length.shape)
    h_index = keys_length.numpy().reshape(-1, 1) - 1   # [batch_size, 1]
    # print("h_index: ", h_index)
    line = np.arange(r_size).reshape(-1, 1)            # [batch_size, 1]
    index = np.hstack((line, h_index))                 # [batch_size, 2]
    result = tf.gather_nd(items_id, index)
    # print("result: ", result)
    # print("result shape: ", result.shape)
    return result


def get_diff_col_test():
    g = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8], [9, 8, 7, 6, 5, 4, 3, 2]])
    print("g: ", g)
    print("g.shape: ", g.shape)                  # (2, 8)
    g_np = np.squeeze(g.numpy())                 # (2, 8)
    print("g_np.shape: ", g_np.shape)
    h_index = np.array([2, 7]).reshape(-1, 1) - 1
    print("h_index: ", h_index)
    print("h_index.shape: ", h_index.shape)      # (2, 1)
    line = np.arange(2).reshape(-1, 1)
    print("line: ", line)
    print("line.shape: ", line.shape)            # (2, 1)
    index = np.hstack((line, h_index))
    print("index: ", index)                      # [[0 2] [1 7]]
    print("index.shape: ", index.shape)          # (2, 2)
    result = tf.gather_nd(g, index)
    print("result: ", result)                    # tf.Tensor([3 2], shape=(2,), dtype=int32)
    print("result.shape: ", result.shape)        # (2,)


if __name__ == "__main__":
    print(tf.__version__)
    # get_diff_col_test()  # 从每行取指定列的元素

    # itemx = tf.constant([[1, 2, 4, 0, 0, 0], [2, 3, 5, 6, 1, 0],
    #                     [9, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6]])
    # keys_len = tf.constant([[3], [5], [1], [1], [6]])
    # get_diff_col(itemx, keys_len)

    debug()

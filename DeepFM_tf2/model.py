"""
Created on July 31, 2020
Updated on May 18, 2021

model: DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dropout, Dense, Input, Layer

from modules import *


class DeepFM(Model):
    def __init__(self, feature_columns, hidden_units=(200, 200, 200), dnn_dropout=0.,
                 activation='relu', fm_w_reg=1e-6, embed_reg=1e-6):
        """
        DeepFM
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. A list of dnn hidden units.
        :param dnn_dropout: A scalar. Dropout of dnn.
        :param activation: A string. Activation function of dnn.
        :param fm_w_reg: A scalar. The regularizer of w in fm.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DeepFM, self).__init__()
        self.feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_normal',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.feature_columns)
        }

        self.index_mapping = []
        self.feature_length = 0
        for feat in self.feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.embed_dim = self.feature_columns[0]['embed_dim']  # all sparse features have the same embed_dim
        self.fm = FM(self.feature_length, len(feature_columns), fm_w_reg)
        self.dnn = DNN(hidden_units, activation, dnn_dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        inputs_index = inputs[0]
        inputs_value = inputs[1]
        # embedding
        sparse_embed = tf.concat([self.embed_layers['embed_{}'.format(i)](inputs_index[:, i])
                                  for i in range(inputs_index.shape[1])], axis=-1)  # (batch_size, embed_dim * fields)

        second_inputs = tf.reshape(sparse_embed,
                   shape=(-1, inputs_index.shape[1], self.embed_dim)) # (batch_size, fields, embed_dim)

        second_inputs = tf.multiply(
            second_inputs, tf.reshape(inputs_value,(-1, len(self.feature_columns), 1)))

        # wide
        inputs_index = inputs_index + tf.convert_to_tensor(self.index_mapping)
        wide_inputs = {'data_inputs': [inputs_index, inputs_value],
                       'embed_inputs': second_inputs}
        wide_outputs = self.fm(wide_inputs)  # (batch_size, 1)
        # deep
        deep_outputs = self.dnn(sparse_embed)
        deep_outputs = self.dense(deep_outputs)  # (batch_size, 1)
        # outputs
        outputs = tf.nn.sigmoid(tf.add(wide_outputs, deep_outputs))
        return outputs

    def summary(self, **kwargs):
        inputs_index = Input(shape=(len(self.feature_columns),),
                             dtype=tf.int32)
        inputs_values = Input(shape=(len(self.feature_columns),),
                              dtype=tf.float32)
        Model(inputs=([inputs_index, inputs_values]),
              outputs=self.call([inputs_index, inputs_values])).summary()

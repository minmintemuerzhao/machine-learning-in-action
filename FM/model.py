"""
Created on August 25, 2020
Updated on May, 18, 2021

model: Factorization Machines

@author: Ziyao Geng(zggzy1996@163.com)
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.regularizers import l2


class FM_Layer(Layer):
    def __init__(self, features_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
        :param v_reg: the regularization coefficient of parameter v
        """
        super(FM_Layer, self).__init__()
        self.feature_columns = features_columns
        self.index_mapping = []
        self.feature_length = 0
        for feat in self.feature_columns:
            self.index_mapping.append(self.feature_length)
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        # mapping
        inputs_index = inputs[0]
        inputs_value = inputs[1]
        inputs_index = inputs_index + tf.convert_to_tensor(self.index_mapping)
        # first order
        inputs_value = tf.reshape(inputs_value,
                                  (-1, len(self.feature_columns), 1))
        one = tf.nn.embedding_lookup(self.w, inputs_index)
        first_order = tf.multiply(one, inputs_value)
        first_order = self.w0 + tf.reduce_sum(first_order,
                                              axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V,
                                               inputs_index)  # (batch_size,
        # fields, embed_dim)

        second_inputs = tf.multiply(second_inputs, inputs_value)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1,
                                             keepdims=True))  # (batch_size,
        # 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1,
                                   keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square,
                                           axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FM(Model):
    def __init__(self, features_columns, k, w_reg=1e-6, v_reg=1e-6):
        """
        Factorization Machines
        :param feature_columns: A list. sparse column feature information.
        :param k: the latent vector
        :param w_reg: the regularization coefficient of parameter w
		:param v_reg: the regularization coefficient of parameter v
        """
        super(FM, self).__init__()
        self.feature_columns = features_columns
        self.fm = FM_Layer(features_columns, k, w_reg, v_reg)

    def call(self, inputs, **kwargs):
        fm_outputs = self.fm(inputs)
        outputs = tf.nn.sigmoid(fm_outputs)
        return outputs

    def summary(self, **kwargs):
        inputs_index = Input(shape=(len(self.feature_columns),),
                             dtype=tf.int32)
        inputs_values = Input(shape=(len(self.feature_columns),),
                              dtype=tf.float32)
        Model(inputs=([inputs_index, inputs_values]),
              outputs=self.call([inputs_index, inputs_values])).summary()

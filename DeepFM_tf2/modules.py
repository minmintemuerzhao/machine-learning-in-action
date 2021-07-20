"""
Created on April 25, 2021

modules of DeepFM: FM, DNN

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout, Dense, Layer


class FM(Layer):
    """
    Wide part
    """
    def __init__(self, feature_length, len_feature_columns, w_reg=1e-6):
        """
        Factorization Machine
        In DeepFM, only the first order feature and second order feature intersect are included.
        :param feature_length: A scalar. The length of features.
        :param w_reg: A scalar. The regularization coefficient of parameter w.
        """
        super(FM, self).__init__()
        self.feature_length = feature_length
        self.len_feature_columns = len_feature_columns
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        inputs_index, inputs_value = inputs['data_inputs']
        embed_inputs = inputs['embed_inputs']
        # first order
        # first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs_index), axis=1)  # (batch_size, 1)

        inputs_value = tf.reshape(inputs_value,
                                  (-1, self.len_feature_columns, 1))
        first_or = tf.nn.embedding_lookup(self.w, inputs_index)
        first_order = tf.multiply(first_or, inputs_value)
        first_order = tf.reduce_sum(first_order, axis=1)  # (batch_size, 1)

        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class DNN(Layer):
    """
    Deep part
    """
    def __init__(self, hidden_units, activation='relu', dnn_dropout=0.):
        """
        DNN part
        :param hidden_units: A list like `[unit1, unit2,...,]`. List of hidden layer units's numbers
        :param activation: A string. Activation function.
        :param dnn_dropout: A scalar. dropout number.
        """
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x
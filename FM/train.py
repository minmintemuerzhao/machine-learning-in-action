"""
Created on August 25, 2020

train FM model

@author: Ziyao Geng(zggzy1996@163.com)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from sklearn.preprocessing import MinMaxScaler

from model import FM
from FM.criteo import create_criteo_dataset

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    # =============================== GPU ==============================
    # gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
    # print(gpu)
    # If you have GPU, and the value is GPU serial number.
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # ========================= Hyper Parameters =======================
    # you can modify your file path
    file_train = 'data/train.csv'
    read_part = True
    sample_num = 5000000
    test_size = 0.2

    k = 8

    learning_rate = 0.001
    batch_size = 4096
    epochs = 10
    sparse_features = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
                       'C10', 'C11',
                       'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
                       'C20', 'C21', 'C22',
                       'C23', 'C24', 'C25', 'C26']
    dense_features = ['I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9',
                      'I10', 'I11',
                      'I12', 'I13']
    # ========================== Create dataset =======================
    features_columns, train, test = \
        create_criteo_dataset(
            file_train=file_train,
            sparse_features=sparse_features,
            dense_features=dense_features,
            read_part=read_part,
            sample_num=sample_num,
            test_size=test_size)
    train_X, train_y = train
    test_X, test_y = test
    # ============================Build Model==========================
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = FM(features_columns=features_columns, k=k)
        model.summary()
        # ============================Compile============================
        model.compile(loss=binary_crossentropy,
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=[AUC()])
    # ============================model checkpoint======================
    # check_path = '../save/fm_weights.epoch_{epoch:04d}.val_loss_{
    # val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path,
    # save_weights_only=True,
    #                                                 verbose=1, period=5)
    # ==============================Fit==============================

    # 连续和离散的分别处理，一起送到网络中去
    # 数据分为index和value. index数据用于标识特征是embedding中的那个
    # value用于标识权重。比如离散特征的权重都是1，连续的权重就是数值本身.
    train_X = train_X[dense_features + sparse_features]
    train_X_index = pd.DataFrame(
        np.zeros((len(train_X), len(train_X.columns))),
        columns=train_X.columns)

    train_X_index.iloc[:, len(dense_features):] = train_X.iloc[:,
                                                  len(dense_features):].values

    train_X_values = pd.DataFrame(
        np.ones((len(train_X), len(train_X.columns))),
        columns=train_X.columns)
    print('归一化结束')
    test_X = test_X[dense_features + sparse_features]
    test_X_index = pd.DataFrame(np.zeros((len(test_X), len(train_X.columns))),
                                columns=train_X.columns)

    test_X_index.iloc[:, len(dense_features):] = test_X.iloc[:,
                                                 len(dense_features):].values

    test_X_values = pd.DataFrame(np.ones((len(test_X), len(train_X.columns))),
                                 columns=train_X.columns)

    scaler = MinMaxScaler()

    for col in dense_features:
        train_X_values[col] = scaler.fit_transform(
            train_X[col].values.reshape(-1, 1))
        test_X_values[col] = scaler.transform(
            test_X_values[col].values.reshape(-1, 1))

    train_X_index.iloc[:, :] = train_X_index.values.astype('int32')
    train_X_values.iloc[:, :] = train_X_values.values.astype('float32')

    test_X_index.iloc[:, :] = test_X_index.values.astype('int32')
    test_X_values.iloc[:, :] = test_X_values.values.astype('float32')
    model.fit(
        (train_X_index, train_X_values),
        train_y,
        epochs=epochs,
        callbacks=[EarlyStopping(monitor='val_loss', patience=2,
                                 restore_best_weights=True)],  # checkpoint
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print(
        'test AUC: %f' % model.evaluate((test_X_index, test_X_values), test_y,
                                        batch_size=batch_size)[1])

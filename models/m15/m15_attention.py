import os

#os.environ['CUDA_VISIBLE_DEVICES']='-1'

import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.layers import Dropout, BatchNormalization, Activation, Input, dot, Reshape, UpSampling2D
from keras.layers import Lambda
from models.base_model import BaseModel
import tensorflow as tf
from keras.models import Model as m
import numpy as np


def attention(x):
    q, v, k = x
    print(q.shape, v.shape, k.shape)
    s = dot([q, k], axes=2, normalize=False)
    s = K.softmax(s, axis=2)
    print(s.shape)
    s1 = dot([s, v], axes=(2,1))
    print(s1.shape)
    return s1

def output_of_attention(input_shape):
    assert isinstance(input_shape, list)
    shape_a, shape_b, shape_c = input_shape
    return shape_a

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'm15'

        inputs = Input(shape=(256, 256, 1))

        # Load earlier trained model which will decide attention region
#        att_model = keras.models.Sequential()
#        att_model = load_model('../experiments/256x256/m15/m15.h5py')

#        att_model.pop()     # dense
#        att_model.pop()     # drop out
#        att_model.pop()     # flatten

#        att_keys = att_model(inputs)
#        att_model.trainable = False

        # 256x256
        conv_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        conv_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv_1)
        maxpool_1 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_2)

        # 128 x 128
        conv_3 = Conv2D(128, (3, 3), padding='same', activation='relu')(maxpool_1)
        conv_4 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv_3)
        maxpool_2 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_4)

        # 64 x 64
        conv_5 = Conv2D(256, (3, 3), padding='same', activation='relu')(maxpool_2)
        conv_6 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv_5)
        maxpool_3 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_6)

        # 32 x 32
        conv_7 = Conv2D(512, (3, 3), padding='same', activation='relu')(maxpool_3)
        conv_8 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv_7)
        maxpool_4 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_8)

        # 16 x 16
        conv_9 = Conv2D(1024, (3, 3), padding='same', activation='relu')(maxpool_4)
        conv_10 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv_9)
        maxpool_5 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_10)

        # 8x8
        conv_11 = Conv2D(2048, (3, 3), padding='same', activation='relu')(maxpool_5)
        conv_12 = Conv2D(2048, (3, 3), padding='same', activation='relu')(conv_11)
        maxpool_6 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_12)

        # 4x4
        conv_13 = Conv2D(4096, (3, 3), padding='same', activation='relu')(maxpool_6)
        conv_14 = Conv2D(4096, (3, 3), padding='same', activation='relu')(conv_13)
        maxpool_7 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_14)

        # apply Attention to layer 10
        layer_to_att = conv_10
        query = Reshape((int(layer_to_att.shape[1] * layer_to_att.shape[2]), int(layer_to_att.shape[3])))(layer_to_att)
        conversion_factor = int(int(layer_to_att.shape[1]) / int(conv_14.shape[1]))
        conv_14_us = UpSampling2D(size=(conversion_factor, conversion_factor))(conv_14)
        conv_14_rs = Conv2D(int(layer_to_att.shape[3]), (1, 1), padding='same', activation='relu')(conv_14_us)
        key = Reshape((int(conv_14_rs.shape[1] * conv_14_rs.shape[2]), int(conv_14_rs.shape[3])))(conv_14_rs)
        attention_out = Lambda(attention, output_shape=output_of_attention)([query, query, key])

        # final layer
        flatten1 = Flatten()(attention_out)
        flatten2 = Flatten()(maxpool_7)
        flatten = keras.layers.concatenate([flatten1, flatten2])

        # Adding Dropout after flatten in user intends to
        if params is not None and 'dropout' in params:
            dropout = Dropout(params['dropout'])(flatten)
            dense = Dense(1, activation=tf.nn.sigmoid)(dropout)
        else:
            dense = Dense(1, activation=tf.nn.sigmoid)(flatten)

        # Model
        self.model = m(inputs=inputs, outputs=dense)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
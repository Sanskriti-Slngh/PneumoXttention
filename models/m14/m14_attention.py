import os
import keras
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Reshape, UpSampling2D, Lambda, dot
from keras.models import Model as m
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from models.base_model import BaseModel
import tensorflow as tf


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
        self.name = 'm14_attention'
        inputs = Input(shape=(1024,1024,1))

        # 256 x 256
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

        # apply Attention to layer 10
        conv_13_rs = Reshape((int(conv_13.shape[1] * conv_13.shape[2]), int(conv_13.shape[3])))(conv_13)
        conversion_factor = int(int(conv_13.shape[1])/int(conv_14.shape[1]))
        conv_14_us = UpSampling2D(size=(conversion_factor,conversion_factor))(conv_14)
        conv_14_rs = Conv2D(int(conv_13.shape[3]), (1,1), padding='same', activation='relu')(conv_14_us)
        key = Reshape((int(conv_14_rs.shape[1]*conv_14_rs.shape[2]), int(conv_14_rs.shape[3])))(conv_14_rs)
        attention_out = Lambda(attention, output_shape=output_of_attention)([conv_13_rs, conv_13_rs, key])

        # combine attention + final output
        # varies
        flatten = Flatten()(attention_out)

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
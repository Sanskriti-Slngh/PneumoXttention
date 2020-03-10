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
    s = K.softmax(s, axis=1)
    print(s.shape)
    s1 = dot([s, v], axes=1)
    print(s1.shape)
    return s1

def output_of_attention(input_shape):
    assert isinstance(input_shape, list)
    shape_a, shape_b, shape_c = input_shape
    return shape_a

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'm14_attention_multiclass'
        inputs = Input(shape=(256,256,1))

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

        #8x8
        conv_11 = Conv2D(2048, (3, 3), padding='same', activation='relu')(maxpool_5)
        conv_12 = Conv2D(2048, (3, 3), padding='same', activation='relu')(conv_11)
        maxpool_6 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_12)

        # 1 x 1 convolution (to reduce number of parameters)
        # conv_11 = Conv2D(params['channels_1x1'], (1, 1), padding='same', activation='relu')(maxpool_6)

        # 4 x 4
        conv_13 = Conv2D(4096, (3, 3), padding='same', activation='relu')(maxpool_6)
        conv_14 = Conv2D(4096, (3, 3), padding='same', activation='relu')(conv_13)
        # maxpool_4 = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(conv_13)

        # 1 x 1 convolution (to reduce number of parameters)
        #conv_14 = Conv2D(8, (1, 1), padding='same', activation='relu')(maxpool_4)

        # apply Attention to layer 10
        conv_10_rs = Reshape((int(conv_10.shape[1] * conv_10.shape[2]), int(conv_10.shape[3])))(conv_10)
        conversion_factor = int(int(conv_10.shape[1])/int(conv_14.shape[1]))
        conv_14_us = UpSampling2D(size=(conversion_factor,conversion_factor))(conv_14)
        conv_14_rs = Conv2D(int(conv_10.shape[3]), (1,1), padding='same', activation='relu')(conv_14_us)
        key = Reshape((int(conv_14_rs.shape[1]*conv_14_rs.shape[2]), int(conv_14_rs.shape[3])))(conv_14_rs)
        attention_out = Lambda(attention, output_shape=output_of_attention)([conv_10_rs, conv_10_rs, key])

        # apply attention to layer 12
        conv_12_rs = Reshape((int(conv_12.shape[1] * conv_12.shape[2]), int(conv_12.shape[3])))(conv_12)
        conversion_factor_2 = int(int(conv_12.shape[1]) / int(conv_14.shape[1]))
        conv_14_us_2 = UpSampling2D(size=(conversion_factor_2, conversion_factor_2))(conv_14)
        conv_14_rs_2 = Conv2D(int(conv_12.shape[3]), (1, 1), padding='same', activation='relu')(conv_14_us_2)
        key_2 = Reshape((int(conv_14_rs_2.shape[1] * conv_14_rs_2.shape[2]), int(conv_14_rs_2.shape[3])))(conv_14_rs_2)
        attention_out_2 = Lambda(attention, output_shape=output_of_attention)([conv_12_rs, conv_12_rs, key_2])

        # combine attention + final output
        # varies
        flatten0 = Flatten()(attention_out)
        flatten1 = Flatten()(attention_out_2)
        flatten = keras.layers.concatenate([flatten0,flatten1])

        # Adding Dropout after flatten in user intends to
        if params is not None and 'dropout' in params:
            dropout = Dropout(params['dropout'])(flatten)
            dense = Dense(3, activation=tf.nn.sigmoid)(dropout)
        else:
            dense = Dense(3, activation=tf.nn.sigmoid)(flatten)

        # Model
        self.model = m(inputs=inputs, outputs=dense)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
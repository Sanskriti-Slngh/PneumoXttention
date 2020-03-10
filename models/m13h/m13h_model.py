import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, Input
from models.base_model import BaseModel
import tensorflow as tf
from keras.models import Model as m


class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'm13h'

        inputs = Input(shape=(256, 256, 1))
        inputs_aux = Input(shape=(17,17,1))

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

        # 1 x 1 convolution (to reduce number of parameters)
        conv_13 = Conv2D(params['channels_1x1'], (1, 1), padding='same', activation='relu')(maxpool_6)

        flatten1 = Flatten()(conv_13)
        flatten2 = Flatten()(inputs_aux)
        flatten = keras.layers.concatenate([flatten1, flatten2])

        # Adding Dropout after flatten in user intends to
        if params is not None and 'dropout' in params:
            dropout = Dropout(params['dropout'])(flatten)
            dense = Dense(1, activation=tf.nn.sigmoid)(dropout)
        else:
            dense = Dense(1, activation=tf.nn.sigmoid)(flatten)

        # Model
        self.model = m(inputs=[inputs, inputs_aux], outputs=dense)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
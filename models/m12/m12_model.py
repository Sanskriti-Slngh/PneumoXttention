import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from models.base_model import BaseModel
import tensorflow as tf

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'm12'
        self.model = keras.models.Sequential()
        # 256 x 256
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256,256,1)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 128 x 128
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 64 x 64
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 32 x 32
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 16 x 16
        self.model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(1024, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 8 x 8
        self.model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(2048, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(3, 3), padding='same', strides=2))

        # 4 x 4
        self.model.add(Flatten())

        # Adding Dropout after flatten in user intends to
        if params is not None:
            if 'dropout' in params:
                self.model.add(Dropout(params['dropout']))

        self.model.add(Dense(1, activation=tf.nn.sigmoid))

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
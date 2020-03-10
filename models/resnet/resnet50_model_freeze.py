import os
import keras
from keras import applications
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from models.base_model import BaseModel
import tensorflow as tf
from keras.models import Model as m

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'resnet50'
        base_model = applications.resnet50.ResNet50(weights=None, input_shape=(256,256,1))
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(1, activation=tf.nn.sigmoid))

        for layer in base_model.layers:
            layer.trainable = False
        model.load_weights('resnet50.h5py')

        for layer in base_model.layers[-26:]:
            layer.trainable = True

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
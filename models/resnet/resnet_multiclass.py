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

        x = base_model.output

        predictions = Dense(3, activation=tf.nn.softmax)(x)
        self.model = m(inputs = base_model.input, outputs = predictions)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)

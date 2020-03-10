import keras
from keras.layers import Dense
from models.base_model import BaseModel
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Concatenate, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.models import Model as m

class Model(BaseModel):
    def __init__(self, history, params=None):
        self.name = 'vgg16'

        inputs = Input(shape=(224, 224, 1))
        inputs = Concatenate()([inputs, inputs, inputs])

        vgg_model = VGG16(input_tensor=inputs, include_top = False, weights='imagenet')
        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block3_pool'].output
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(x)

        flatten = Flatten()(x)

        # Adding Dropout after flatten in user intends to
        if params is not None and 'dropout' in params:
            dropout = Dropout(params['dropout'])(flatten)
            dense = Dense(1, activation=tf.nn.sigmoid)(dropout)
        else:
            dense = Dense(1, activation=tf.nn.sigmoid)(flatten)

        # Model
        self.model = m(inputs=vgg_model.input, outputs=dense)

        # save history
        self.history = history

        # run base model __init__
        super(Model, self).__init__(params)
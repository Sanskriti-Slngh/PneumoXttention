import pickle
import numpy as np
from keras.backend import binary_crossentropy, mean
from keras.models import load_model
import keras
from statistics import mode
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import random

# fixing randomization for repeat

# user options
use_adam = True
learning_rates = [0.000001]
decay_rate = 0.8
decay_epochs = 50
momentum = 0.9
batch_sizes = [16]
channels_1x1 = [128]
epochs = 300
model_name = 'm15'
data_generation_enable = True
margin_values = [0.7]

train_data = '../data/binaryclass/train_data_256'
val_data = '../data/binaryclass/val_data_256_no_hv'

# import models
import models.m12.m12_model as m12
import models.m13.m13_model as m13
import models.m14.m14_model as m14
import models.m15.m15_2mod_attn as m15_2mod_attn


# define history class
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if epoch%10 == 0 and epoch:
            # Save model
            model.save('../experiments/current/m_change_' + str(epoch))

# learning rate scheduler
def lr_scheduler(epoch, lr):
    global decay_rate, decay_epochs
    if epoch%decay_epochs == 0 and epoch:
        return lr * decay_rate
    return lr

history = LossHistory()


# instantiate model
params = {'resetHistory': False,
          'models_dir': '../experiments/data_gen/',
          'dropout': 0.5,
          'print_summary': True,
          'channels_1x1': 1}

for batch_size in batch_sizes:
    for lr in learning_rates:
        for channels in channels_1x1:
            for margin_value in margin_values:
                history = LossHistory()
                params['channels_1x1'] = channels
                params['models_dir'] = '../experiments/data_gen/channels_1x1/width_height_change_redo/' + str(batch_size) + '/' + str(lr) + '/' + str(channels) + '/m14_d50'
                if model_name == 'm12':
                    model = m12.Model(history, params)

                elif model_name == 'm13':
                    model = m13.Model(history, params)

                elif model_name == 'm14':
                    model = m14.Model(history, params)

                elif model_name == 'm15':
                    model = m15_2mod_attn.Model(history, params)

                # optimizer
                if use_adam:
                    optimizer = keras.optimizers.Adam(lr=lr)
                else:
                    optimizer = keras.optimizers.SGD(lr=lr, momentum=momentum)

                model.compile(optimizer, multi_classification=False)

                # get the data
                with open(val_data, 'rb') as fin:
                    x_val, y_val = pickle.load(fin)

                # change of False Positives real value to 1
                model.change_fn(train_data, 16, 100, margin_value)

                # training
                model.train(model.x_train, model.y_train, x_val, y_val, batch_size, epochs, lr_scheduler, data_generation_enable)

                ## save the model
                model.save('../experiments/fp_change/100/margins/' + str(margin_value) + '/' + str(channels) + '/m14')
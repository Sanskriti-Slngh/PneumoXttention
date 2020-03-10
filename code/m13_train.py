import os
import tensorflow as tf
import keras
import pickle
import matplotlib.pyplot as plt

# import models
import models.m12.m12_model as m12
import models.m13.m13_model as m13
import models.m14.m14_model as m14

# user options
model_name = 'm13'
use_adam = False
learning_rates = [0.001]
decay_rate = 0
decay_epochs = 5
momentum = 0.9
batch_sizes = [16]
epochs = 50
minimize_false_negative = False
plot_only = False
channels_1x1 = [128]
predict = False
data_aug = True
data_gen = False

if data_aug:
    train_data = '../data/binaryclass/train_data_256'
    val_data = '../data/binaryclass/val_data_256'
else:
    train_data = '../data/train_data_256_no_hv'
    val_data = '../data/val_data_256_no_hv'
data = '../data/test_data_256'

# define history class
class LossHistory(keras.callbacks.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        self.acc_epochs = 0
        super(LossHistory, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if epoch%10 == 0 and epoch:
            # Save model
            model.save('../experiments/current/m_' + str(epoch))

# learning rate scheduler
def lr_scheduler(epoch, lr):
    global decay_rate, decay_epochs
    if epoch%decay_epochs == 0 and epoch and decay_rate != 0:
        return lr * decay_rate
    return lr

#history = LossHistory()

# instantiate model
params = {'resetHistory': False,
          'models_dir': '../experiments/channels_1x1/',
          'dropout': 0.5,
          'print_summary': True,
          'channels_1x1': 1,
          'add_class_weight':False}

if not predict:
    fig, ax = plt.subplots(nrows=1, ncols=2)

for batch_size in batch_sizes:
    for lr in learning_rates:
        for channels in channels_1x1:
            history = LossHistory()
            params['channels_1x1'] = channels
            params['models_dir'] = '../experiments/256x256/channels_1x1'
            if model_name == 'm12':
                model = m12.Model(history, params)

            elif model_name == 'm13':
                model = m13.Model(history, params)

            elif model_name == 'm14':
                model = m14.Model(history, params)

            if plot_only:
                model.is_train = False

            if not plot_only:
                ## get the data
                with open(train_data, 'rb') as fin:
                    x_train, y_train = pickle.load(fin)

                with open(val_data, 'rb') as fin:
                    x_val, y_val = pickle.load(fin)

            ## training
            if use_adam:
                optimizer = keras.optimizers.Adam(lr=lr)
            else:
                optimizer = keras.optimizers.SGD(lr=lr, momentum=momentum)

            model.compile(optimizer, multi_classification=False)

            # soon to be done false negative code
            if not plot_only:
                model.train(x_train, y_train, x_val, y_val, batch_size, epochs, lr_scheduler, data_gen)

                ## save the model
                model.save()

            if not predict and plot_only:
                # plot the results
                model.train_plot(fig, ax, show_plot=False, label=str(channels) + ' ' + str(lr))

            if predict:
                model.predict(data)
                #model.plot_predict_image(image_wanted='mistakes')
                #model.plot_predict_image()

if not predict:
    plt.legend()
    plt.show()

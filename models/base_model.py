# Libraries
import os
import keras
import pickle
from matplotlib.patches import Rectangle
import random
import keras.backend as K

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
from sklearn import metrics
import seaborn as sn

from keras.models import load_model
from keras.utils import plot_model
from keras.preprocessing. image import ImageDataGenerator
import shutil
from shapely.geometry import Polygon
from skimage.transform import resize

from keras.models import Model as m
from matplotlib import pyplot
from keras.layers import Input, Reshape, UpSampling2D, Lambda, dot
import bz2
from PIL import Image

random.seed(1223143)

# base class for all models
# Describe each method (function)
class BaseModel:
    def __init__(self, params=None):
        self.params = {}
        # One line description for each variable
        self.params['resetHistory'] = False
        self.params['models_dir'] = "."
        self.params['print_summary'] = True
        self.params['split_h'] = 256
        self.params['split_w'] = 256
        self.params['split_x_stride'] = 16
        self.params['split_y_stride'] = 16
        self.params['split_random_sample'] = False
        self.params['split_num_samples'] = 1
        self.params['use_heat_map'] = False
        self.params['multi_classification'] = False
        self.params['data_generation_enable'] = False
        self.params['train_with_mask_aug'] = False
        self.params['random_256_generation'] = False
        self.params['y_hat_threshold'] = 0.5
        self.params['y_reshape_to_2d'] = False
        self.params['do_confusion_matrix'] = True
        self.params['force_normalize'] = False

        self.patience = 4
        self.is_train = True
        self.all_plot = False
        self.overlap_threshold = 0.1
        self.count = 0
        self.x_train = None
        self.y_train = None
        self.b_train = None
        self.h_train = None

        self.x_val = None
        self.y_val = None
        self.b_val = None
        self.h_val = None

        self.x_test = None
        self.y_test = None
        self.b_test = None
        self.h_test = None

        # setting parameters based on the params as given by the user
        if params is not None:
            for key, value in params.items():
                self.params[key] = value

        # combining model directory and name of model
        self.name = self.params['models_dir'] + '/' + self.name

        # create model directory if unknown
        if not os.path.isdir(self.params['models_dir']):
            os.makedirs(self.params['models_dir'])

        # Copy the train.py
        if self.is_train:
            shutil.copyfile('train.py', self.params['models_dir'] + '/train.py.copy')

        # load model if model is already there
        if not self.params['resetHistory'] and os.path.isfile(self.name + '.h5py'):
            print("Loading model from " + self.name + '.h5py')
            self.model = load_model(self.name + '.h5py', custom_objects={'dot': dot, 'get_f1': self.get_f1})
            if self.history:
                with open(self.name + '.aux_data', 'rb') as fin:
                    self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc = pickle.load(fin)

        # Check if model is defined
        if not self.model:
            exit("model not defined")

        # print summary if user intends to
        if self.params['print_summary']:
            print(self.model.summary())

    def save(self, name=None):
        self.sfname = self.name
        if name:
            self.sfname = name
        self.model.save(self.sfname + '.h5py')
        with open(self.sfname + '.aux_data', 'wb') as fout:
            pickle.dump((self.history.train_losses, self.history.val_losses, self.history.train_acc, self.history.val_acc), fout)
        if not name:
            plot_model(self.model, to_file=self.name + '.png')

    def get_f1(self, y_true, y_pred):  # taken from old keras source code
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())

        return f1_val

    def calculate_intersection(self, bx_x, bx_y, bx_w, bx_h, x, y, w=256, h=256):
        box1 = [[bx_x, bx_y], [bx_x + bx_w, bx_y], [bx_x + bx_w, bx_y + bx_h], [bx_x, bx_y + bx_h]]
        box2 = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        poly_1 = Polygon(box1)
        poly_2 = Polygon(box2)
        intersection = poly_1.intersection(poly_2).area / poly_1.area
        return intersection

    def get_random_256(self, images, y_values, boxes):
        final_images = []
        targets = []

        # Always check to see if we need to normalize
        if images.dtype == np.uint8:
            norm_const = np.array(255).astype('float16')
            images = images / norm_const

        # Get a random 256x256 sample, and if sample overlaps with any bounding box
        # set the target to 1.
        for idx, image in enumerate(images):
            x = random.randint(1, 512 - 256)
            y = random.randint(1, 512 - 256)
            final_images.append(image[x:x + 256, y:y + 256, 0])

            target = [0]
            if y_values[idx][0]:
                for bb in boxes[idx]:
                    box_x, box_y, width, height = bb[0], bb[1], bb[2], bb[3]
                    intersection = self.calculate_intersection(box_x, box_y, width, height, x, y)
                    if intersection > self.overlap_threshold:
                        target = [1]
            targets.append(target)

        final_images = np.array(final_images)
        final_images = np.reshape(final_images, (final_images.shape[0], 256, 256, 1))
        targets = np.array(targets)
        del images
        return final_images, targets

    def split(self, images, y_values, boxes):
        final_images = []
        targets = []

        # check to see if we need to normalize
        if images.dtype == np.uint8 or self.params['force_normalize']:
            norm_const = np.array(255).astype('float16')
            images = images / norm_const

        a = len(self.y_all)

        # Split into multiple 256x256 images based on x/y strides
        for idx, image in enumerate(images):
            if self.params['split_random_sample']:
                x_samples = 1
                y_samples = 1
            else:
                x_samples = int((image.shape[0]-self.params['split_w']+self.params['split_x_stride'])/self.params['split_x_stride'])
                y_samples = int((image.shape[0] - self.params['split_h'] + self.params['split_y_stride']) / self.params['split_y_stride'])
            self.params['split_num_samples'] = x_samples*y_samples
            self.params['num_x_samples'] = x_samples
            self.params['num_y_samples'] = y_samples
            for j in range(y_samples):
                for i in range(x_samples):
                    if self.params['split_random_sample']:
                        x = random.randint(1, image.shape[0] - self.params['split_w'])
                        y = random.randint(1, image.shape[1] - self.params['split_h'])
                    else:
                        x = i*self.params['split_x_stride']
                        y = j*self.params['split_y_stride']
                    final_images.append(image[x:x+self.params['split_w'], y:y+self.params['split_h'], 0])
                    target = [0]
                    if y_values[idx][0]:
                        for bb in boxes[idx]:
                            box_x, box_y, width, height = bb[0], bb[1], bb[2], bb[3]
                            intersection = self.calculate_intersection(box_x, box_y, width, height, x, y)
                            if intersection > self.overlap_threshold:
                                target = [1]
                    targets.append(target)
                    self.y_all.append(target)
        b = len(self.y_all)
        assert((b-a) == self.params['split_num_samples']*len(images))
        self.count += 1

        final_images = np.array(final_images)
        final_images = np.reshape(final_images, (final_images.shape[0], self.params['split_w'], self.params['split_h'], 1))
        targets = np.array(targets)
        del images
        return final_images, targets

    def compile(self, optimizer):
        if self.params['multi_classification']:
            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', self.get_f1])
        else:
            self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', self.get_f1])

    def train(self, batch_size, epochs, lr_scheduler):
        if (self.params['force_normalize']):
            print("WARNING: Force normalization is on")
        if self.params['data_generation_enable']:
            if self.x_train.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_train = self.x_train / norm_const

            if self.x_val.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_val = self.x_val / norm_const

            self.datagen = ImageDataGenerator(rotation_range = 40, width_shift_range=0.3, height_shift_range=0.3)
            self.model.fit_generator(self.datagen.flow(self.x_train, self.y_train, batch_size=batch_size),
                                     steps_per_epoch=len(self.x_train)/batch_size,
                                     epochs=epochs, validation_data=(self.x_val, self.y_val),
                                     callbacks=[self.history,
                                                keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                                keras.callbacks.EarlyStopping(monitor='val_acc', patience=self.patience)])
            return

        if self.params['use_heat_map']:
            if self.x_train.dtype == np.uint8 or self.params['force_normalize']:
                norm_const = np.array(255).astype('float16')
                self.x_train = self.x_train / norm_const

            if self.x_val.dtype == np.uint8 or self.params['force_normalize']:
                norm_const = np.array(255).astype('float16')
                self.x_val = self.x_val / norm_const

            self.h_val = self.h_val[0:self.x_val.shape[0]]
            self.h_train = self.h_train[0:self.x_train.shape[0]]
            self.model.fit([self.x_train, self.h_train], self.y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=([self.x_val, self.h_val], self.y_val),
                           callbacks = [self.history,
                                        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)])
            return

        # default
        if self.x_train.dtype == np.uint8 or self.params['force_normalize']:
            norm_const = np.array(255).astype('float16')
            self.x_train = self.x_train / norm_const

        if self.x_val.dtype == np.uint8 or self.params['force_normalize']:
            norm_const = np.array(255).astype('float16')
            self.x_val = self.x_val / norm_const

        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(self.x_val, self.y_val),
                           callbacks = [self.history,
                                        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)])


    def train_plot(self, fig=None, ax=None, show_plot=True, label=None):
        if not fig:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(self.history.train_losses[self.history.acc_epochs:], label=label + ' train', color='red')
        ax[0].plot(self.history.val_losses[self.history.acc_epochs:], label=label +' val', color='blue')
        ax[0].set_ylabel('Loss')
        ax[0].set_xlabel('epocs')
        ax[0].set_title("Loss vs epocs, train(Red)")

        ax[1].plot(self.history.train_acc[self.history.acc_epochs:], label=label + ' train', color='red')
        ax[1].plot(self.history.val_acc[self.history.acc_epochs:], label=label + ' val', color='blue')
        ax[1].set_ylabel('Accuracy')
        ax[1].set_xlabel('epocs')
        ax[1].set_title("Accuracy vs epocs, train(Red)")

        print('train_loss: ' + str(self.history.train_losses[-5:-1]))
        print('val_loss: ' + str(self.history.val_losses[-5:-1]))
        print('train_acc: ' + str(self.history.train_acc[-5:-1]))
        print('val_acc: ' + str(self.history.val_acc[-5:-1]))
        print('epochs:   ' + str(len(self.history.train_losses)))

        if show_plot:
            plt.show()

    def predict(self, x, batchsize):
        return self.model.predict(x, batchsize)

    def prediction(self, batch_size = 4, plot_options = None):
        if (self.params['force_normalize']):
            print("WARNING: Force normalization is on")

        if self.params['random_256_generation']:
            self.y_all = [] # new target for all samples
            def createGenerator(X, I, Y):
                while True:
                    idx = [i for i in range(X.shape[0])]
                    datagen = ImageDataGenerator()
                    batches = datagen.flow(X[idx], Y[idx], batch_size=batch_size, shuffle=False)
                    idx0 = 0
                    for batch in batches:
                        idx1 = idx0 + batch[0].shape[0]
                        # yield self.split(batch[0], batch[1], I[idx[idx0:idx1]])
                        yield self.split(batch[0], batch[1], I[idx[idx0:idx1]])
                        idx0 = idx1
                        if idx1 >= X.shape[0]:
                            break

            self.gen_flow = createGenerator(self.x_test, self.b_test, self.y_test)
            self.y_hat = self.model.predict_generator(self.gen_flow, steps=len(self.x_test)/batch_size, verbose=True)
            self.y_pred = self.y_hat > self.params['y_hat_threshold']
            print('done predicting on input %s with split producing %s' %(self.x_test.shape, self.y_hat.shape))
            # if any sample generates 1, then make predicted value as 1
            if False:
                self.y_pred = np.zeros(self.y.shape)
                self.y_hat_max = np.zeros(self.y.shape)
                self.y = np.array(self.y_all)
                p = self.y_hat > 0.5
                for i in range(self.y.shape[0]):
                      aaa = self.params['split_num_samples']
                      x = np.sum(p[i*aaa:i*aaa+aaa])
                      maxs = np.max(self.y_hat[i*aaa:i*aaa+aaa])
                      self.y_hat_max[i,0] = maxs
                      if x > 0:
                          self.y_pred[i,0] = True
                      else:
                          self.y_pred[i,0] = False
                print ("After merging the prediction: %s " %(self.y_pred.shape,))
                print (self.y_hat[0:10])
                self.y_hat = self.y_hat_max
            if True:
                if self.params['y_reshape_to_2d']:
                    self.y_pred = np.reshape(self.y_pred,(self.x_test.shape[0],self.params['num_y_samples'],self.params['num_x_samples']))
                    self.y_hat = np.reshape(self.y_hat,(self.x_test.shape[0],self.params['num_y_samples'],self.params['num_x_samples']))
                    print(self.y_pred.shape)
            if not self.params['do_confusion_matrix']:
                return
            self.y = self.y_all[0:self.y_hat.shape[0]]
        else:
            with open('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/final_test_images', "rb") as fin:
                self.x_test, self.y_test = pickle.load(fin)

            with open('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/heatmaps', 'rb') as fin:
                self.h_test = pickle.load(fin)

            self.h_test = np.array(self.h_test)
            self.x_test = self.x_test[0:25, :, :, :]
            self.y_test = self.y_test[0:25]
            self.h_test = self.h_test[0:25, :, :]
            # normalize if needed
            print(self.x_test.shape)
            #print(self.x_test[0,0,0,0])
            if self.x_test.dtype == np.uint8 and self.params['force_normalize']:
                print ("Normalizing x_test")
                norm_const = np.array(255).astype('float16')
                self.x_test = self.x_test / norm_const

            if not self.params['use_heat_map']:
                print("x=%s, y=%s" % (self.x_test.shape, self.y_test.shape))
                self.y_hat = self.model.predict(self.x_test, batch_size=batch_size)
            else:
                print("x=%s, y=%s, heatmap=%s" % (self.x_test.shape, self.y_test.shape, self.h_test.shape))
                self.y_hat = self.model.predict([self.x_test, self.h_test], batch_size = batch_size)
            print('done predicting')
            self.y_pred = self.y_hat > 0.5
            self.y = self.y_test

        # Confusion matrix results of predictions
        cm = confusion_matrix(self.y, self.y_pred)
        print(cm)
        df_cm = pd.DataFrame(cm, index=['True 0', 'True 1'], columns=['Predicted 0', 'Predicted 1'])
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True, fmt='g')
        acc = (cm[0, 0] + cm[1, 1]) / (cm[1, 0] + cm[0, 1] + cm[0, 0] + cm[1, 1])
        print("Accuracy:   " + str(acc))
        rec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        pre = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        f1 = 2 * ((pre * rec) / (pre + rec))
        print('Recall:    ' + str(rec))
        print('Precision:    ' + str(pre))
        print('F1 score:    ' + str(f1))
        fpr, tpr, thresholds = metrics.roc_curve(self.y, self.y_hat, pos_label=0)
        #plt.show()
        
        # Print ROC curve
        if plot_options and plot_options['roc_curve']:
            plt.figure()
            plt.plot(fpr, tpr)
            plt.show()

        # Print AUC
        auc = metrics.auc(tpr, fpr)
        print('AUC:', auc)

        if plot_options and plot_options['fp']:
            plt.figure()
            y_hat_for_plt = []
            id = []
            for i in range(self.y.shape[0]):
                if self.y_pred[i] == 1 and self.y[i] == 0:
                    y_hat_for_plt.append(self.y_hat[i])
                    id.append(i)
            plt.scatter(id, y_hat_for_plt)
            plt.show()

    def gen_fifty(self, data_file, batch_size):
        with open('C:/Users/Manish/projects/tiya/scienceFair-2020/data/multiclass/rsna_test.dat','rb') as fin:
            self.x, self.y = pickle.load(fin)

        print(self.y.shape)
        print(self.y)
        num_positive = 12
        num_false_opac = 8
        num_false = 5
        positives = []
        false_opacs = []
        falses = []
        y_values = []
        go = True
        while go == True:
            index = random.randint(0, 4321)
            #image = self.x[index, :, :, 0]
            if num_positive == 0 and num_false_opac == 0 and num_false == 0:
                go = False
            if self.y[index][2] == 1 and num_positive != 0:
                if index in positives:
                    continue
                positives.append(index)
                num_positive = num_positive-1
            elif self.y[index][0] == 1 and num_false_opac != 0:
                if index in false_opacs:
                    continue
                false_opacs.append(index)
                num_false_opac = num_false_opac-1
            elif self.y[index][1] == 1 and num_false != 0:
                if index in falses:
                    continue
                falses.append(index)
                num_false = num_false-1
            else:
                continue

        number = 1

        positives.extend(false_opacs)
        positives.extend(falses)
        np.random.shuffle(positives)

        images = []
        box = []
        heatmaps = []

        with open('../data/binaryclass/box_data/box_data_256_test', 'rb') as fin:
            self.boxes = pickle.load(fin)

        for index in positives:
            image = self.x[index, :, :, 0]
            images.append(image)
            heatmaps.append(self.h_test[index, :, :])
            if self.y[index][2] == 1:
                y_values.append(1)
            else:
                y_values.append(0)
            box.append(self.boxes[index])
            fig, ax = plt.subplots(1)
            ax.imshow(image, cmap='gray')
            #plt.show()
            fig.savefig('C:/Users/projects/tiya/scienceFair-2020/reports/test_data/' + str(number) + '.png', dpi=fig.dpi)
            number = number +1

        images = np.array(images)
        print(images.shape)
        y_values = np.array(y_values)
        print(box)
        print(len(box))

        for i in y_values:
            print(i)

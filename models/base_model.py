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
        # data placeholders(b - box data, h - heat map)
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

        if self.params['random_256_generation']:
            def createGenerator(X, I, Y):
                while True:
                    # shuffle indices
                    idx = np.random.permutation(X.shape[0])
                    datagen = ImageDataGenerator()
                    batches = datagen.flow(X[idx], Y[idx], batch_size=batch_size, shuffle=False)
                    idx0 = 0
                    for batch in batches:
                        idx1 = idx0 + batch[0].shape[0]
                        yield self.get_random_256(batch[0], batch[1], I[idx[idx0:idx1]])
                        idx0 = idx1
                        if idx1 >= X.shape[0]:
                            break

            # Finally create generator
            self.gen_flow = createGenerator(self.x_train, self.b_train, self.y_train)
            self.gen_flow_val = createGenerator(self.x_val, self.b_val, self.y_val)

            self.model.fit_generator(self.gen_flow,
                                     steps_per_epoch=len(self.x_train) / batch_size,
                                     use_multiprocessing=False,
                                     epochs=epochs, validation_data=self.gen_flow_val,
                                     validation_steps=len(self.x_val)/batch_size,
                                     callbacks=[self.history, keras.callbacks.LearningRateScheduler(lr_scheduler,verbose=1),
                                                keras.callbacks.EarlyStopping(monitor='val_acc',patience=self.patience)])
            return

        if self.params['train_with_mask_aug']:
            if self.x_train.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_train = self.x_train / norm_const

            if self.x_val.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_val = self.x_val / norm_const

            def apply_image_mask(xs,ys,boxs):
                M = 64
                N = 64
                mwidth = int(256/M)
                mheight = int(256/N)
                percentage_dropout = 0
                max_limit_on_mask = 4
                mask_cnt = 0
                for idx in range(xs.shape[0]):
                    for i in range(M):
                        for j in range(N):
                            if ys[idx] == 0:
                                if random.randint(1, 100) < percentage_dropout * 100:
                                    # compute pixel coordinates
                                    px = i * mwidth
                                    py = j * mheight
                                    xs[idx, px:px + mwidth, py:py + mheight, 0] = np.zeros((mwidth, mheight))
                                continue
                            # if we reached max_limit_on_mask don't mask anything
                            if mask_cnt > max_limit_on_mask:
                                continue

                            # Get to rectangle co-ordinates from data
                            x1, y1 = (boxs[idx][0], boxs[idx][1])
                            w, h = (boxs[idx][2], boxs[idx][3])
                            x2, y2 = (x1 + w, y1 + h)

                            # Transforming to grid co-ordinates (MxN)
                            x1, y1 = (int(x1 / mwidth), int(y1 / mheight))
                            x2, y2 = (int((x2 + mwidth - 1) / mwidth), int((y2 + mheight - 1) / mheight))

                            # to see whether the coordinates overlap
                            if i >= x1 and i <= x2 and j >= y1 and j <= y2:
                                continue
                            # if not overlapping
                            else:
                                if random.randint(1, 100) < percentage_dropout * 100:
                                    # compute pixel coordinates
                                    px = i * mwidth
                                    py = j * mheight
                                    xs[idx, px:px+mwidth, py:py+mheight, 0] = np.zeros((mwidth, mheight))
                                    mask_cnt += 1
                return xs
            self.datagen = ImageDataGenerator()
            def gen_flow_with_mask(x_train, boxes, y_train):
                genX1 = self.datagen.flow(x_train, y_train, batch_size=batch_size, seed=666)
                genX2 = self.datagen.flow(x_train, boxes, batch_size=batch_size, seed=666)
                print (len(x_train))
                while True:
                    X1i = genX1.next()
                    X2i = genX2.next()

                    yield apply_image_mask(X1i[0], X1i[1], X2i[1]), X1i[1]
                    #yield X1i[0], X1i[1]

            # Finally create generator
            self.gen_flow = gen_flow_with_mask(self.x_train, self.b_train, self.y_train)
            self.model.fit_generator(self.gen_flow,
                                     steps_per_epoch=len(self.x_train)/batch_size,
                                     epochs=epochs, validation_data=(self.x_val, self.y_val),
                                     callbacks=[self.history,
                                                keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                                keras.callbacks.EarlyStopping(monitor='val_acc', patience=self.patience)])
            return

        if self.params['add_class_weight']:
            if self.x_train.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_train = self.x_train / norm_const

            if self.x_val.dtype == np.uint8:
                norm_const = np.array(255).astype('float16')
                self.x_val = self.x_val / norm_const

            y_train_labels = self.params['y_train_labels']
            class_weight = compute_class_weight('balanced', np.unique(y_train_labels), y_train_labels)
            self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                           validation_data=(self.x_val, self.y_val),
                           class_weight=class_weight,
                           callbacks = [self.history,
                                        keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1),
                                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience)])
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

    def rcnn(self, data_file):
        def draw_image_with_boxes(filename, boxes_list):
            # load the image
            data = filename[0, :, :, 0]
            # plot the image
            pyplot.imshow(data)
            # get the context for drawing boxes
            ax = pyplot.gca()
            # plot each box
            for box in boxes_list:
                # get coordinates
                y1, x1, y2, x2 = box
                # calculate width and height of the box
                width, height = x2 - x1, y2 - y1
                # create the shape
                rect = Rectangle((x1, y1), width, height, fill=False, color='red')
                # draw the box
                ax.add_patch(rect)
            # show the plot
            pyplot.show()

        # draw an image with detected objects
        with open(data_file, 'rb') as fin:
            self.x, self.y = pickle.load(fin)
        index = random.randint(0, 5337)
        image = self.x[index, :, :, :]
        img = np.reshape(image, (1, 256, 256, 1))
        # make prediction
        results = self.model.detect([img], verbose=0)
        # visualize the results
        draw_image_with_boxes(img, results[0]['rois'])

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

#    # over write predict method
    def predict(self, x, batchsize):
        return self.model.predict(x, batchsize)

    # Predicting on test data
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

    def visualize_cnn(self, data_file):
        #with open(data_file, 'rb') as fin:
         #   self.x, self.y = pickle.load(fin)
        with open('../data/binaryclass/box_data/box_data_256_test', 'rb') as fin:
            self.boxes = pickle.load(fin)
        self.x = self.x_test
        self.y = self.y_test
        final = []
        self.y_hat = self.model.predict(self.x, batch_size=32)
        self.y_pred = self.y_hat > 0.5
        for i in range(20):
            while True:
                index = random.randint(0, 5337)
                if self.y[index] == 0 and self.y_pred[index] == 1:
                    break
            print(index)
            image = self.x[index, :, :, :]
            img = np.reshape(image, (1, 256, 256, 1))
            y_hat = self.model.predict(img)
            weights = self.model.get_weights()

            # create new model to evalulate activation of last conv2d layer
            output = self.model.layers[-1].output
            model = keras.models.Model(inputs=self.model.inputs, outputs=output)
            act = model.predict(img)
            w = weights[-2]
            b = self.model.get_weights()[-1]
            print (act.shape)
            print (w.shape)
            y_dash = np.dot(act,w) + b
            y_cal = 1/(1 + np.exp(-y_dash))
            print (y_cal, y_hat)

            # reshape act into 4x4x8
            act = np.reshape(act, (4,4,8))
            print(act)
            # reshape weight into 4x4x8
            weight = np.reshape(w, (4,4,8))
            print(weight)
            # element wise multiply act and weight (output still 4x4x8)
            y = act * weight
            # sum across depth (axis=2) (output will be 4x4)
            f = np.sum(y, axis=2)
            a = np.zeros((256, 256))
            for i in range(256):
                for j in range(256):
                    v = int(f[int(i/64), int(j/64)]*255)
                    a[i,j] = v
            print(a.shape)
            print(a)
            fig, ax = plt.subplots(2)
            ax[0].imshow(image[:,:,0],cmap='gray')
            ax[1].imshow(a, cmap='gray')
            for i in range(4):
                for j in range(4):
                    ax[1].text(i*64, j*64, "%0.2f " %f[j,i], color='red')

            ax[0].text(0, 0, 'y hat value: ' + str(self.model.predict(img)))
            xy = ((self.boxes[index][0][0][0]), (self.boxes[index][0][0][1]))
            rect = patches.Rectangle(xy, (self.boxes[index][0][0][2]), (self.boxes[index][0][0][3]),
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
            plt.show()
        exit()

        self.y_hat = self.model.predict(self.x, batch_size=32)
        self.y_pred = self.y_hat > 0.5
        delta = 0.9
        while True:
            index = random.randint(0, 5337)
            #if index == 4917 or index == 1721 or index == 2563 or index == 1464:
             #   continue
            if self.y[index] == 1 and self.y_pred[index] == 1:
                break
        print(index)
        image = self.x[index, :, :, :]
        img = np.reshape(image, (1, 256, 256, 1))
        y_hat_old = self.model.predict(img)
        fig, ax = plt.subplots(2)
        for i in range(256):
            for j in range(256):
                curr_pixel_value = image[i,j,0]
                image[i,j,0] += delta
                img = np.reshape(image, (1, 256, 256, 1))
                y_hat_new = self.model.predict(img)
                final.append((y_hat_new-y_hat_old)/delta)
                image[i,j,0] = curr_pixel_value
        final = np.array(final)
        #min_value = min(final)
        #max_value = max(final)
        #spread = max_value - min_value
        #if min_value < 0:
        #    final = final + (-min_value)
        #final = final * 255/spread
        final = final.astype(int)
        final = np.reshape(final, (256, 256))
        ax[0].imshow(image[:, :, 0], cmap='gray')
        ax[0].text(0, 0, 'real value: ' + str(self.y[index])+ '   predicted value: ' + str(self.y_pred[index]))
        ax[1].imshow(final, cmap='gray')
        xy = ((self.boxes[index][0][0][0]) * 0.25, (self.boxes[index][0][0][1]) * 0.25)
        ax[0].axis('off')
        rect = patches.Rectangle(xy, (self.boxes[index][0][0][2]) * 0.25, (self.boxes[index][0][0][3]) * 0.25,
                                 linewidth=1, edgecolor='r', facecolor='none')
        rect1 = patches.Rectangle(xy, (self.boxes[index][0][0][2]) * 0.25, (self.boxes[index][0][0][3]) * 0.25,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax[0].add_patch(rect)
        ax[1].add_patch(rect1)
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
#            image = resize(image, (256, 256))
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
            fig.savefig('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/test_data/' + str(number) + '.png', dpi=fig.dpi)
            number = number +1

        images = np.array(images)
        print(images.shape)
        y_values = np.array(y_values)
        print(box)
        print(len(box))

        #with open('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/final_test_images', "wb") as fout:
         #   pickle.dump((images, y_values), fout)

        #with open('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/heatmaps', "wb") as fout:
         #   pickle.dump(heatmaps, fout)

        for i in y_values:
            print(i)

    # plotting images with their target and predicted value
    def plot_predict_image(self, image_wanted = 'mistakes_fn', num = 4):
        with open('../data/box_data_256_test', 'rb') as fin:
            self.boxes = pickle.load(fin)
        with open('../data/test_data_256', 'rb') as fin:
            self.x_no_hv, self.y_no_hv = pickle.load(fin)
        if image_wanted=='all':
            self.y_hat_no_hv = self.model.predict(self.x_no_hv, batch_size=128)
            self.y_pred_no_hv = self.y_hat_no_hv > 0.5
            y_indices = self.y_pred_no_hv
            print(len(self.boxes))
            print(len(self.x_no_hv))
            for index in range(1, y_indices.shape[0]):
                img = self.x_no_hv[index, :, :, 0]
                fig, ax = plt.subplots(1)
                if self.y_no_hv[index][0]==0:
                    print('nan')
                else:
                    if len(self.boxes[index]) == 1 and isinstance(self.boxes[index][0][0], int):
                        print(self.boxes[index])
                        print(self.boxes[index][0])
                        print(self.boxes[index][0][0])
                        xy = ((self.boxes[index][0][0]) * 0.25, (self.boxes[index][0][1]) * 0.25)
                        ax.imshow(img, cmap='gray')
                        ax.axis('off')
                        ax.text(0, 0,
                                "P=" + str([self.y_pred[index][0]]) + '   R=' + str(self.y_no_hv[index][0]) + "  Y^=" + str(
                                    self.y_hat_no_hv[index][0]))
                        rect = patches.Rectangle(xy, (self.boxes[index][0][2]) * 0.25, (self.boxes[index][0][3]) * 0.25,
                                                 linewidth=1, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                    elif len(self.boxes[index]) == 1 and isinstance(self.boxes[index][0][0], tuple):
                        for i in range(len(self.boxes[index])):
                            print(i)
                            print(self.boxes[index])
                            print(self.boxes[index][0])
                            print(self.boxes[index][0][0])
                            print('box:  ' + str(self.boxes[index][i]))
                            print(self.boxes[index][0][i])
                            xy = ((self.boxes[index][0][i][0]) * 0.25, (self.boxes[index][0][i][1]) * 0.25)
                            ax.imshow(img, cmap='gray')
                            ax.axis('off')
                            ax.text(0, 0, "P=" + str([self.y_pred[index][0]]) + '   R=' + str(self.y_no_hv[index][0]) + "  Y^=" + str(
                                self.y_hat_no_hv[index][0]))
                            rect = patches.Rectangle(xy, (self.boxes[index][0][i][2]) * 0.25, (self.boxes[index][0][i][3]) * 0.25,
                                                     linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                    else:
                        for i in range(len(self.boxes[index])):
                            print(i)
                            print(self.boxes[index])
                            print(self.boxes[index][0])
                            print(self.boxes[index][0][0])
                            print('box:  ' + str(self.boxes[index][i]))
                            print(self.boxes[index][0][i])
                            xy = ((self.boxes[index][i][0]) * 0.25, (self.boxes[index][i][1]) * 0.25)
                            ax.imshow(img, cmap='gray')
                            ax.axis('off')
                            ax.text(0, 0, "P=" + str([self.y_pred[index][0]]) + '   R=' + str(self.y_no_hv[index][0]) + "  Y^=" + str(
                                self.y_hat_no_hv[index][0]))
                            rect = patches.Rectangle(xy, (self.boxes[index][i][2]) * 0.25, (self.boxes[index][i][3]) * 0.25,
                                                     linewidth=1, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                    fig.savefig('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/' + str(index) + '.png', dpi=fig.dpi)
        else:
            for q in range(316):
                if image_wanted == 'random':
                    y_indices = self.y_pred
                elif image_wanted == 'mistakes':
                    y_indices = []
                    for i in range(0, self.y_pred.shape[0]):
                        if self.y_pred[i][0] != self.y[i][0]:
                            y_indices.append(i)

                elif image_wanted == 'mistakes_fn':
                    y_indices = []
                    for i in range(0, self.y_pred.shape[0]):
                        if self.y_pred[i][0] == False and self.y[i][0]==1:
                            y_indices.append(i)
                else:
                    exit('Not Defined')
                fig,ax = plt.subplots(1)
                if image_wanted == 'mistakes' or image_wanted=='mistakes_fn':
                    index = random.choice(y_indices)
                else:
                    index = random.randint(0, y_indices.shape[0])
                img = self.x[index, :, :, 0]
                xy = ((self.boxes[index][0][0][0])*0.25, (self.boxes[index][0][0][1])*0.25)
                ax.imshow(img, cmap='gray')
                ax.axis('off')
                ax.text(0, 0, "P=" + str([self.y_pred[index][0]]) + '   R=' + str(self.y[index][0]) + "  Y^=" + str(self.y_hat[index][0]))
                rect = patches.Rectangle(xy, (self.boxes[index][0][0][2])*0.25, (self.boxes[index][0][0][3])*0.25, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                fig.savefig('C:/Users/Manish/projects/tiya/scienceFair-2020/reports/images'+str(index)+'.png', dpi=fig.dpi)
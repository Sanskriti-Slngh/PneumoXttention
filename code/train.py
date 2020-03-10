import sys
import keras
from keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt
import numpy as np
import bz2

# import models
import models.m12.m12_model as m12
import models.m12.m12_multiclass as m12_multiclass
import models.m13.m13_model as m13
import models.m14.m14_model as m14
import models.m14.m14_multiclass as m14_multiclass
import models.m14.m14_no_dropout as m14_no_d
import models.m14.m14_attention as m14_attention
import models.m14.m14_attention_class3 as m14_attention_multiclass
import models.m15.m15_nih_ensemble as m15_nih_ensemble
import models.m15.m15_model as m15
import models.m15.m15_attention as m15_attention
import models.m15.m15_2mod_attn as m15_2mod_attn
import models.resnet.resnet50_model as resnet50
import models.resnet.resnet_multiclass as resnet_multiclass
import models.vgg16.pretrained_resnet50 as pre_resnet50
import models.vgg16.pretrained_vgg16 as vgg16
import models.m13h.m13h_model as m13h
import models.m15h.m15h_model as m15h
import gc
import re

# user options
model_name = 'm13'
use_adam = True
learning_rates = [0.00001]
decay_rate = 0
decay_epochs = 50
momentum = 0.9
batch_sizes = [16]
epochs = 300
minimize_false_negative = False
plot_only = False
channels_1x1 = [128]
predict = True
data_aug = True
data_gen = False
if model_name == 'vgg16':
    enable_vgg_224 = True
else:
    enable_vgg_224 = False

normalize = False
# set both to same
generate_heatmap = 'train' ;# 'train', 'test', 'val'
predict_on = 'test' ;# 'train', 'test, 'val'
multi_classification = False
binary = True
x1024 = False
x512 = False
x256 = True
add_nih = False
use_heatmap = False
reshape_h_data = False
train_with_mask_aug = False
plot_options = {'fp':False, 'roc_curve':False}

params = {}
params['resetHistory']  = False
params['print_summary'] = True
params['channels_1x1'] = 1
params['add_class_weight'] = False
params['use_heat_map'] = False
params['random_256_generation'] = False
params['split_random_sample'] = False
params['force_normalize'] = True
params['dropout'] = 0.5
#params['models_dir'] = 'C:/Users/Manish/projects/tiya/scienceFair-2020/experiments/256x256/random_256x256_generator'
#params['models_dir'] = '../experiments/256x256/m13h'
params['models_dir'] = 'F:/2020_science_fair_current_exp/channels_1x1/16/0.001/128/m13_d50'

# data files
# 0 - train data
# 1 - val data
# 2 - test data
if enable_vgg_224:
    data_files = ('../data/binaryclass/224/train_data.dat',
                  '../data/binaryclass/224/val_data_noaug.dat',
                  None)
    heatmap_files = (None, None, None)
elif binary and x1024:
    data_files = ('../data/binaryclass/1024x1024/train_data.dat',
                  '../data/binaryclass/1024x1024/rsna_val_no_hv.dat',
                  None)
    heatmap_files = (None, None, None)
elif binary and x512:
    data_files = ('../data/binaryclass/512x512/rsna_train.dat',
                  '../data/binaryclass/512x512/rsna_val_no_hv.dat',
                  '../data/binaryclass/512x512/rsna_test.dat')
    heatmap_files = ('../data/binaryclass/512x512/rsna_train_heatmap.dat.bz2',
                     '../data/binaryclass/512x512/rsna_val_no_hv_heatmap.dat.bz2',
                     '../data/binaryclass/512x512/rsna_test_heatmap.dat.bz2')
elif binary and x256:
    data_files = ('../data/binaryclass/256/train_data.dat.bz2',
                  '../data/binaryclass/256/val_data.dat.bz2',
                  '../data/binaryclass/256/test_data.dat.bz2')
    if use_heatmap:
        heatmap_files = ('../data/binaryclass/512x512/rsna_train_heatmap.dat.bz2',
                         '../data/binaryclass/512x512/rsna_val_no_hv_heatmap.dat.bz2',
                         '../data/binaryclass/512x512/rsna_test_heatmap.dat.bz2')
    else:
        heatmap_files = (None, None, None)
elif multi_classification and data_aug and not x1024:
    data_files = ('../data/multiclass/256x256/rsna_train_multiclassification.dat',
                  '../data/multiclass/256x256/rsna_val_multiclassification.dat',
                  None)
    heatmap_files = (None, None, None)
elif not data_aug and not multi_classification and not x1024:
    data_files = ('../data/binaryclass/train_data_256_no_hv',
                  '../data/binaryclass/val_data_256_no_hv',
                  None)
    heatmap_files = (None, None, None)
else:
    print('WRONG')
    exit()

def load_data_from_file(fname, dname):
    if re.match(".*.bz2", fname):
        fin = bz2.BZ2File(fname, 'rb')
        try:
            print("Reading data from file %s" % (fname))
            if len(dname) == 4:
                data0, data1, data2, data3 = pickle.load(fin)
            elif len(dname) > 1:
                data0, data1 = pickle.load(fin)
            else:
                data = pickle.load(fin)
        finally:
            fin.close()
    else:
        print("Reading data from file %s" % (fname))
        with open(fname, 'rb') as fin:
            if len(dname) == 4:
                data0, data1, data2, data3 = pickle.load(fin)
            elif len(dname) > 1:
                data0, data1  = pickle.load(fin)
            else:
                data = pickle.load(fin)
    if len(dname) == 4:
        print("%s shape: %s" % (dname[0], data0.shape))
        print("%s shape: %s" % (dname[1], data1.shape))
        print("%s shape: %s" % (dname[2], data2.shape))
        print("%s shape: %s" % (dname[3], data3.shape))
        return data0, data1, data2, data3
    elif (len(dname)) > 1:
        print ("%s shape: %s" %(dname[0], data0.shape))
        print("%s shape: %s" %(dname[1], data1.shape))
        return data0, data1
    else:
        print("%s shape: %s" % (dname[0], data.shape))
        return data

# Load data
x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None

b_train = None
b_train = None
b_val = None
b_val = None
b_test = None
b_test = None

h_train = None
h_train = None
h_val = None
h_val = None
h_test = None
h_test = None

for i,fname in enumerate(data_files):
    if not fname:
        continue
    if i == 0:
        x_train, y_train, b_train, p_train = load_data_from_file(fname, ("x_train", "y_train", "b_train", "p_train"))
    if i == 1:
        x_val, y_val, b_val, p_val = load_data_from_file(fname, ("x_val", "y_val", "b_val", "p_val"))
    if i == 2:
        x_test, y_test, b_test, p_test = load_data_from_file(fname, ("x_test", "y_test", "b_test", "p_test"))

for i, fname in enumerate(heatmap_files):
    if not fname:
        continue
    if i == 0:
        h_train = load_data_from_file(fname, ("h_train",))
        if reshape_h_data:
            h_train = np.reshape(h_train, (h_train.shape[0],17,17,1))
    if i == 1:
        h_val = load_data_from_file(fname, ("h_val",))
        if reshape_h_data:
            h_val = np.reshape(h_val, (h_val.shape[0],17,17,1))
    if i == 2:
        h_test = load_data_from_file(fname, ("h_test",))
        if reshape_h_data:
            h_test = np.reshape(h_test, (h_test.shape[0],17,17,1))

# if Multi classification, overwrite y values
if multi_classification:
    diseases = {"Normal":0,"No Lung Opacity / Not Normal":1, "Lung Opacity":2}
    with open("../data/multiclass/256x256/train_classes_data_256", 'rb') as fin:
        yyy, ids = pickle.load(fin)
    y_train = [diseases[x[0]] for x in yyy]
    with open("../data/multiclass/256x256/val_classes_data_256", 'rb') as fin:
        yyy, ids = pickle.load(fin)
    y_val = [diseases[x[0]] for x in yyy]

    print("Converting to category:Start")
    y_train_labels = y_train
    y_train = to_categorical(y_train, num_classes=3)
    y_val = to_categorical(y_val, num_classes=3)
    print("Converting to category:Done")

    # Shuffle data
    indices = np.arange(x_train.shape[0])
    np.random.shuffle(indices)
    x_train = x_train[indices]
    y_train = y_train[indices]

    indices = np.arange(x_val.shape[0])
    np.random.shuffle(indices)
    x_val = x_val[indices]
    y_val = y_val[indices]
    y_train_labels = y_train_labels[indices].tolist()

    print("x_train shape = %s, y_train shape = %s" % (x_train.shape, y_train.shape))

# dd nih data
if add_nih:
    nihd = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/all_nih_diseases/'
    diseases = ("Pleural_Thickening","Pneumothorax","Consolidation","Nodule","No Finding","Effusion","Cardiomegaly","Mass","Pneumonia","Atelectasis","Emphysema","Infiltration","Hernia","Fibrosis","Edema")
    d2idex = {}
    for i, disease in enumerate(diseases):
        d2idex[disease] = i
    x = []
    for i, disease in enumerate(diseases):
        print ("Reading nih data %s" %(disease))
        with open(nihd + disease + ".data", 'rb') as fin:
            x_tmp = pickle.load(fin)
            print (x_tmp.shape)
            if disease == "Pneumonia" or disease == 'Pneumothorax':
                if not multi_classification:
                    y_tmp = [1]*x_tmp.shape[0]
                else:
                    y_tmp = [2]*x_tmp.shape[0]
            elif disease == 'No Finding':
                y_tmp = [0]*x_tmp.shape[0]
            else:
                if not multi_classification:
                    y_tmp = [0] * x_tmp.shape[0]
                else:
                    y_tmp = [1] * x_tmp.shape[0]

            x.append(x_tmp)
            y_train.extend(y_tmp)

    x_train = np.concatenate((x_train, x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14]))

print("x_train shape = %s, y_train shape = %s" %(x_train.shape, y_train.shape))
print("class 0 images count = %d" %(np.sum(y_train == 0)))
print("class 1 images count = %d" %(np.sum(y_train == 1)))
print("class 2 images count = %d" %(np.sum(y_train == 2)))

# LossHistory Class
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
        gc.collect()
        if epoch%5 == 0:
            # Save model
            print ("Saving the model in ../experiments/current/m_" + str(epoch))
            model.save('../experiments/current/m_' + str(epoch))

# learning rate scheduler
def lr_scheduler(epoch, lr):
    global decay_rate, decay_epochs
    if epoch%decay_epochs == 0 and epoch and decay_rate != 0:
        return lr * decay_rate
    return lr

## Define the model
for batch_size in batch_sizes:
    for lr in learning_rates:
        for channels in channels_1x1:
            history = LossHistory()
            params['channels_1x1'] = channels
            if model_name == 'm12':
                model = m12.Model(history, params)

            elif model_name == 'm12_multiclass':
                model = m12_multiclass .Model(history, params)

            elif model_name == 'm13':
                model = m13.Model(history, params)

            elif model_name == 'm13h':
                model = m13h.Model(history, params)

            elif model_name == 'm14':
                model = m14.Model(history, params)

            elif model_name == 'm14_multiclass':
                model = m14_multiclass.Model(history, params)

            elif model_name == 'm14_no_d':
                model = m14_no_d.Model(history, params)

            elif model_name == 'm14_attention':
                model = m14_attention.Model(history, params)

            elif model_name == 'm14_attention_multiclass':
                model = m14_attention_multiclass.Model(history, params)

            elif model_name == 'm15':
                model = m15.Model(history, params)

            elif model_name == 'm15_attention':
                model = m15_attention.Model(history, params)

            elif model_name == 'm15_2mod_attn':
                model = m15_2mod_attn.Model(history, params)

            elif model_name == 'm15_nih_ensemble':
                model = m15_nih_ensemble.Model(history, params)

            elif model_name == 'm15h':
                model = m15h.Model(history, params)

            elif model_name == 'resnet50':
                model = resnet50.Model(history, params)

            elif model_name == 'resnet_multiclass':
                model = resnet_multiclass.Model(history, params)

            elif model_name == 'pre_resnet50':
                model = pre_resnet50.Model(history, params)

            elif model_name == 'vgg16':
                model = vgg16.Model(history, params)

            else:
                model = None

            if plot_only:
                model.is_train = False

            ## training
            if use_adam:
                optimizer = keras.optimizers.Adam(lr=lr)
            else:
                optimizer = keras.optimizers.SGD(lr=lr, momentum=momentum)

            model.compile(optimizer)

            # Load data into model
            model.x_train = x_train
            model.y_train = y_train
            model.b_train = b_train
            model.h_train = h_train
            model.x_val = x_val
            model.y_val = y_val
            model.b_val = b_val
            model.h_val = h_val
            if predict_on == 'train' and predict:
                print('predicting on %s' %(x_train.shape,))
                model.x_test = x_train
                model.y_test = y_train
                model.b_test = b_train
                model.h_test = h_train
            elif predict_on == 'val' and predict:
                print('predicting on %s' %(x_val.shape,))
                model.x_test = x_val
                model.y_test = y_val
                model.b_test = b_val
                model.h_test = h_val
            elif predict:
                print('predicting on %s' %(x_test.shape,))
                model.x_test = x_test
                model.y_test = y_test
                model.b_test = b_test
                model.h_test = h_test

            print (model.x_train.dtype)
            print (model.x_val.dtype)
            print (model.x_test.dtype)

            # instantiate model
            if not predict:
                fig, ax = plt.subplots(nrows=1, ncols=2)
            if not plot_only and not predict:
                model.train(batch_size, epochs, lr_scheduler)

            # save the model
            if not predict:
                model.save()

            if not predict:
                model.train_plot(fig, ax, show_plot=False, label=str(channels) + ' ' + str(lr))

            if predict and binary and x512:
                dump_heatmap = False
                if generate_heatmap == 'train':
                    assert(generate_heatmap == predict_on)
                    print('Generating heatmap for train data')
                    dump_heatmap = True
                    model.params['y_reshape_to_2d'] = True
                    model.params['random_256_generation'] = True
                    model.params['do_confusion_matrix'] = False
                    fname = heatmap_files[0]
                elif generate_heatmap == 'val':
                    assert (generate_heatmap == predict_on)
                    print('Generating heatmap for val data')
                    dump_heatmap = True
                    model.params['y_reshape_to_2d'] = True
                    model.params['random_256_generation'] = True
                    model.params['do_confusion_matrix'] = False
                    fname = heatmap_files[1]
                elif generate_heatmap == 'test':
                    assert (generate_heatmap == predict_on)
                    print('Generating heatmap for test data')
                    dump_heatmap = True
                    model.params['y_reshape_to_2d'] = True
                    model.params['random_256_generation'] = True
                    model.params['do_confusion_matrix'] = False
                    fname = heatmap_files[2]
                else:
                    dump_heatmap = False

                model.prediction(1, plot_options)
                model.gen_fifty(data_files[2], 16)
                if dump_heatmap:
                    fout = bz2.BZ2File(fname, 'wb')
                    try:
                        print('Dumping prediction in %s' % (fname))
                        pickle.dump(model.y_pred, fout, protocol=4)
#                        for i in range(model.y_test.shape[0]):
#                            print(model.y_test[i], model.y_hat[i], model.y_pred[i])
                    finally:
                        fout.close()
            elif predict:
                print('here')
#                model.gen_fifty(data_file=data_files[2], batch_size=4)
#                model.visualize_cnn(data_file=data_files[2])
                model.prediction(batch_size, plot_options)

if not predict:
    plt.legend()
    plt.show()


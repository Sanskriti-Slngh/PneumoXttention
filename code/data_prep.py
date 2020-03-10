# importing libraries
import bz2
import pickle
import numpy as np
import random
from keras.utils import to_categorical
from skimage.transform import resize
from skimage import img_as_ubyte
import math
import sys
import os

augmentation = False
write_train = True
write_test = True
random.seed(10000000)
resolution = 256 #(224|1024|256)
num_classes = 1 # (1|3)
# getting and saving the data
data_in = ("../data/train_data_pf.dat.bz2", "../data/test_data_pf.dat.bz2")
if (num_classes == 1):
    data_out_dir = "../data/binaryclass/" + str(resolution)
    diseases = {"Normal":0,"No Lung Opacity / Not Normal":0, "Lung Opacity":1}
else:
    data_out_dir = "../data/multiclass/" + str(resolution)
    diseases = {"Normal":0,"No Lung Opacity / Not Normal":1, "Lung Opacity":2}

def class2val(x):
    return diseases[x]

# create model directory if unknown
if not os.path.isdir(data_out_dir):
    os.makedirs(data_out_dir)

# 0 - No, 1 - h flip, 2 - v flip, 3 - hv flip
# id = 0 (train)
# id = 1 (test)
for id in range(2):
    # skip test loop if no write
    if id == 1 and not write_test:
        continue

    x_data = []
    y_data = []
    boxes = []
    patient_ids = []

    fin = bz2.BZ2File(data_in[id], 'rb')
    set = pickle.load(fin)

    number = set['patientId'].count()
    total_work = number
    work_done = 0
    for i in range(number):
        n_box = []
        h_box = []

        v_box = []
        hv_box = []
        dcm_pixels = set.iloc[i]['pixel_data']
        patient_ids.append([set.iloc[i]['patientId']])
        box = set.iloc[i]['boxes']
        factor = resolution/1024
        for idx, value in enumerate(box):
            x, y, width, height = set.iloc[i]['boxes'][idx][0], set.iloc[i]['boxes'][idx][1], set.iloc[i]['boxes'][idx][2], \
                set.iloc[i]['boxes'][idx][3]
            
            if not math.isnan(x):
                x = int(x * factor)
                y = int(y * factor)
                width = int(width * factor)
                height = int(height * factor)
                
            n_box.append([x,y, width, height])
            h_box.append([x, resolution-y-height, width, height])
            v_box.append([resolution-x-width, y, width, height])
            hv_box.append([resolution - x - width, resolution - y - height, width, height])

        x_data.append([dcm_pixels])
        y_data.append([class2val(set.iloc[i]['class'])])
        boxes.append(n_box)

        if id == 0 and augmentation and set.iloc[i]['class'] == 'Lung Opacity':
            img = dcm_pixels
            # horizontal flip
            img_h = np.flip(img, axis = 1)
            # vertical flip
            img_v = np.flip(img, axis = 0)
            # horizontal and vertical flip
            num = random.randint(0, 1)

            x_data.append([img_h])
            y_data.append([class2val('Lung Opacity')])
            boxes.append(h_box)
            patient_ids.append([set.iloc[i]['patientId']])

            x_data.append([img_v])
            y_data.append([class2val('Lung Opacity')])
            boxes.append(v_box)
            patient_ids.append([set.iloc[i]['patientId']])

            if num == 1:
                img_hv = np.flip(img_h, axis=0)
                x_data.append([img_hv])
                y_data.append([class2val('Lung Opacity')])
                patient_ids.append([set.iloc[i]['patientId']])
                boxes.append(hv_box)


        work_done += 1
        if work_done%200 == 0:
            sys.stdout.write('\r')
            # the exact output you're looking for:
            fmt = "[%-" + str(int(total_work/200)) + "s] %d%%"
            sys.stdout.write(fmt %('='*int(work_done/200), work_done*100/total_work))
            sys.stdout.flush()

    x = np.array(x_data)
    y = np.array(y_data)
    x = np.reshape(x, (x.shape[0], resolution, resolution, 1))
    patient_ids = np.array(patient_ids)

    print (x.shape)
    print (y.shape)

    if id == 0:
        # splitting into train and validation
        indices = [i for i in range(x.shape[0])]
        random.shuffle(indices)
        n_train = int(len(indices)*0.8)

        x_train = x[indices[0:n_train]]
        y_train = y[indices[0:n_train]]
        print (boxes[0:4])
        box_train = [boxes[i] for i in indices[0:n_train]]
        patient_ids_train = patient_ids[indices[0:n_train]]

        x_val = x[indices[n_train:]]
        y_val = y[indices[n_train:]]
        box_val = [boxes[i] for i in indices[n_train:]]
        patient_ids_val = patient_ids[indices[n_train:]]
        
        x_train = np.array(x_train)
        x_train = np.reshape(x_train, (x_train.shape[0], resolution, resolution, 1))
        y_train = np.array(y_train)
        box_train = np.array(box_train)

        x_val = np.array(x_val)
        x_val = np.reshape(x_val, (x_val.shape[0], resolution, resolution, 1))
        y_val = np.array(y_val)
        box_val = np.array(box_val)

        if write_train:
            fout = bz2.BZ2File(data_out_dir + "/train_data.dat.bz2", 'wb')
            print ("Dumping %s" %(data_out_dir + "/train_data.dat.bz2"))
            print (x_train.shape, y_train.shape)
            pickle.dump((x_train, y_train, box_train, patient_ids_train), fout, protocol=4)

        fout = bz2.BZ2File(data_out_dir + "/val_data.dat.bz2", 'wb')
        print (x_val.shape, y_val.shape)
        print ("Dumping %s" %(data_out_dir + "/val_data.dat.bz2"))
        pickle.dump((x_val, y_val, box_val, patient_ids_val), fout, protocol=4)

    else:
        fout = bz2.BZ2File(data_out_dir + "/test_data.dat.bz2", 'wb')
        print (x.shape, y.shape)
        print ("Dumping %s" %(data_out_dir + "/test_data.dat.bz2"))
        pickle.dump((x, y, np.array(boxes), patient_ids), fout, protocol=4)


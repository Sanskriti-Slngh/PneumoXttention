# importing libraries
import pickle
import numpy as np
import random
from keras.utils import to_categorical
from skimage.transform import resize
import math
import sys
augmentation = True
random.seed(10000000)
resolution = 1024 #(224|1024|256)
num_classes = 3 # (1|3)
# getting and saving the data
if resolution == 1024:
    data_in = "C:/Users/Manish/projects/tiya/scienceFair-2020/data/multiclass/1024x1024/rsna_train_pf.dat"
    data_out = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/rsna_train.dat'
    data_out_box_train = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/box_data/rsna_train_box.dat'
    if augmentation:
        data_out_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/rsna_val.dat'
        data_out_box_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/box_data/rsna_val_box.dat'
    else:
        data_out_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/rsna_val_no_hv.dat'
        data_out_box_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/1024x1024/box_data/rsna_val_box_no_hv.dat'
elif resolution == 512:
    data_in = "C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512/train_data_pf.dat"
    data_out = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/rsna_train.dat'
    data_out_box_train = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/box_data/rsna_train_box.dat'
    if augmentation:
        data_out_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/rsna_val.dat'
        data_out_box_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/box_data/rsna_val_box.dat'
    else:
        data_out_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/rsna_val_no_hv.dat'
        data_out_box_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/512x512/box_data/rsna_val_box_no_hv.dat'
elif resolution == 224:
    data_in = "C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/224/train_data_pf.dat"
    data_out = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/224/train_data_noaug.dat'
    data_out_val = 'C:/Users/Manish/projects/tiya/scienceFair-2020/data/binaryclass/224/val_data_noaug.dat'
else:
    print ("ERROR")
    exit()

x_train = []
y_train = []
boxes = []
patient_id = []

# 0 - No, 1 - h flip, 2 - v flip, 3 - hv flip
with open(data_in, 'rb') as fin:
    set = pickle.load(fin)
    number = set['patientId'].count()
    for i in range(number):
        n_box = []
        h_box = []
        v_box = []
        hv_box = []
        y_train.append([set.iloc[i]['class']])
        dcm_pixels = set.iloc[i]['pixel_data']
        if i%100==0:
            print (i, dcm_pixels.shape)
        x_train.append([dcm_pixels])
        patient_id.append([set.iloc[i]['patientId']])
        box = set.iloc[i]['boxes']
        for idx, value in enumerate(box):
            x, y, width, height = set.iloc[i]['boxes'][idx][0], set.iloc[i]['boxes'][idx][1], set.iloc[i]['boxes'][idx][2], \
                                  set.iloc[i]['boxes'][idx][3]
            factor = resolution/1024

            if not math.isnan(x):
                x = int(x * factor)
                y = int(y * factor)
                width = int(width * factor)
                height = int(height * factor)

            n_box.append([x,y, width, height])
            h_box.append([x, resolution-y-height, width, height])
            v_box.append([resolution-x-width, y, width, height])
            hv_box.append([resolution - x - width, resolution - y - height, width, height])
        boxes.append(n_box)

        if augmentation and set.iloc[i]['class'] == 'Lung Opacity':
            y_train.append(['Lung Opacity'])
            y_train.append(['Lung Opacity'])
            img = dcm_pixels
            # horizontal flip
            img_h = np.flip(img, axis = 1)
            # vertical flip
            img_v = np.flip(img, axis = 0)
            # horizontal and vertical flip
            num = random.randint(0, 1)
            if num == 1:
                img_hv = np.flip(img_h, axis=0)
                x_train.append([img_hv])
                y_train.append(['Lung Opacity'])
                patient_id.append([set.iloc[i]['patientId']])
                boxes.append(hv_box)
            x_train.append([img_h])
            patient_id.append([set.iloc[i]['patientId']])
            boxes.append(h_box)
            x_train.append([img_v])
            patient_id.append([set.iloc[i]['patientId']])
            boxes.append(v_box)

del set

print (y_train[0:10])
print (boxes[0:10])

x_train = np.array(x_train)
print (x_train.shape)
x_train = np.reshape(x_train, (x_train.shape[0], resolution, resolution, 1))
print (x_train.shape)
patient_id = np.array(patient_id)
if num_classes == 3:
    diseases = {"Normal":0,"No Lung Opacity / Not Normal":1, "Lung Opacity":2}
    y_train_d = [diseases[i[0]] for i in y_train]
    y_train = to_categorical(y_train_d, num_classes=3)
else:
    diseases = {"Normal":0,"No Lung Opacity / Not Normal":0, "Lung Opacity":1}
    y_train_d = [[diseases[i[0]]] for i in y_train]
    y_train = y_train_d

print(y_train[0:10])
print (boxes[0:10])

#splitting into train and validation
x_val_data = []
y_val_data = []
box_val_data = []
patient_ids_val = []
x_train_data = []
y_train_data = []
box_train_data = []
patient_ids_train = []

for i in range(0,x_train.shape[0]):
    random_number = random.randint(1,100)
    img = x_train[i, :, :, 0]
    y_value = y_train[i]
    classification = patient_id[i]
    box = boxes[i]
    if random_number > 80:
        x_val_data.append(img)
        y_val_data.append(y_value)
        box_val_data.append(box)
        patient_ids_val.append(classification)
    else:
        x_train_data.append(img)
        y_train_data.append(y_value)
        box_train_data.append(box)
        patient_ids_train.append(classification)

del x_train, y_value, boxes

x_val_data = np.array(x_val_data)
x_val_data = np.reshape(x_val_data, (x_val_data.shape[0], resolution, resolution, 1))
y_val_data = np.array(y_val_data)
x_train_data = np.array(x_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], resolution, resolution, 1))
y_train_data = np.array(y_train_data)
box_val_data = np.array(box_val_data)
box_train_data = np.array(box_train_data)

def sizeof(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict): return size + sum(map(sizeof, obj.keys())) + sum(map(sizeof, obj.values()))
    if isinstance(obj, (list, tuple, set, frozenset)): return size + sum(map(sizeof, obj))
    return size

#for i in range(y_val_data.shape[0]):
#    print ("%s %s" %(y_val_data[i], box_val_data[i]))
print (x_train_data.shape)
print (sizeof(x_train_data[:,:,:,:]))
print (x_train_data.astype)
print (sizeof(y_train_data))

if augmentation:
    with open(data_out, 'wb') as fout:
        print ("Dumping %s" %(data_out))
        pickle.dump((x_train_data, y_train_data), fout, protocol=4)

    with open(data_out_box_train, 'wb') as fout:
        print("Dumping %s" % (data_out_box_train))
        pickle.dump(box_train_data, fout, protocol=4)

with open(data_out_val, 'wb') as fout:
    print("Dumping %s" % (data_out_val))
    pickle.dump((x_val_data, y_val_data), fout, protocol=4)

#with open(data_out_box_val, 'wb') as fout:
 #   print("Dumping %s" % (data_out_box_val))
  #  pickle.dump(box_val_data, fout, protocol=4)

# Libraries
import pickle
import random
import numpy as np

random.seed(1234567)

data_in = "C:/Users/Manish/projects/tiya/scienceFair-2019.junk/data/rsna_train_data_with_256.dat"
data_out = "../data/train_data_256_no_hv"
data_out_val = "../data/val_data_256_no_hv"
data_out_box = "../data/box_data_256_no_hv"

x_train = []
y_train = []
box_train = []

with open(data_in, 'rb') as fin:
    set = pickle.load(fin)
    number = set['patientId'].count()
    for i in range(number):
        y_train.append([set.iloc[i]['Target']])
        x_train.append([set.iloc[i]['pixel_data']])
        box_train.append([set.iloc[i]['boxes']])

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], 256, 256, 1))
y_train = np.array(y_train)

x_val_data = []
y_val_data = []
box_val_data = []
x_train_data = []
y_train_data = []
for i in range(0,y_train.shape[0]):
    random_number = random.randint(1,100)
    img = x_train[i, :, :, 0]
    box = box_train[i]
    y_value = y_train[i]
    if random_number > 80:
        x_val_data.append(img)
        y_val_data.append(y_value)
        box_val_data.append(box)
    else:
        x_train_data.append(img)
        y_train_data.append(y_value)

x_val_data = np.array(x_val_data)
x_val_data = np.reshape(x_val_data, (x_val_data.shape[0], 256, 256, 1))
y_val_data = np.array(y_val_data)
box_val_data = np.array(box_val_data)
x_train_data = np.array(x_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], 256, 256, 1))
y_train_data = np.array(y_train_data)

#with open(data_out, 'wb') as fout:
  #  pickle.dump((x_train_data, y_train_data), fout, protocol=4)

#with open(data_out_val, 'wb') as fout:
 #   pickle.dump((x_val_data, y_val_data), fout, protocol= 4)

with open(data_out_box, 'wb') as fout:
    pickle.dump(box_val_data, fout, protocol= 4)

# importing libraries
import pickle
import numpy as np
import random
import math
from keras.utils import to_categorical

random.seed(10000000)
# getting and saving the data
data_in = "C:/Users/Manish/projects/tiya/scienceFair-2020/data/multiclass/rsna_test_mutliclassification_pf_1024.dat"
data_out = '../data/multiclass/rsna_test.dat'
resolution = 1024

x = []
y = []
ids = []
boxes = []

with open(data_in, 'rb') as fin:
    set = pickle.load(fin)
    number = set['patientId'].count()
    for i in range(number):
        n_box = []
        y.append(np.array([set.iloc[i]['class']]))
        x.append([set.iloc[i]['pixel_data']])
        ids.append([set.iloc[i]['patientId']])

x_test = np.array(x)
x_test = np.reshape(x_test, (x_test.shape[0], 1024, 1024, 1))

diseases = {"Normal":0,"No Lung Opacity / Not Normal":1, "Lung Opacity":2}
y_test = [diseases[i[0]] for i in y]
y_test = to_categorical(y_test, num_classes=3)

boxes = np.array(boxes)

print(x_test.shape)
print(y_test.shape)
print(boxes.shape)

with open(data_out, "wb") as fout:
    print("Saving data file into %s" % (data_out))
    pickle.dump((x_test, y_test), fout, protocol=4)

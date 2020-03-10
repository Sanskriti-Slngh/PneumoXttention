# importing libraries
import pickle
import numpy as np
import random
import math
from keras.utils import to_categorical

random.seed(10000000)
# getting and saving the data
# data_in = "../data/binaryclass/512x512/test_data_pf.dat"
data_in = "C:/Users/Manish/projects/tiya/scienceFair-2020/data/multiclass/rsna_test_mutliclassification_pf_1024.dat"
#data_out = '../data/binaryclass/512x512/rsna_test.dat'
data_out = '../data/multiclass/rsna_test.dat'
#data_out_box = '../data/binaryclass/512x512/box_data/rsna_test_box.dat'
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
        #box = set.iloc[i]['boxes']
        #for idx, value in enumerate(box):
        #    x_c, y_c, width, height = set.iloc[i]['boxes'][idx][0], set.iloc[i]['boxes'][idx][1], set.iloc[i]['boxes'][idx][
        #        2], \
        #                          set.iloc[i]['boxes'][idx][3]
        #    factor = resolution / 1024
#
        #    if not math.isnan(x_c):
        #        x_c = int(x_c * factor)
        #        y_c = int(y_c * factor)
        #        width = int(width * factor)
        #        height = int(height * factor)
#
        #    n_box.append([x_c, y_c, width, height])
        #boxes.append(n_box)


x_test = np.array(x)
x_test = np.reshape(x_test, (x_test.shape[0], 1024, 1024, 1))
#y = np.array(y)

#diseases = {"Normal":0,"No Lung Opacity / Not Normal":0, "Lung Opacity":1}
#y_test_d = [[diseases[i[0]]] for i in y]
#y_test = y_test_d
#y_test = np.array(y_test)

diseases = {"Normal":0,"No Lung Opacity / Not Normal":1, "Lung Opacity":2}
y_test = [diseases[i[0]] for i in y]
y_test = to_categorical(y_test, num_classes=3)

#print(boxes)
boxes = np.array(boxes)

#print (boxes[0])
print(x_test.shape)
print(y_test.shape)
print(boxes.shape)

with open(data_out, "wb") as fout:
    print("Saving data file into %s" % (data_out))
    pickle.dump((x_test, y_test), fout, protocol=4)

#with open(data_out_box, 'wb') as fout:
 #   print("Saving data file into %s" % (data_out_box))
  #  pickle.dump(boxes, fout, protocol=4)
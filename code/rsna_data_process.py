import random
# fix the seed so that we get same train/test splitpy
import pandas as pd
import pydicom
import os
import bz2
import pickle
from skimage.transform import resize
from skimage import img_as_ubyte
import sys
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
import math

# set seed so we are able to reproduce the results.
random.seed(10000000)

# Change this to your directories
rsna_data_dir = 'F:/'
data_out_dir = "../data/"
forceRun = False ;# if set it to false and file exist, it will skip generation of data

## Code from here
if not os.path.isdir(data_out_dir):
    os.mkdir(data_out_dir)

for resolution in (256, 512):

    out_dir = data_out_dir + str(resolution) + "/"
    if not os.path.isdir(out_dir):
        os.mkdir(data_out_dir + str(resolution) + "/")

    out_train_pf = out_dir + "train_data_pf.dat.bz2"
    out_test_pf = out_dir + "test_data_pf.dat.bz2"

    if not os.path.isfile(out_train_pf) or forceRun:
        tr = pd.read_csv(rsna_data_dir + 'stage_2_train_labels.csv')
        tr['aspect_ratio'] = (tr['width'] / tr['height'])
        tr['area'] = tr['width'] * tr['height']

        tr2 = pd.read_csv(rsna_data_dir + 'stage_2_detailed_class_info.csv')
        def get_info(patientId, tr, tr2, root_dir=rsna_data_dir + 'stage_2_train_images'):
            fn = os.path.join(root_dir, f'{patientId}.dcm')
            dcm_data = pydicom.read_file(fn)
            a = tr.loc[lambda tr: tr.patientId==patientId, :]
            boxes = []
            for i in range(len(a)):
                boxes.append((a.iloc[i]['x'],a.iloc[i]['y'],a.iloc[i]['width'],a.iloc[i]['height']))
            dcm_pixels = np.array(dcm_data.pixel_array)
            dcm_pixels = resize(dcm_pixels, (resolution, resolution))
            dcm_pixels = img_as_ubyte(dcm_pixels)
            return {'age': int(dcm_data.PatientAge),
                    'gender': dcm_data.PatientSex,
                    'id': os.path.basename(fn).split('.')[0],
                    'pixel_spacing': float(dcm_data.PixelSpacing[0]),
                    'boxes':boxes,
                    'Modality':dcm_data.Modality,
                    'pixel_data': dcm_pixels}


        patient_ids = list(tr.patientId.unique())
        result = []
        total_work = len(patient_ids)
        work_done = 0
        for i in patient_ids:
            result.append(get_info(i, tr, tr2))
            work_done += 1
            if work_done%200 == 0:
                sys.stdout.write('\r')
                # the exact output you're looking for:
                fmt = "[%-" + str(int(total_work/200)) + "s] %d%%"
                sys.stdout.write(fmt %('='*int(work_done/200), work_done*100/total_work))
                sys.stdout.flush()

        demo = pd.DataFrame(result)
        demo['gender'] = demo['gender'].astype('category')
        demo['age'] = demo['age'].astype(int)
        tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left').drop(columns=['id','x','y','width','height']))
        tr = (tr.merge(tr2, left_on='patientId', right_on='patientId', how='left'))
        tf = tr.drop_duplicates(subset=['patientId'])
        print (sys.getsizeof(tf))

        # train, test split
        train, test = train_test_split(tf, train_size=0.8, test_size=0.2, random_state=112341134)

        # saving train into data file
        fout = bz2.BZ2File(out_train_pf, "wb")
        print("Saving data file into %s" % (out_train_pf))
        pickle.dump((train), fout, protocol=4)

        # saving test into data file
        fout = bz2.BZ2File(out_test_pf, 'wb')
        print("Saving data file into %s" % (out_test_pf))
        pickle.dump((test), fout, protocol=4)

    else:
        # getting train set from data file
        fin = bz2.BZ2File(out_train_pf, "rb")
        print("Reading data from file %s" % (out_train_pf))
        train = pickle.load(fin)

        # getting test set from data file
        fin = bz2.BZ2File(out_test_pf, "rb")
        print("Reading data from file %s" % (out_test_pf))
        test = pickle.load(fin)


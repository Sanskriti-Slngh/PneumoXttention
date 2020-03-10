import os
import keras
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Reshape, UpSampling2D, Lambda, dot
from keras.models import Model as m
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation
from models.base_model import BaseModel
import tensorflow as tf
import numpy as np
import scipy.special as scp

def attention(x):
    q, v, k = x
    print(q.shape, v.shape, k.shape)
    s = dot([q, k], axes=2, normalize=False)
    s = K.softmax(s, axis=2)
    print(s.shape)
    s1 = dot([s,v], axes=(2,1))
    print(s1.shape)
    return s1

q = tf.constant([[[0.,1.,2.,3.,4.,5.,6.,7.], [1.,2.,3.,4.,5.,6.,7.,8.], [2.,3.,4.,5.,6.,7.,8.,9.], [3.,4.,5.,6.,7.,8.,9.,10.]],[[0.,1.,2.,3.,4.,5.,6.,7.], [1.,2.,3.,4.,5.,6.,7.,8.], [2.,3.,4.,5.,6.,7.,8.,9.], [3.,4.,5.,6.,7.,8.,9.,10.]]])
k = tf.constant([[[0,.1,.2,.3,.4,.5,.6,.7], [.1,.2,.3,.4,.5,.6,.7,.8], [.2,.3,.4,.5,.6,.7,.8,.9], [.3,.4,.5,.6,.7,.8,.9,.10]],[[0,.1,.2,.3,.4,.5,.6,.7], [.1,.2,.3,.4,.5,.6,.7,.8], [.2,.3,.4,.5,.6,.7,.8,.9], [.3,.4,.5,.6,.7,.8,.9,.10]]])
s1 = attention((q,q,k))
print (s1)

x1 = np.array([[0,1,2,3,4,5,6,7], [1,2,3,4,5,6,7,8], [2,3,4,5,6,7,8,9], [3,4,5,6,7,8,9,10]])
x2 = np.array([[0,.1,.2,.3,.4,.5,.6,.7], [.1,.2,.3,.4,.5,.6,.7,.8], [.2,.3,.4,.5,.6,.7,.8,.9], [.3,.4,.5,.6,.7,.8,.9,.10]])
x3 = np.dot(x1, x2.T)
x3 = scp.softmax(x3, axis=1)
x4 = np.dot(x3,x1)
print (x4)

sess = tf.Session()
# Print the result
print(sess.run(s1))
# Close the session
sess.close()
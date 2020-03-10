## Credit to https://github.com/eclique/keras-gradcam/blob/master/gradcam_vgg.ipynb

import os
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from skimage.transform import resize
import pickle
from keras.models import load_model
import matplotlib.patches as patches

import bz2
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.layers import Input, Reshape, UpSampling2D, Lambda, dot

#np.random.seed(100)

H = 256
W = 256
is_m13h = False

if not is_m13h:
    model_name = 'F:/2020_science_fair_current_exp/channels_1x1/16/0.001/128/m13_d50/m13'
else:
    model_name = '../experiments/256x256/m13h/m13h'

# functions
def get_f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_val

# load data
#data = '../data/binaryclass/256/train_data.dat.bz2'
data = '../data/binaryclass/256/val_data.dat.bz2'
if is_m13h:
    heat_map = '../data/binaryclass/512x512/rsna_val_no_hv_heatmap.dat.bz2'
else:
    heat_map = None

fin = bz2.BZ2File(data, 'rb')
try:
    print("Reading data from file %s" % (data))
    x_val, y_val, box_val, patient_id_val = pickle.load(fin)
finally:
    fin.close()

if heat_map:
    fin = bz2.BZ2File(heat_map, 'rb')
    try:
        print("Reading heat map from file %s" % (heat_map))
        h_val = pickle.load(fin)
    finally:
        fin.close()

# normalize
norm_const = np.array(255).astype('float16')

def load_image(index, val=True, preprocess=True):
    """Load and preprocess image."""
    x = x_val[index]
    if preprocess:
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        x = x / norm_const
    return x


def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    x = x.copy()
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)

# Function returning modified model.
def build_model():
    print ("Loading model %s.h5py" %(model_name))
    model = load_model(model_name + '.h5py', custom_objects={'dot': dot, 'get_f1': get_f1})
    print (model.summary())
    return model


#Changes gradient function for all ReLu activations
#according to Guided Backpropagation.

def build_guided_model():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        new_model = build_model()
    return new_model

def guided_backprop(input_model, images, layer_name):
    # Guided Backpropagation method for visualizing input saliency
    input_imgs = input_model.input
    layer_output = input_model.get_layer(layer_name).output
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([images, 0])[0]
    return grads_val


def grad_cam(input_model, image, cls, layer_name):
    # GradCAM method for visualizing input saliency
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    # grads = normalize(grads)
    if is_m13h:
        gradient_function = K.function([input_model.input[0]], [conv_output, grads])
    else:
        gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    print(output.shape)
    print(grads_val.shape)
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    print(grads_val.shape)

    weights = np.mean(grads_val, axis=(0, 1))
    print(weights.shape)
    cam = np.dot(output, weights)
    print (cam.shape)

    # Process CAM
    cam = resize(cam, (H, W))
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_saliency(model, guided_model, img_index, val_image, layer_name='conv2d_101', cls=0, visualize=True):
    # Compute saliency using all three approaches.
    # layer_name: layer to compute gradients;
    # cls: class number to localize (-1 for most probable class).
    preprocessed_input = load_image(img_index, val_image)

    if not heat_map:
        y_hat = model.predict(preprocessed_input)
        gradcam = grad_cam(model, preprocessed_input, cls, layer_name)
    else:
        h = np.reshape(h_val[img_index],(1,17,17))
        y_hat = model.predict([preprocessed_input, h])
        gradcam = grad_cam(model, [preprocessed_input, h], cls, layer_name)
    print('True prediction, Model prediction:')
    print(y_val[img_index], y_hat)
    gb = guided_backprop(guided_model, preprocessed_input, layer_name)
    guided_gradcam = gb * gradcam[..., np.newaxis]

    if visualize:
        #plt.figure(figsize=(15, 10))
        fig, ax = plt.subplots(1,3)
        fig.suptitle('%s %s' %(y_val[img_index], y_hat[0]))
        ax[0].axis('off')
        ax[0].imshow(load_image(img_index, val_image, preprocess=False)[:,:,0], cmap='gray')
        ax[0].title.set_text('Input Image')
        boxes = box_val[img_index]
        print (boxes)
        factor = 1.0
        for i in range(len(boxes)):
            xy = ((boxes[i][0]) * factor, (boxes[i][1]) * factor)
            rect = patches.Rectangle(xy, (boxes[i][2]) * factor, (boxes[i][3]) * factor,
                                         linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
        ax[0].imshow(gradcam, cmap='gray', alpha=0.5)

        ax[1].axis('off')
        ax[1].imshow((np.flip(deprocess_image(gb[0]), -1)[:,:,0]), cmap='gray')
        ax[1].title.set_text('Guided Backprop')

        ax[2].axis('off')
        ax[2].imshow(load_image(img_index, val_image, preprocess=False)[:,:,0], cmap='gray')
        ax[2].imshow((np.flip(deprocess_image(guided_gradcam[0]), -1)[:,:,0]), cmap='cool', alpha=0.5)
        ax[2].title.set_text('Guided GradCam')
        plt.show()

    return gradcam, gb, guided_gradcam

model = build_model()
guided_model = build_guided_model()
# negative (3001)
#gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, 4231, True)
# positive
print (y_val[10])
pos_indices = [i for i, x in enumerate(y_val) if x > 0]
for i in range(20):
    index = np.random.choice(pos_indices)
    if is_m13h:
        gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, index, True, 'conv2d_12')
    else:
        gradcam, gb, guided_gradcam = compute_saliency(model, guided_model, index, True, 'conv2d_103')
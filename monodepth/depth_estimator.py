import os
import time
import numpy as np
import tensorflow as tf

from PIL import Image
from skimage.transform import resize

from monodepth.model import MVAAutoEncoder


def resize_image(color):
    return np.asarray(Image.fromarray(color.copy()).resize((640, 480)))
    
    
def preprocess_image(color):
    # Normalize image. [0, 1]
    normalized_color = color / 255.

    # Expand dimension. (batch, height, width, channel)
    net_input = np.expand_dims(normalized_color, axis=0)
    return net_input


def load_trained_model(weights_path):
    net = MVAAutoEncoder()
    model = net.build_model()
    model.load_weights(weights_path)
    return model


def estimate_depth(net_input, height, width, weights_path):
    model = load_trained_model(weights_path)
    print('Model loaded...')

    pred = model.predict(net_input)
    pred = pred.reshape((240, 320))
    pred = np.clip((1000 / pred), 10, 1000)
    pred = resize(pred, (height, width), order=1, preserve_range=True, mode='reflect', anti_aliasing=True)
    print('Depth predicted...')
    return pred
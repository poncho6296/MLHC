
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

import tensorflow as tf
from tensorflow import keras

from keras.preprocessing import image

from keras.applications.resnet import preprocess_input
import numpy as np

from sklearn.random_projection import GaussianRandomProjection

model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

import os

#here we get the path of the images
my_path="/home/rymekabak/appended_two/"
train = os.listdir(my_path)


vgg16_feature_list = []
print('feature extraction start')

for img in os.listdir(my_path):
    img_path = my_path+'/'+img
    img = image.load_img(img_path, target_size=(256, 256))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    vgg16_feature = model.predict(img_data)
    vgg16_feature_np = np.array(vgg16_feature)
    vgg16_feature_list.append(vgg16_feature_np.flatten())
    print(img_path)

vgg16_feature_list_np = np.array(vgg16_feature_list)
jl = GaussianRandomProjection(eps=.25)
embeddings_lowdim = jl.fit_transform(vgg16_feature_list_np)

np.save("embeddings_lowdim.npy", embeddings_lowdim)
print(embeddings_lowdim.shape)
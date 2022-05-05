from sys import exit
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os #to load the data
import random
import PIL #manage images 

import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.utils import class_weight



# Here we  get the path with the images
# %%
data_dir='/home/jjuarezp/appended/appended/'
df = pd.read_csv('/home/jjuarezp/appended/dataset_2images.csv')
df_interest = df[['eid_broad', 'has_disease']]
#path of the images
train = os.listdir(data_dir)
path_train = os.path.join(data_dir)


# Plot some images from the train set
w = 2
h = 2

# we resize the format of the images with a 200x200 shape
# this function uses the open, resize and array functions to manage data visualizations
#load_img = lambda filename: np.array(PIL.Image.open(f"/Users/rymekabak/Desktop/HEALTHCARE/Images/appended/{filename}"))
load_img = lambda filename: np.array(PIL.Image.open(data_dir+filename))

#Here we read and resize all images
list_tensor=[]
list_error=[]
for t in train:
    array=Image.open(data_dir+t)
    try:
        array=array.convert("RGB")
    except:
        list_error.append(t)
        continue        
    array = array.resize((103, 92))
    tensor = tf.convert_to_tensor(np.array(array))
    list_tensor.append(tensor)

print("Total number of images:", len(train))
print("Number of images left out are:", len(list_error))
#Here we get the image id to later match with our target variable
image_names = pd.DataFrame(train, columns=['image_id'])
image_names['eid_broad'] = image_names.image_id.str.rsplit(pat='_').str[0].astype(int)
#here we have the new df with the labels
image_names = image_names.merge(df_interest, how='left', on='eid_broad')
print("Target person before filtering:",image_names["has_disease"].sum())
image_names=image_names[~image_names["image_id"].isin(list_error)]
print("Target person after filtering:",image_names["has_disease"].sum())

#Here we divided the images between training and test set
X_train, X_test, y_train, y_test = train_test_split(list_tensor, image_names.has_disease, test_size=0.33, random_state=42)
X_train=tf.stack(X_train)
X_test=tf.stack(X_test)



#Here we set the data augmentation layer
data_augmentation_complete = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.2),
    ]
)



#Here we comute the weights for balacing the training
weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)

class_weight = {0: weights[0],
                1: weights[1]}



#Here we create the model and run the model 
# %%
input = keras.Input(shape=X_train.shape[1:])
input = data_augmentation_complete(input)
x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(input) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer
x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_2")(x) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer

x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_3")(x) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer

x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_4")(x) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer

x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_5")(x) # convolutional layer!
x = keras.layers.MaxPool2D(padding='same')(x) # pooling layer



x = keras.layers.Flatten()(x)
x = keras.layers.Dense(256, activation="relu")(x)   
output = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(input, output)


# %%
model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy',"AUC","Precision","Recall"])

history = model.fit(X_train, y_train, batch_size=64, epochs=10,class_weight=class_weight)



score = model.evaluate(X_test, y_test)
print("Test accuracy:", score[1])

print("scores ", score)

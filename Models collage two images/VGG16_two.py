import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os #to load the data
import random
import PIL #manage images 
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from sklearn.utils import class_weight
import sklearn

print(sklearn.__version__)

#Here we read the df with the labels
df_interest = pd.read_csv('/home/jjuarezp/appended/dataset_2images.csv')
#Create a subset of the df
df_interest = df_interest[['eid_broad', 'has_disease']]
# Here we need to change the path
my_path="/home/jjuarezp/appended/appended_two/"
train = os.listdir(my_path)

path_train = os.path.join(my_path)


# Plot some images from the train set 
w = 2
h = 2

# we resize the format of the images with a 200x200 shape
# this function uses the open, resize and array functions to manage data visualizations
#load_img = lambda filename: np.array(PIL.Image.open(f"/Users/rymekabak/Desktop/HEALTHCARE/Images/appended/{filename}"))
load_img = lambda filename: np.array(PIL.Image.open(my_path+filename))

#Here we read and resize all images
list_tensor=[]
list_error=[]
for t in train:
    
    array=Image.open(my_path+t)
    #print(my_path+t)
    try:
        array=array.convert("RGB")
    except:
        list_error.append(t)
        continue        
    array = array.resize((103, 92))
    tensor = tf.convert_to_tensor(np.array(array))
    #tensor=tf.reshape(tensor, [103,92,3])
    list_tensor.append(tensor)

#Here we have the imageid form the name of the file
image_names = pd.DataFrame(train, columns=['image_id'])
image_names['eid_broad'] = image_names.image_id.str.rsplit(pat='.').str[0].astype(int)
#here we have the new df with the labels
image_names = image_names.merge(df_interest, how='left', on='eid_broad')
image_names=image_names[~image_names["image_id"].isin(list_error)]
    
#Here we divided the images between training and test set
X_train, X_test, y_train, y_test = train_test_split(list_tensor, image_names.has_disease, test_size=0.33, random_state=42)
X_train=tf.stack(X_train)
X_test=tf.stack(X_test)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Here we set the data augmentation layer
data_augmentation_complete = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.2),
    ]
)

weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)

class_weight = {0: weights[0],
                1: weights[1]}

#Here we start setting the VGG16 model

top_model = keras.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
    ]
)


def construct_model_16(no_classes, input_shape, metrics=['accuracy',"AUC","Precision","Recall"]):

#Here we name the model
  base_model = keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="sigmoid",
  )

  # Freeze the base_model
  base_model.trainable = False

  #
  # Create new model on top
  inputs = keras.Input(shape=input_shape) 
  print(input_shape)

  x = keras.layers.Rescaling(1./255)(inputs) #normalizing

  # Apply random data augmentation
  x = data_augmentation_complete(x)  

  # The base model contains batchnorm layers. We want to keep them in inference mode
  # when we unfreeze the base model for fine-tuning, so we make sure that the
  # base_model is running in inference mode here. We didn't cover batchnorm 
  # layers in class so just take our word for it :-)
  x = base_model(x, training=False)
  
  # Next we feed the output from our base model to the top model we designed. 
  x = top_model(x)
  
  outputs = keras.layers.Dense(no_classes, activation='sigmoid')(x)
  
  model = keras.Model(inputs, outputs)
  model.summary()

  #unfreeze the last 10 layers of the model so that some tweaks can be done to the weights of the VGG19 model. 
  for layer in model.layers[-10:]:
      if not isinstance(layer, keras.layers.BatchNormalization): #the batch normalization layer is untouched 
          layer.trainable = True

  model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.2*1e-4), metrics=metrics) #here we choose a different rate for Adam than default for a better convergence
  
  return model

#We named the model and also we trained with the class_weight

no_classes = 1
NO_EPOCHS = 10

model_16 = construct_model_16(no_classes,(92,103,None))

history = model_16.fit(X_train, y_train, epochs=NO_EPOCHS, validation_split=0.2,class_weight=class_weight)

score = model_16.evaluate(X_test, y_test)
print("Test accuracy:", score[1])

print("all scores:", score)


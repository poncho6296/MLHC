from sys import exit
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




# %%
data_dir='/home/jjuarezp/appended/appended_two/'
data_dir2='/home/jjuarezp/appended/'    
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

image_names = pd.DataFrame(train, columns=['image_id'])
image_names['eid_broad'] = image_names.image_id.str.rsplit(pat='.').str[0].astype(int)
#here we have the new df with the labels
image_names = image_names.merge(df_interest, how='left', on='eid_broad')
image_names=image_names[~image_names["image_id"].isin(list_error)]

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

from sklearn.utils import class_weight


weights = class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)

class_weight = {0: weights[0],
                1: weights[1]}




# %%
input = keras.Input(shape=X_train.shape[1:])
input = data_augmentation_complete(input)
x = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_1")(input) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer
x = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(x) # convolutional layer!
x = keras.layers.MaxPool2D()(x) # pooling layer
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


#Here we create the data frame from the test set and add probability and label
test_sal=df_interest.filter(items = y_test.index, axis=0)
test_sal["probability"]=model.predict(X_test)
test_sal["prediction"]=np.where(model.predict(X_test)>0.5,1,0)
test_sal.sort_values(by=["probability"],inplace=True,ascending=False)
#Here we compute TP,FP,TN,FN
condlist = [(test_sal.prediction==0 & (test_sal.has_disease==test_sal.prediction)), 
            (test_sal.prediction==0 & (test_sal.has_disease!=test_sal.prediction)),
           (test_sal.prediction==1 & (test_sal.has_disease==test_sal.prediction)),
           (test_sal.prediction==1 & (test_sal.has_disease!=test_sal.prediction))]
choicelist = ["TN","FN","TP","FP"]
test_sal["status"]=np.select(condlist, choicelist, 42)
status_to_filter=test_sal["status"].unique().tolist()
sample_images=[]

for s in status_to_filter:
    filt=test_sal[test_sal["status"]==s]
    
    
    if s=="TP" or s=="FP":
        sample_images.append(filt.head(1).index.values[0])
        
    elif s=="TN" or s=="FN":
         sample_images.append(filt.tail(1).index.values[0])  
#Here we compute the saliency map
def saliency_map(index,model,title):
    img=list_tensor[index].numpy()
    img = img.reshape((1, *img.shape))
    y_pred = model.predict(img)
    images = tf.Variable(img, dtype=float)
    
    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
        
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    ## normalize to range between 0 and 1
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    fig, axes = plt.subplots(1,2,figsize=(14,5))
    plt.title(title)
    axes[0].imshow(list_tensor[index])
    i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
    fig.colorbar(i)  
    fig.savefig(data_dir2+title+'.png')

i=0
for s in sample_images:
    saliency_map(s,model,status_to_filter[i])
    i+=1  

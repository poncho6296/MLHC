import pandas as pd
import numpy as np
import os
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
from sklearn import metrics
import seaborn as sns
from sklearn.linear_model import LogisticRegression

#Here we read the df with the labels
df_interest = pd.read_csv('/home/rymekabak/dataset_2images.csv')
#Create a subset of the df
df_interest['Diastolic blood pressure'] = df_interest[['Diastolic blood pressure, manual reading.0.0',
    'Diastolic blood pressure, automated reading.0.0']].sum(axis=1)

print(len(df_interest))
df_analysis = df_interest[['sex', 'Age when attended assessment centre.2.0',
        'Townsend deprivation index at recruitment.0.0', 'prevalent_disease', 'LDL direct.0.0',
               'Body mass index (BMI).0.0', 'Smoking status.0.0', 'Alcohol drinker status.0.0',
                     # 'Systolic blood pressure, manual reading.0.0', not included -> many NANs
                                   'Diastolic blood pressure',
                                       'Glycated haemoglobin (HbA1c).0.0',
                                           'incident_disease']]


#here we read the features extracted
vgg16_extracted_feature_list_np = np.load("embeddings_lowdim.npy")
features_extracted = pd.DataFrame(vgg16_extracted_feature_list_np)
print(len(features_extracted))

#here we gather all the information we have
df_final = pd.concat([df_analysis, features_extracted], axis=1)
df_final = df_final[~df_final.isnull().any(axis=1)]

X = df_final.loc[:, df_final.columns != 'incident_disease']
y = df_final['incident_disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify = y, random_state=511)


class_ratio = sum(y_train==0) / sum(y_train==1)
model_logreg_baseline = LogisticRegression(class_weight={0: 1, 1: class_ratio}).fit(X_train, y_train)
print('model has run')

predicted = model_logreg_baseline.predict(X_test)
predicted_probs = model_logreg_baseline.predict_proba(X_test)


print("Classifer Accuracy:", metrics.accuracy_score(y_test, predicted), "Classifer Precision:", metrics.precision_score(y_test, predicted),
                  "Classifer Recall:", metrics.recall_score(y_test, predicted), "Classifer F1:", metrics.f1_score(y_test, predicted),
                      "Classifer AUC:", metrics.roc_auc_score(y_test, predicted_probs[:, 1]))


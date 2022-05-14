from julia import Julia
Julia(sysimage='/home/gridsan/groups/IAI/images/2.0.0/julia-1.5.2/sys.so', compiled_modules = False)
from interpretableai import iai
import pyarrow
import numpy as np
import pandas as pd
from pathlib import Path
import time
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import xgboost as xgb
import math
from imblearn.over_sampling import SMOTE 
import statistics
from sklearn.model_selection import train_test_split
#import shap
def classifiers(method,train_X,train_y,max_dep,minbuc,n_fold,*args):
    if method=='XG Python':
        negative_examples=train_y.shape[0]-sum(train_y)
        positive_examples=sum(train_y)
        parameters = {'objective':['binary:logistic','binary:logitraw'],'max_depth':max_dep,
                      'n_estimators':n_tree,"eta":learn_param,"gamma":comp}
        XG = xgb.XGBClassifier(random_state=0,scale_pos_weight=math.sqrt(negative_examples/positive_examples),
                        eval_metric="auc",use_label_encoder=False,n_jobs=0)
        clf = GridSearchCV(XG, parameters,cv=n_fold,scoring="roc_auc",verbose=2)
        clf.fit(train_X,train_y)
        best_param=clf.best_params_
        clf=clf.best_estimator_
        AUC_is=roc_auc_score(train_y, clf.predict_proba(train_X)[:, 1])
        return best_param,clf,AUC_is
    else:
        return print("Not a valid method")
    best_param=grid.get_best_params()
    learner=grid.get_learner()
    AUC_is=grid.score(train_X, train_y, criterion='auc')   
    Accuracy_is=grid.score(train_X, train_y, criterion='misclassification')   
    AUC_os=grid.score(test_X, test_y, criterion='auc')   
    Accuracy_os=grid.score(test_X, test_y, criterion='misclassification')   
    return best_param,learner,AUC_is,Accuracy_is,AUC_os,Accuracy_os
def run_models(Algorithm,max_dep,minbuc,n_fold,n_tree,comp,learn_param,train_X,train_y):
    learners=[]
    best_param=[]
    models=[]
    data=[]
    for l in Algorithm:
        bp,learn,AUC_is=classifiers(l,train_X,train_y,max_dep,minbuc,n_fold,n_tree,comp,learn_param)
        print("The current algorithm is",l)
        print("\nBest Parameters:",bp)
        print("\nThe AUC in sample is", AUC_is)
        learners.append(l)
        best_param.append(bp)
        models.append(learn)
        data.append([l,bp,AUC_is])
    df = pd.DataFrame(data, columns=['Model', 'Best_parameters', 'AUC_In_Sample'])
    return(learn,best_param,models,df)
# Parameters to train
Algorithm=['XG Python']
max_depp=[4,5] #6 or 7
minbuc=[50,75,150,200]
n_fold=5
n_tree=[50,75,100,150]
comp=[0.01,0.001,0.1,0.0001]
learn_param=[0.001,0.01,0.1]


# Reading files
X=pd.read_csv("DB_research_BST_shared/MGH/Endovenous_Thermal_Ablation/EVA_complete.csv")
X.rename(columns={"othdvt": "target"},inplace=True)
X.drop(columns=['Race_sum'],inplace=True)
df_target=X[X["target"]==1]
df_not_target=X[X["target"]!=1]


initial_not_target=df_not_target
initial_target=df_target
learn_ASA=[]
best_ASA=[]
models_ASA=[]
#df_final_ASA=pd.DataFrame(columns=['Model', 'Best_parameters', 'AUC_In_Sample','Run','AUC_avg','AUC_std'])
df_final_ASA=pd.DataFrame(columns=['Model', 'Best_parameters', 'AUC_In_Sample','Run',"AUC_val"])

def data_set(df_target,df_not_target,seed=1):
    # Training set
    X_train_temp_not_target=df_not_target.sample(1000,random_state=seed)
    X_train_temp_target=df_target.sample(63,random_state=seed)
    X_train=pd.concat([X_train_temp_not_target, X_train_temp_target])
    y_train = X_train['target']
    X_train=X_train.drop(columns=['target'])
    return X_train,y_train

# Outsample data
seed=1
X_test_temp_not_target=df_not_target.sample(1000,random_state=seed)
initial_not_target=initial_not_target.drop(X_test_temp_not_target.index)
X_test_temp_target=df_target.sample(63,random_state=seed)
initial_target=initial_target.drop(X_test_temp_target.index)
X_test=pd.concat([X_test_temp_not_target, X_test_temp_target])
y_test = X_test['target']
X_test=X_test.drop(columns=['target'])

# Fill our array with all the data sets to train and test
comp_X_train=[]
comp_y_train=[]
comp_X_val=[]
comp_y_val=[]

X=pd.concat([initial_not_target, initial_target])
X.rename(columns={"othdvt": "target"},inplace=True)
y = X['target']
X.drop(columns=['target'],inplace=True)


for i in range(100):
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=i)
    comp_X_train.append(X_train)
    comp_y_train.append(y_train)
    comp_X_val.append(X_val)
    comp_y_val.append(y_val)
    
AUC_avg=[]
AUC_std=[]
for i in range(75,100):
    #AUC_model=[]    
    current_check=i
    #Train the model
    print("The current run is: ", current_check)
    l,bp,mod,df=run_models(Algorithm,max_depp,minbuc,n_fold,n_tree,comp,learn_param,comp_X_train[i],comp_y_train[i])
    learn_ASA.append(l)
    best_ASA.append(bp)
    models_ASA.append(mod)
    #for j in range(100):
    #    if current_check!=j:
    #        AUC_os=roc_auc_score(comp_y_train[j], l.predict_proba(comp_X_train[j])[:, 1])
    #        AUC_model.append(AUC_os)
    AUC_os=roc_auc_score(comp_y_val[i], l.predict_proba(comp_X_val[i])[:, 1])
    #AUC_model.append(AUC_os)
    df["AUC_val"]=AUC_os
    #df["AUC_avg"]=statistics.mean(AUC_model)
    #df["AUC_std"]=statistics.pstdev(AUC_model)
    df["Run"]=i
    df_final_ASA=pd.concat([df_final_ASA, df],ignore_index=True)
    df_final_ASA.to_csv('DB_research_BST_shared/MGH/Endovenous_Thermal_Ablation/Experiment1000XGBoost_final_4.csv', index=False)
best_model=df_final_ASA[df_final_ASA["AUC_val"]==max(df_final_ASA["AUC_val"])].index[0]
df_best= df_final_ASA.filter(items = [best_model], axis=0)
df_best["AUC_testing"]=roc_auc_score(y_test, learn_ASA[best_model].predict_proba(X_test)[:, 1])
df_best.to_csv('DB_research_BST_shared/MGH/Endovenous_Thermal_Ablation/Experiment1000XGBoost_best.csv', index=False)

#df_final_ASA.to_csv('DB_research_BST_shared/MGH/Endovenous_Thermal_Ablation/Experiment1000XGBoost_part3.csv', index=False) 

    
    

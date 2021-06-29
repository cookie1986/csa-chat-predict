'''
Main pipeline for analysing chatlogs scraped from PAN12 xml file.
'''


import os

import numpy as np
import pandas as pd





#### Files necessary for pre-processing original xml data

# set path of working dir
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/'
os.chdir(path)


# xml file containing chat data
with open('xml_files/pan12_trainingcorpus.xml', 'r', encoding='utf8') as f: 
    xml_data = f.read() 
    
    
# get predator IDs as a list
predators = open('xml_files/pan12_predator_id.txt','r').readlines()
predators = [id_.rstrip() for id_ in predators]


# get resampled chat IDs as a list
resampled = open('resampling/resampled_convo_id.csv','r').readlines()[1:]
resampled = [id_.rstrip() for id_ in resampled]



#### Pre-processing

# pre-process xml data
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/scripts/csa-chat-predict/python'
os.chdir(path)
from pan12_preprocessing import preprocess_xml

data = preprocess_xml(xml_data, predators, resampled)



#### Feature Engineering

# Subtractive Conditional Probability - Danescu-Nicelescu et al, 2012 
from accommodation_features import SubCondProb
bow_1 = SubCondProb(data)


# Add other features here......





#### Chat Type Classification

# add labels to feature vector
chat_type = data.groupby(['chat_id']).max(['speaker_type'])

chat_type['speaker_type'].value_counts()


# merge chat type labels with feature vector
# group other features here......
features = pd.merge(bow_1, chat_type, left_index=True, right_index=True)

# set X and y vars
y = np.array(features['speaker_type'])
X = np.array(features.drop(['speaker_type','msg_id'], axis=1).reset_index(drop=True))

# prep cls task
from sklearn.ensemble import RandomForestClassifier

clsfr = RandomForestClassifier(n_estimators=100,
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               min_weight_fraction_leaf=0.0,
                               max_features='auto',
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_impurity_split=None,
                               bootstrap=True,
                               oob_score=False,
                               n_jobs=-1,
                               random_state=123,
                               warm_start=False,
                               class_weight=None,
                               ccp_alpha=0.0,
                               max_samples=None)


# prep Kfold
from sklearn.model_selection import StratifiedKFold

cross_val = StratifiedKFold(n_splits=10,
                            shuffle=False,
                            random_state=None)


# store preds over all folds
all_preds = []

# store performance metrics per fold
from sklearn.metrics import confusion_matrix, classification_report
conf_metrices = []

reports = []

# run model
for train_index, test_index in cross_val.split(X, y):
    
    # fold counter
    fold_num = 1
    
    # set train and test regions
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # fit model
    clsfr.fit(X_train, y_train)
    
    # get predictions
    y_preds = clsfr.predict(X_test)
    
    all_preds.append([y_test, y_preds])
    
    # performance
    output = confusion_matrix(y_test, y_preds)
    conf_metrices.append(output)
    
    # classification report
    target_names = ['Non-Pred','Pred']
    report = classification_report(y_test, y_preds, target_names=target_names)
    reports.append(report)
    
    
    
    fold_num +=1
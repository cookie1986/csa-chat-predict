'''
Main pipeline for analysing chatlogs scraped from PAN12 xml file.
'''


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

## load classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


RandomForest = RandomForestClassifier(random_state=123, n_jobs=-1)
SupportVec = SVC(random_state=123)
NaiveBayes = GaussianNB()
LogRegr = LogisticRegression(random_state=123, n_jobs=-1)

models = [RandomForest, SupportVec, NaiveBayes, LogRegr]


## feature pre-processing
# add labels to feature vector
chat_type = data.groupby(['chat_id']).max(['speaker_type'])
chat_type['speaker_type'].value_counts()


# merge chat type labels with feature vector
features = pd.merge(bow_1, chat_type, left_index=True, right_index=True) # add other models here


# # export to CSV (if necessary)
# features.to_csv('C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/models/model1_features.csv')


# set target
y = np.array(features['speaker_type'])

# set predictors
X = features.drop(['speaker_type','msg_id'], axis=1).reset_index(drop=True)

# get feature names
X_names = X.columns

# convert to array and scale (not needed for random forest but shouldn't hurt)
from sklearn.preprocessing import StandardScaler
X = np.array(X) # for random forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # for SVC, NB, LR


# prep cross validation and feature importance
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

cross_val = StratifiedKFold(n_splits=10,
                            shuffle=False,
                            random_state=None)


# store preds over all models and folds
all_preds = []
# store performance metrics 
metrics = []
# store importance scores over all folds
rf_importances = pd.DataFrame()

# store performance metrics per fold
from sklearn.metrics import confusion_matrix, classification_report
conf_metrices = []

# fold counter
fold_num = 1

# run model
for train_index, test_index in cross_val.split(X, y):
    
    ## Splitting dataset
    
    # set train (scaled and unscaled) and test regions
    X_train, X_test = X[train_index], X[test_index]
    Xsc_train, Xsc_test = X_scaled[train_index], X_scaled[test_index]
    
    y_train, y_test = y[train_index], y[test_index]
    
    
    ## Scaled model fitting 
    
    # fit scaled models
    for model in models:
        
        # set model name
        if str(model).startswith('S'):
            model_name = 'svm'
        elif str(model).startswith('G'):
            model_name = 'nb'
        elif str(model).startswith('R'):
            model_name = 'rf'
        else:
            model_name = 'lr'
        
        # fit
        if model_name == 'rf':
            model.fit(X_train, y_train)
            
            importance = permutation_importance(model, 
                                                X_train, 
                                                y_train, 
                                                scoring='recall')
            importance_mean = importance.importances_mean
            
            
            # predict
            y_preds = model.predict(X_test)
        
        else:
            model.fit(Xsc_train, y_train)
            
            # predict
            y_preds = model.predict(Xsc_test)

        # add predictions to main predictions list
        all_preds.append([model_name, y_test, y_preds])
        
        # measure performance predicting predator chats
        conf = confusion_matrix(y_test, y_preds)
        
        precision = conf[1][1]/(conf[0][1] + conf[1][1])
        recall = conf[1][1]/(conf[1][0] + conf[1][1])
        f1 = 2*precision*recall / (precision+recall)
        
        metrics.append([model_name, precision, recall, f1])

    
    
    
    ##  Feature Importance (random forest only)
    
    # # get feature importances
    # feat_imp = RandomForest.feature_importances_
    
    # # SD of feature importances
    # std = np.std([tree.feature_importances_ for tree in RandomForest.estimators_],
    #           axis=0)
    
    # # get indices of top 10 features
    # indices = np.argsort(feat_imp)

    # # get feature importance scores
    # feat_impDF = pd.DataFrame(RandomForest.feature_importances_, 
    #                         index=X_names,  
    #                         columns=['importance']).sort_values('importance', 
    #                         ascending=False).reset_index(drop=False)
    
                                                                
    # # add to main importance DF
    # importances = importances.append(feat_impDF)


    # # plot the feature importances of the forest
    # plt.figure()
    # plt.title("Top 10 Features: Fold #%d"%fold_num)
    # plt.barh(range(10), importances[indices[-10:]], color='b', align='center')
    
    # # plt.barh(range(X_train.shape[1]), feat_imp[indices],
    # #        color="r", xerr=std[indices], align="center")
    # # If you want to define your own labels,
    # # change indices to a list of labels on the following line.
    # plt.yticks(range(X.shape[1]), indices)
    # plt.ylim([-1, X.shape[1]])
    # plt.show()                                           
    
    


    fold_num +=1
    



# change working dir
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/'
os.chdir(path)

# export metrics values
metricsDF = pd.DataFrame(metrics, columns=['model','Precision','Recall','F1'])
# metricsDF.to_csv('models/model1_metrics.csv')
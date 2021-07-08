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






#### Pre-processing xml data

# pre-process xml data
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/scripts/csa-chat-predict/python'
os.chdir(path)
from pan12_preprocessing import preprocess_xml, remove_n_msgs

data = preprocess_xml(xml_data, predators, resampled)

# trim messages from beginning of chat
data = remove_n_msgs(data, n=10, resample=True)





#### Feature Engineering

# baseline (BoW) features
from baselines import baseline_bow
bow_baseline = baseline_bow(data)




# Subtractive Conditional Probability - Danescu-Nicelescu-Mizil et al, 2012 
from accommodation_features import SubCondProb
bow_scp = SubCondProb(data, feature_sel = True)

# get list of column names
bow_scp_cols = bow_scp.columns

# add scp marker to each feature
bow_scp_cols = ["scp_"+ k for k in bow_scp_cols]
bow_scp = bow_scp.set_axis(bow_scp_cols, axis=1)


# Add other features here......



# Format final feature set
features = pd.merge(bow_baseline, 
                    bow_scp, 
                    left_index=True, 
                    right_index=True)



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

# combine
models = [RandomForest, SupportVec, NaiveBayes, LogRegr]


## feature pre-processing
# add target labels to feature vector
chat_type = data.groupby(['chat_id']).max(['speaker_type']).drop(['msg_id'], axis=1)
chat_type['speaker_type'].value_counts()


# merge chat type labels with feature vector
features = pd.merge(features, chat_type, 
                    left_index=True, 
                    right_index=True)


# export to CSV (if necessary)
# features.to_csv('C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/models/baseline_features.csv')


# set target
y = np.array(features['speaker_type'])

# set predictors
X = features.drop(['speaker_type'], axis=1).reset_index(drop=True)

# get feature names
X_names = X.columns

# convert to array and scale (not needed for random forest but shouldn't hurt)
from sklearn.preprocessing import StandardScaler
X = np.array(X)

# convert to z scores
scaler = StandardScaler()
X = scaler.fit_transform(X)



# prep cross validation and feature importance
from sklearn.model_selection import StratifiedKFold
# from sklearn.inspection import permutation_importance

cross_val = StratifiedKFold(n_splits=10,
                            shuffle=False,
                            random_state=None)


# store preds over all models and folds
all_preds = []
# store performance metrics 
metrics = []


# store performance metrics per fold
from sklearn.metrics import confusion_matrix
conf_metrices = []

# run model
for train_index, test_index in cross_val.split(X, y):
    
    ## Splitting dataset
    
    # set train (scaled and unscaled) and test regions
    X_train, X_test = X[train_index], X[test_index]
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
        model.fit(X_train, y_train)
            
        # predict
        y_preds = model.predict(X_test)
        

        # add predictions to main predictions list
        all_preds.append([model_name, y_test, y_preds])
        
        # measure performance on unseen data
        conf = confusion_matrix(y_test, y_preds)
        
        precision = conf[1][1]/(conf[0][1] + conf[1][1])
        recall = conf[1][1]/(conf[1][0] + conf[1][1])
        f1 = 2*precision*recall / (precision+recall)
        
        # add output to main list
        metrics.append([model_name, precision, recall, f1])



# change working dir
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/'
os.chdir(path)

# convert to DF
metricsDF = pd.DataFrame(metrics, columns=['model','Precision','Recall','F1'])

# get means over all folds
metrics_meanDF = metricsDF.groupby('model').mean()

# export
# metricsDF.to_csv('models/baseline_metrics.csv')




#### Feature importance

## load packages
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from collections import defaultdict
from sklearn.inspection import permutation_importance

# models to run feature importance analysis on
importance_models = [LogRegr, NaiveBayes]

# empty DF to store scores
importance_global = pd.DataFrame()


## permutation feature importance

# run model
for train_index, test_index in cross_val.split(X, y):
    
    ## Splitting dataset
    
    # set train (scaled and unscaled) and test regions
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    
    ## Permutation Feature Importance with Cluster Analysis
    
    # fit scaled models
    for model in importance_models:
        
        # set model name
        if str(model).startswith('L'):
            model_name = 'lr'
        else:
            model_name = 'nb'
        
        # hierarchical clustering between features
        corr = spearmanr(X_train).correlation
        corr_linkage = hierarchy.ward(corr) # ward's linkage
        
        # # plot dendrogram - useful for setting value for 't' (see below)
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # dendro = hierarchy.dendrogram(
        #     corr_linkage, labels=X_names.tolist(), ax=ax1, leaf_rotation=90)
        # dendro_idx = np.arange(0, len(dendro['ivl']))
        # ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
        # ax2.set_xticks(dendro_idx)
        # ax2.set_yticks(dendro_idx)
        # ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
        # ax2.set_yticklabels(dendro['ivl'])
        # fig.tight_layout()
        # plt.show()
        
        # specify clustering threshold
        cluster_ids = hierarchy.fcluster(corr_linkage, t=1.5, criterion='distance')
        
        # sort features into clusters
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        

        # map features to their corresponding cluster
        cluster_to_featureDF = pd.DataFrame({'feature': X_names,
                                            'cluster_id': cluster_ids})
        
        # take the first feature from each cluster
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
        
        # collate a list of cluster keys
        cluster_list = [k for k in cluster_id_to_feature_ids.keys()]
        
        # filter X values on selected features
        X_train_filtered = X_train[:, selected_features]
        X_test_filtered = X_test[:, selected_features]
        
        # fit model with reduced training features
        model.fit(X_train_filtered, y_train)
        
        # generate predictions 
        y_preds_filtered = model.predict(X_test_filtered)
        
        # perform permutation importance test
        importance = permutation_importance(model, 
                                            X_test_filtered, 
                                            y_test, 
                                            scoring='recall',
                                            n_jobs=-1,
                                            n_repeats=5,
                                            random_state=123)
        
        # get mean and std importances over n_repeats
        importance_mean = importance.importances_mean
        importances_std = importance.importances_std
        
        # build dataframe from mean cluster scores
        clusterDF = pd.DataFrame({'cluster_id':cluster_list,
                                  'importance':importance_mean,
                                  'importance_stdev': importances_std})
        
        # collate score for each feature, based on cluster ID
        feature_importance = pd.merge(cluster_to_featureDF, 
                                      clusterDF, on='cluster_id', how='inner')
        
        # add model label
        feature_importance['model'] = model_name
        
        # add to global DF
        importance_global = importance_global.append(feature_importance)


# average feature importance scores 
importance_feature_mean = importance_global.groupby(['model','feature']).mean()

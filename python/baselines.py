'''
Baseline estimator that uses PAN12 word frequency counts to predict chat type
'''

import os

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def baseline_bow(dataframe):
    
    
    # text cleaning 
    def msg_clean(msg):
        
        import re
        from num2words import num2words
        
        '''
        input: string
        
        performs cleaning
        
        returns: cleaned text
        '''
        
        # remove punctuation
        msg_clean = re.sub(r"[^\w\s\d]","", msg)
        
        
        # replace numbers with word alternatives
        msg_clean = [int(i) if i.isdigit() else i for i in msg_clean.split()]
        
        msg_clean = [num2words(i) if isinstance(i, int) 
                      else i for i in msg_clean]
        
        # remove tokens longer than 10 chars
        msg_clean = [x for x in msg_clean if len(x) <10]
        
        # remove common nonlinguistic tokens
        msg_clean = ' '.join(
            [x for x in msg_clean if "www" not in x and "http" not in x]).rstrip()
        
        return msg_clean
    
    
    # list to store cleaned chats
    corpus_list = []
    
    
    # loop through each chat
    for chat in dataframe.groupby(['chat_id']):
    
            # parse individual tuple elements
            chat_id = chat[0]
            chatlog = chat[1].reset_index()
            
            # # record chat type (1=pred, 0=np)
            # if sum(chatlog['speaker_type'])>0:
            #     chat_type = 1
            # else:
            #     chat_type = 0
            
            # fill nans
            chatlog['content'] = chatlog['content'].fillna('')
            
            # clean msg
            chatlog['content'] = chatlog['content'].apply(lambda x: msg_clean(x))
            
            
            
            # merge adjacent turns that share the same speaker
            chatlog['prev_speaker'] = chatlog[
                'speaker'].shift(1).mask(pd.isnull, chatlog['speaker'])
            chatlog['speaker_change'] = np.where(
                chatlog['speaker'] != chatlog['prev_speaker'], 'True','False')
            
            # create list to store speaker change keys
            speaker_changes = []    
            
            # start speaker change counter
            speaker_change_counter =1
            
            # loop through each message to record speaker changes
            for i in chatlog['speaker_change']:
                if i == 'False':
                    speaker_changes.append(speaker_change_counter)
                else:
                    speaker_change_counter+=1
                    speaker_changes.append(speaker_change_counter)
            
            # group into chunks
            chatlog['chunking'] = speaker_changes
            
            # join adjacent messages sharing the same speaker ID 
            chatlog['content'] = chatlog.groupby(
                ['chunking'])['content'].transform(lambda x: ' '.join(x))
            
            # drop duplicate rows based on message content and conversation chunk
            chatlog = chatlog.drop_duplicates(subset=['content','chunking'])
            
            # drop non-needed columns
            chatlog = chatlog[['speaker','content', 'speaker_type',
                               'time','chat_id']]
            
            # create global variable of all tokens in the chat
            tokens_global = " ".join([turn for turn in chatlog['content']])
            
            # remove stopwords
            stop_words_list = list(stopwords.words('english'))
            
            tokens_global = " ".join(
                [word for word in word_tokenize(
                    tokens_global) if word not in stop_words_list])
            
            # add to corpus list
            corpus_list.append([chat_id, tokens_global])
            
    
    
    #### Count Vectorize Global Tokens
    
    vectorizer = CountVectorizer(strip_accents='ascii', 
                                 decode_error='ignore', 
                                 stop_words='english')

    # isolate labels and tokens
    chat_id = [i[0] for i in corpus_list]
    chat_tokens = [i[1] for i in corpus_list]
    
    
    # fit vectorizer
    freq_vec = vectorizer.fit_transform(chat_tokens).toarray()
    words = vectorizer.vocabulary_
    
    # set as DF
    baselineDF = pd.DataFrame(freq_vec)
    baselineDF = baselineDF.set_axis(words, axis=1)
    baselineDF['chat_id'] = chat_id
    baselineDF = baselineDF.set_index('chat_id')
    
    # remove features appearing in less than 10% of corpus
    min_nonZero = int(len(baselineDF)*0.1)
    baselineDF = baselineDF.loc[:, (
        baselineDF.replace(0, np.nan).notnull().sum(axis=0) >= min_nonZero)]
    

        
    
    return baselineDF


















# #### Count Vectorize Global Tokens

# # isolate labels and tokens
# chat_labels = [i[0] for i in corpus_list]
# chat_tokens = [i[1] for i in corpus_list]


# # fit vectorizer
# freq_vec = vectorizer.fit_transform(chat_tokens).toarray()
# words = vectorizer.vocabulary_

# # set as DF
# baselineDF = pd.DataFrame(freq_vec)
# baselineDF['chat_type'] = chat_labels

# # remove features appearing in less than 10% of corpus
# min_nonZero = int(len(baselineDF)*0.1)
# baselineDF = baselineDF.loc[:, (baselineDF.replace(0, np.nan).notnull().sum(axis=0) >= min_nonZero)]

# # # export to CSV if needed
# # baseline.to_csv('C:/Users/Darren Cook/Documents/PhD Research/csa_chats/predicting_predators/models/baseline_features.csv')


# # set target
# y = np.array(baselineDF['chat_type'])

# # set predictors
# X = baselineDF.drop(['chat_type'], axis=1).reset_index(drop=True)
# X_names = X.columns
# X = np.array(X)



# #### Chat Type Classification

# # store performance metrics per fold
# from sklearn.metrics import confusion_matrix, classification_report
# conf_metrices = []

# # fold counter
# fold_num = 1

# # store preds over all folds
# all_preds = []
# # store classification reports over all folds
# reports = []
# # store importance scores over all folds
# importances = pd.DataFrame()


# # perform cross validation
# for train_index, test_index in cross_val.split(X, y):
    
#     ## Fitting Random Forest
    
#     # set train and test regions
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     # fit model
#     clsfr.fit(X_train, y_train)
    
#     # get predictions
#     y_preds = clsfr.predict(X_test)
#     all_preds.append([y_test, y_preds])
    
    
    
#     ##  Feature Importance
    
#     # get feature importances
#     feat_imp = clsfr.feature_importances_
    
#     # SD of feature importances
#     std = np.std([tree.feature_importances_ for tree in clsfr.estimators_],
#               axis=0)
    
#     # get indices of top 10 features
#     indices = np.argsort(feat_imp)


    
#     # get feature importance scores
#     feat_impDF = pd.DataFrame(clsfr.feature_importances_, 
#                             index=X_names,  
#                             columns=['importance']).sort_values('importance', 
#                             ascending=False).reset_index(drop=False)
    
                                                                
#     # add to main importance DF
#     importances = importances.append(feat_impDF)
    
    
#     ## Visualize feature importance
    
#     # plot the feature importances of the forest  
#     # plt.figure()
#     # plt.title("Top 10 Features: Fold #%d"%fold_num)
#     # plt.barh(range(10), importances[indices[-10:]], color='b', align='center')
    
#     # # plt.barh(range(X_train.shape[1]), feat_imp[indices],
#     # #        color="r", xerr=std[indices], align="center")
#     # # If you want to define your own labels,
#     # # change indices to a list of labels on the following line.
#     # plt.yticks(range(X.shape[1]), indices)
#     # plt.ylim([-1, X.shape[1]])
#     # plt.show()                                           
    
    
    
    
#     ## Prediction Performance
    
#     # performance
#     output = confusion_matrix(y_test, y_preds)
#     conf_metrices.append(output)
    
#     # classification report
#     target_names = ['Non-Pred','Pred']
#     report = classification_report(y_test, y_preds, target_names=target_names)
#     reports.append(report)
    
    
    
#     fold_num +=1
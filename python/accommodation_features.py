'''
Features based on Communication Accommodation Theory (Giles et al., 1991)

Models:
    Bag of Words models based on Subtractive Conditional Probabilities 
    (Danescu-Niiculescu-Mizil et al, 2012). SCP is comprised of three sub-models:
        - N-Grams (Uni, Bi, Trigrams)
     
'''

def SubCondProb(dataframe, 
                feature_sel = True):
    
    import numpy as np
    import pandas as pd
    
    
    
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
        
        msg_clean = ' '.join([num2words(i) if isinstance(i, int) 
                      else i for i in msg_clean]).rstrip()
        
        return msg_clean
    
    
    
    def ngramize(msg, ngram_len):
        
        from advertools import word_tokenize
        
        '''
        input: string
        
        converts to a list of n-gram tokens 
        
        returns: token set
        '''
        
        try:
            msg_tokens = word_tokenize(msg, ngram_len)
            msg_tokens = [i for sub in msg_tokens for i in sub]
        except:
            AttributeError
            msg_tokens = ''
        
        return msg_tokens
        
        
    
    # empty dict for SCP scores
    scp_vals_global = {}
    
    
    for chat in dataframe.groupby(['chat_id']):
        
        # parse individual tuple elements
        chat_id = chat[0]
        chatlog = chat[1].reset_index()
        
        
        # record chat type (1=pred, 0=np)
        if sum(chatlog['speaker_type'])>0:
            chat_type = 1
        else:
            chat_type = 0

        
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
        
        
        
        # if non-predatory chat, create pseudo-speaker roles
        if chat_type==0:
            
            speaker_ids = list(set(chatlog['speaker']))
            
            chatlog['speaker_type'] = np.where(chatlog['speaker']==speaker_ids[0], 1, 0)
        
        
        # store chat level scp scores 
        scp_vals_local = {}
        
        # number of n-grams
        for ngram_size in range(1,4):

            ## pre-process msg content for SCP analysis
            
            # duplicate df
            chatlog_scp = chatlog.copy(deep=True)
            
            # parse msg into n-grams
            chatlog_scp['content'] = chatlog_scp['content'].apply(
                lambda x: ngramize(x, ngram_size))
            
            # take set of tokens in msg
            chatlog_scp['content'] = chatlog_scp['content'].apply(
                lambda x: list(set(x)))
            
                      
            # compare each turn with the previous
            chatlog_scp['previous'] = chatlog_scp['content'].shift().fillna('')
            
            # get intersection between each turn and prior
            chatlog_scp['shared'] = chatlog_scp.apply(
                lambda x: np.intersect1d(x.content, x.previous), axis=1)
            
            
            ## (eq.1) s1 probability per ngram
            
            # filter on speaker 1 msgs
            s1_msgs = chatlog_scp[chatlog_scp['speaker_type'] ==1]
                                               
            # collate all n-grams spoken by speaker 1 into one list
            s1_tokens = [token for turn in s1_msgs['content'] for token in turn]
            
            # create counter object of each n-gram
            from collections import Counter
            s1_ngram_count = Counter(token for token in s1_tokens)
            
            # divide each value by the number of IR turns
            for token, counter in s1_ngram_count.items():
                s1_ngram_count[token] /= len(s1_msgs)

            
            ## (eq.2) conditional probability per ngram given prior usage by s0
            
            # collate shared n-grams into one list
            s1_shared_set = [token for turn in s1_msgs['shared'] for token in turn]
            
            # create counter object of each n-gram
            shared_counter = Counter(token for token in s1_shared_set)
            
            # get list of s0 tokens
            s0_tokens = [token for turn in s1_msgs['previous'] for token in turn]
            # create counter
            s0_ngram_count = Counter(token for token in s0_tokens)
            
            # get final value for minuend
            ngram_conditional = {k : v / s0_ngram_count[k] for k, v in shared_counter.items() if k in s0_ngram_count}  
            
            # identify non-converged S1 tokens
            tokenDiff = list(set(s1_tokens).difference(s1_shared_set))
            
            # create dictionary where the conditional probability is zero
            divergedTokens = dict.fromkeys(tokenDiff, float(0))
            ngram_conditional.update(divergedTokens)
            
            
            ## subtractive conditional probability scores (SCP = eq.2 - eq.1)
            
            # subtract eq1 from eq2 to get final scp score per ngram
            scp_scores = {k : v - s1_ngram_count[k] for k, v in ngram_conditional.items() if k in s1_ngram_count}
            
            # add scp scores to local dict
            scp_vals_local.update(scp_scores)
            
            
        # add (chat level) local SCP scores to main (global) dict
        scp_vals_global.update({chat_id: scp_vals_local})
            
     
    ## feature selection    
    
    # format scp features as a matrix
    scp_features = pd.DataFrame.from_dict(scp_vals_global, orient='index').fillna(0)
    

    # perform simple feature selection
    if feature_sel == True:
        # remove features appearing in less than 10% of corpus
        min_nonZero = int(len(scp_features)*0.1)
        scp_features = scp_features.loc[:, (scp_features.replace(0, np.nan).notnull().sum(axis=0) >= min_nonZero)]
    

    return scp_features
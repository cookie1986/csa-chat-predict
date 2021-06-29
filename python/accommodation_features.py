'''
Features based on Communication Accommodation Theory (Giles et al., 1991)

Models:
    Bag of Words models based on Subtractive Conditional Probabilities 
    (Danescu-Niiculescu-Mizil et al, 2012). SCP is comprised of three sub-models:
        - N-Grams (Uni, Bi, Trigrams)
     
'''

def SubCondProb(dataframe):
    
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
        chatlog = chat[1]
        
        # clean msg
        chatlog['content'] = chatlog['content'].apply(lambda x: msg_clean(x))
        
        
        
        # merge adjacent turns that share the same speaker
        chatlog['prev_speaker'] = chatlog['speaker'].shift(1).mask(pd.isnull, chatlog['speaker'])
        chatlog['speaker_change'] = np.where(chatlog['speaker'] != chatlog['prev_speaker'], 'True','False')
        
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
        chatlog['content'] = chatlog.groupby(['chunking'])['content'].transform(lambda x: ' '.join(x))
        
        # drop duplicate rows based on message content and conversation chunk
        chatlog = chatlog.drop_duplicates(subset=['content','chunking'])
        
        # drop non-needed columns
        chatlog = chatlog[['speaker','content', 'speaker_type','time','chat_id']]
        
        
        
        # parse msg into n-grams
        for ngram_size in range(2,3):
            chatlog['content'] = chatlog['content'].apply(
                lambda x: ngramize(x, ngram_size))
            
            # take set of tokens in msg
            chatlog['content'] = chatlog['content'].apply(
                lambda x: list(set(x)))
            
                      
            # compare each turn with the previous
            chatlog['previous'] = chatlog['content'].shift().fillna('')
            
            # get intersection between each turn and prior
            chatlog['shared'] = chatlog.apply(
                lambda x: np.intersect1d(x.content, x.previous), axis=1)
            
            
            # # get the token set of IR speech
            # ir = chatlog[chatlog['speaker'] ==1]
            # # collate all n-grams into one list
            # ir_turn_set = [token for turn in ir['content'] for token in turn]
            # # create counter object of each n-gram
            # subtrahend = Counter(token for token in ir_turn_set)
            # # divide each value by the number of IR turns
            # for token, counter in subtrahend.items():
            #     subtrahend[token] /= len(ir)
            

            
        
    
    return chatlog
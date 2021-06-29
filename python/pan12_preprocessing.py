

import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup



def preprocess_xml(xml_file, 
                     predator_names,
                     resampled_list):
    
    '''
    Extract chat data from xml file 
    
    returns: a dataframe with formatted chat data
    
    '''

    # replace html entities in messages
    def html_replace (string):
        string = re.sub(r"&apos","'", string)
        string = re.sub(r"&quot",'"', string)
        string = re.sub(r"&amp"," and ", string)
        
        return string
    
    
    # empty dataframe to store chats
    main_chatDF = pd.DataFrame()


    # parse xml file
    soup = BeautifulSoup(xml_file, "html.parser") 


    # isolate chats
    chats = soup.find_all('conversation')


    # loop through each chat
    for chat in chats:
        
        # get chat ID
        chat_id = ''.join(list(chat.attrs.values()))
        
        # check if chat in resampled list
        if chat_id not in resampled_list:
            continue
        
        # extract messages
        chatlog = pd.DataFrame()
        messages = chat.find_all('message')
        
        # loop through messages
        for msg in messages:
            speaker_id = msg.find('author')
            speaker = speaker_id.text
            
            if speaker in predator_names:
                speaker_type = 1
            else:
                speaker_type = 0
            
            time_ref = msg.find('time')
            time = time_ref.text
            
            content_ref = msg.find('text')
            content = content_ref.text
            
            chatlog=chatlog.append({'speaker': speaker,
                                    'speaker_type': speaker_type,
                                    'time': time,
                                    'content': content}, ignore_index=True)
        
        
        # skip any chats with more/less than 2 speakers
        if len(set(chatlog['speaker'])) != 2:
            continue
        
        
        # set all text to lowercase
        chatlog['content'] = chatlog['content'].apply(lambda x: x.lower())
        
        
        # remove any automated messages
        auto_msgs = ['official messages from omegle will not be sent',
                     'now chatting with a random stranger. Say hi']
        chatlog = chatlog[~chatlog['content'].isin(auto_msgs)]

    
        # ignore any chats less than 15 messages long
        if len(chatlog) <15:
            continue

        
        # drop NaN rows
        chatlog = chatlog.dropna()

        
        # replace html entities
        chatlog['content'] = chatlog['content'].apply(lambda x: html_replace(x))
        
        # add chat id to dataframe
        chatlog['chat_id'] = chat_id
        
        
        # add chat to main dataframe
        main_chatDF = main_chatDF.append(chatlog)
        
        print(chat_id + " has been successfully processed")
        
        
    return main_chatDF
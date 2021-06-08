

import os
import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup



# set path
path = 'C:/Users/Darren Cook/Documents/PhD Research/csa_chats/'
os.chdir(path)

# open xml file with chats
with open('xml_files/pan12_trainingcorpus.xml', 'r', encoding='utf8') as f: 
    data = f.read() 

# get predator IDs
pred_file = open('xml_files/pan12_predator_id.txt','r')
pred_file = pred_file.readlines()
pred_file = [id_.rstrip() for id_ in pred_file]

# get resampled chats
resampled = open('resampling/resampled_convo_id.csv','r')
resampled = resampled.readlines()[1:]
resampled = [id_.rstrip() for id_ in resampled]

# function to replace html entities in messages
def html_replace (string):
    string = re.sub(r"&apos","'", string)
    string = re.sub(r"&quot",'"', string)
    string = re.sub(r"&amp"," and ", string)
    
    return string

# parse xml file
soup = BeautifulSoup(data, "html.parser") 

# isolate chats
chats = soup.find_all('conversation')

# empty repository to store processed chats
chats_repos = []



for chat in chats:
    
    # get chat ID
    chat_id = ''.join(list(chat.attrs.values()))
    
    # check if chat in resampled list
    if chat_id not in resampled:
        continue
    
    # extract messages
    chatlog = pd.DataFrame()
    messages = chat.find_all('message')

    for msg in messages:
        speaker_id = msg.find('author')
        speaker = speaker_id.text
        
        time_ref = msg.find('time')
        time = time_ref.text
        
        content_ref = msg.find('text')
        content = content_ref.text
        
        chatlog=chatlog.append({'speaker': speaker,
                                'time': time,
                                'content': content}, ignore_index=True)
    
    # skip any chats with more/less than 2 speakers
    if len(set(chatlog['speaker'])) != 2:
        continue
    
    # set all text to lowercase
    chatlog['content'] = chatlog['content'].apply(lambda x: x.lower())
    
    # remove any automated messages (common in Omegle chats)
    auto_msgs = ['official messages from omegle will not be sent',
                 'now chatting with a random stranger. Say hi']
    chatlog = chatlog[~chatlog['content'].isin(auto_msgs)]

    # ignore any chats less than 15 messages long
    if len(chatlog) <15:
        continue
    
    # check if speaker ID is predator (1=predator)
    speakerTags = list(set(chatlog['speaker']))
    
    if len(np.intersect1d(speakerTags, pred_file)) >0:
        chatType = 1
    else:
        chatType = 0
    
    # drop NaN rows
    chatlog = chatlog.dropna()
    
    # replace html entities
    chatlog['content'] = chatlog['content'].apply(lambda x: html_replace(x))
    
    # export chat as csv
    chatlog.to_csv('pan12_dataset/'+chat_id+'.csv', index=False)
    
    # add to repository
    chats_repos.append([chat_id, chatType, speakerTags])
    
    print(chat_id)
    
# export chat repos
metadata = pd.DataFrame(chats_repos, 
                        columns=['chat_id','chat_type','speaker_id'])
metadata.to_csv('pan12_metadata/pan12_metadata.csv', index=False)

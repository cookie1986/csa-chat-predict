'''
version 8 updates: removed consecutive turn merges to bring pipeline in line
with Panagiotis' density measure (where turns are not merged)
'''

import os
import ast
import re
import glob
import numpy as np
import pandas as pd

# set working dir
os.chdir("C:/Users/Darren Cook/Documents/PhD Research/csa_chats/data/pj_chats")


# Get Speaker Labels

# load list of speaker names
speakers = pd.read_csv('meta/speaker_labels.csv')
# get decoy names
decoy = speakers['ROLE'] == 'DECOY'
decoy_names = speakers[decoy]['USERNAMES']
decoy_list = list(set(decoy_names.values.tolist()))
# get suspect names
suspect = speakers['ROLE'] == 'SUSP'
suspect_names = speakers[suspect]['USERNAMES']
suspect_list = list(set(suspect_names.values.tolist()))


# Get Emoticons Dictionary

# read file
unicode_emot = open ("C:/Users/Darren Cook/Documents/PhD Research/csa_chats/resources/emoticons_v2.txt", 
                     "r", encoding='utf-8')
emoticons = unicode_emot.read()
unicode_emot.close()
# store as dict
emoticons = ast.literal_eval(emoticons)


# loop through each chat
for file in glob.glob('pjchat_43.txt'):
    # set up messages dataframe
    messages = pd.DataFrame()
    # get filename for chat log
    filename = file[:-4]
    # open chat log as text file
    f = open(file, 'r', encoding='utf-8-sig')
    original_text = f.read()
    f.close()
    
    
    # Basic Formatting
    
    # split text on new line char
    original_text = original_text.split('\n')
    # remove empty rows
    original_text = [row for row in original_text if len(row.strip()) >0]  
    # remove annotations with keywords
    original_text = [row for row in original_text if not re.compile(
        r'dateline|link |nsfw|perverted', flags=re.I).search(row)]
    # add row index 
    chat_log = []
    row_counter=1
    for row in original_text:
        chat_log.append([row_counter, row])
        row_counter+=1
        
    
    # For Yahoo Instant Messaging messages
    '''
    MSG Format: "Name (01:01:00 AM): Message"
    '''
    # create copy of original chatlog
    yahoo_copy = chat_log.copy()
    # merge multiline rows
    yahoo_copy = [[row[0],' '.join([row[1], nxt[1]])] if not 
                re.compile(r'\([\d\s:\/]{2,}[APM]{,2}\):').search(nxt[1]) else
                row for row, nxt in zip(yahoo_copy, yahoo_copy[1:])]
    # re-add last row which gets removed in error in the above line
    yahoo_copy.append(chat_log[-1])
    # remove duplicate rows
    yahoo_copy = [row for row in yahoo_copy if 
                  re.compile(r'\([\d\s:\/]{2,}[APM]{,2}\):').search(row[1])]
    # isolate speaker and message content
    yahoo_msgs = []
    for row in yahoo_copy:
        m = re.search(r'^(\S[^(]*?)\s*\([^()]*\)\s*:\s*(.+)', row[1], 
                      flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            yahoo_msgs.append(newrow)
    # add to main data-frame
    messages = messages.append(yahoo_msgs)
    
    
    # For MEETME.COM messages
    '''
    MSG Format: "Name - Message"
    '''
    # create copy of original chatlog
    meetme_copy = chat_log.copy()
    # isolate speaker and message content
    meetme_msgs = []
    for row in meetme_copy:
        m = re.search(r'^(\S[^():]*?)\s+-\s+(.+)', row[1], flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            meetme_msgs.append(newrow)
    # add to main data-frame
    messages = messages.append(meetme_msgs)
    
    
    # For SMS messages
    '''
    MSG Format: Name: Message Time
    '''
    # create copy of original chatlog
    sms_copy = chat_log.copy()   
    # isolate speaker and message content
    sms_msgs = []
    for row in sms_copy:
        m = re.search(r'^(\S[^()]*?)\s*:\s*(.+\d{1,2}:\d{2}\s?[APM]{2}$)', 
                      row[1], flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            sms_msgs.append(newrow)
    # add to main data-frame
    messages = messages.append(sms_msgs)
    
    
    # For Chat Room #1 messages
    '''
    MSG Format: Date Time: {Name} Message
    '''
    #  create copy of original chatlog
    chatroom1_copy = chat_log.copy()    
    # isolate speaker and message content
    chatroom1_msgs = []
    for row in chatroom1_copy:
        m = re.search(r'^(\S[\d\s\/:APM]*)\s*(\{[^}]*\})\s?(.+)', row[1], 
                      flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            chatroom1_msgs.append(newrow)
    # create index of list items minus timing label (index pos=1)
    index=[0,2,3]
    # remove timing label
    chatroom1_msgs=[[row[i] for i in index] for row in chatroom1_msgs]
    # remove curly brackets from speaker ID
    chatroom1_msgs = [[row[0], re.sub(r'[{}]', '', row[1]), row[2]] for row in
                      chatroom1_msgs]
    # add to main data-frame
    messages = messages.append(chatroom1_msgs)
    
    
    # For Chat Room #2 messages
    '''
    MSG Format: Date Time Name: Message
    '''
    # create copy of original chatlog
    chatroom2_copy = chat_log.copy()    
    # isolate speaker and message content
    chatroom2_msgs = []
    for row in chatroom2_copy:
        m = re.search(r'^(\S[\d\s:]+[APM]{2})\s*([^:]*):\s?(.+)', row[1], 
                      flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            chatroom2_msgs.append(newrow)
    # create index of list items minus timing label (index pos=1)
    index=[0,2,3]
    # remove timing label
    chatroom2_msgs=[[row[i] for i in index] for row in chatroom2_msgs]
    # add to main data-frame
    messages = messages.append(chatroom2_msgs)
    
    
    # For Chat Room #3 messages
    '''
    MSG Format: [Time] Name: Message

    '''
    # create copy of original chatlog
    chatroom3_copy = chat_log.copy()
    # isolate speaker and message content
    chatroom3_msgs = []
    for row in chatroom3_copy:
        m = re.search(r'^(\[\d{,2}:\d{2}\])\s(\w+):\s(.+)', row[1], flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            chatroom3_msgs.append(newrow)
    # create index of list items minus timing label (timing pos=1)
    index=[0,2,3]
    # remove timing label
    chatroom3_msgs=[[row[i] for i in index] for row in chatroom3_msgs]
    # add to main data-frame
    messages = messages.append(chatroom3_msgs)
    
    
    # For AOL Chat messages
    '''
    MSG Format: Name [Time]: Message
    '''
    # create copy of original chatlog
    aol_copy = chat_log.copy()   
    # isolate speaker and message content
    aol_msgs = []
    for row in aol_copy:
        m = re.search(r'^([\S ]+)\s*\[[^\[\]]+\]\s*:\s*(.+)', row[1], 
                      flags=re.I)
        if m:
            newrow = [row[0]]
            newrow.extend([z for z in m.groups()])
            aol_msgs.append(newrow)
    # add AOL messages to main message DF
    messages = messages.append(aol_msgs)
    
    
    # Construct Data-Frame
    
    # reorder messages to reflect their original order
    messages = messages.sort_values(by=0).reset_index(drop=True)
    # drop original index column
    messages = messages.drop(0, axis=1)
    # add column names
    messages.columns=['speaker','message']
    # replace emoticons with words
    messages['message'] = messages['message'].apply(
        lambda x: str(" ".join(emoticons.get(ele, ele) for ele in x.split())))
    # remove commentary
    messages['message'] = messages['message'].apply(
        lambda x: re.sub(r'\([^\:;)]+\)', '', x))
    # remove any blank rows
    messages['message'].replace('', np.nan, inplace=True)
    messages.dropna(subset=['message'], inplace=True)
    
    
    # Standardize Data-Frame
    
    # check if speaker label exists in either suspect or decoy list
    speaker_roles = []
    for row in messages['speaker']:
        if row in decoy_list:
            speaker_roles.append('decoy')
        elif row in suspect_list:
            speaker_roles.append('suspect')
        else:
            speaker_roles.append('delete')
    messages['speaker_role'] = speaker_roles
    # remove any rows marked for deletion
    messages = messages[messages.speaker_role != 'delete']
    
    
    # # Merge adjacent same speaker messages
    
    # # highlight where speaker changes occur
    # messages['prev_speaker'] = messages['speaker_role'].shift(1).mask(
    #     pd.isnull, messages['speaker_role'])
    # messages['speaker_change'] = np.where(messages['speaker_role']
    #                                       != messages['prev_speaker'], 
    #                                       'True','False')
    # # list to store speaker change keys
    # speaker_changes = []    
    # # start speaker change counter
    # speaker_change_counter =1
    # # loop through each message to record speaker changes
    # for i in messages['speaker_change']:
    #     if i == 'False':
    #         speaker_changes.append(speaker_change_counter)
    #     else:
    #         speaker_change_counter+=1
    #         speaker_changes.append(speaker_change_counter)
    # messages['chunking'] = speaker_changes
    # # join adjacent messages sharing the same speaker ID 
    # messages['message'] = messages.groupby(
    #     ['chunking'])['message'].transform(lambda x: ' '.join(x))
    # # drop duplicate rows based on message content and conversation chunk
    # messages = messages.drop_duplicates(subset=['message','chunking'])
    
    # drop non-needed columns
    messages = messages[['speaker_role','message']]
    
    
    # export
    messages.to_csv('C:/Users/Darren Cook/Documents/PhD Research/csa_chats/data/pj_chats/cleaned/'+filename+'.csv', index=False)
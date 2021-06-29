'''
Main pipeline for analysing chatlogs scraped from PAN12 xml file.
'''


import os

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

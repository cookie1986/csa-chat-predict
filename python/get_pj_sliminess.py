import re
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup

# load slimyness
data = pd.read_csv('C:/Users/Darren Cook/Documents/PhD Research/chat_logs/notes/chatlog_url.csv')

# empty dataframe to store slimyness scores
chat_log_slimy = pd.DataFrame(columns=['chat_ref','slimyness','votes'])

# loop through each url in list
for ind in data.index: 
     ref = data['chat_ref'][ind]
     print(ref)
     chat = data['url'][ind]
     result = requests.get(chat)
     src = result.content
     page_soup = soup(src, "html.parser")
     
     # locate slimyness ratings from html
     slimy_scale = page_soup.find('big', attrs={'id':'reallyBig'})
     slimy_scale = str(slimy_scale.text)
     slimyness = re.search(r'\d+.\d+', slimy_scale).group()
     total_votes = re.search(r'\d+$', slimy_scale).group(0)
     
     # oranize into dictionary
     chatDict = {'chat_ref': ref,
                 'slimyness': slimyness,
                 'votes': total_votes}
     # add dictionary to main DF
     chat_log_slimy = chat_log_slimy.append(chatDict, ignore_index=True)

# export ratings
chat_log_slimy.to_csv('C:/Users/Darren Cook/Documents/PhD Research/chat_logs/notes/slimyness_ratings.csv', index=False)
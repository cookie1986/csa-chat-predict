import re
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup

# load slimyness
data = pd.read_csv('C:/Users/Darren Cook/Documents/PhD Research/chat_logs/notes/chatlog_url.csv')


# loop through each url in list
for ind in data.index: 
     ref = data['chat_ref'][ind]
     print(ref)
     chat = data['url'][ind]
     result = requests.get(chat)
     src = result.content
     page_soup = soup(src, "html.parser")
     
     # locate chatlog from html
     chat_log = page_soup.find('div', attrs={'class':'chatLog'})
     chat_log = str(chat_log.text)
     
     # write messages to a text file
     text_file = open("C:/Users/Darren Cook/Documents/PhD Research/chat_logs_data/chatlogs_full/pjchat_"+str(ref)+".txt", "wt", encoding="utf-8")
     n = text_file.write(chat_log)
     text_file.close()
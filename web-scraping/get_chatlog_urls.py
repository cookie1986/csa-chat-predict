import requests
from bs4 import BeautifulSoup as soup
import pandas as pd

# PJ chatlog archive
url_main = 'http://perverted-justice.com/?archive=byName'

# get list of chat URLs
req_main = requests.get(url_main)
main_soup = soup(req_main.text, "html.parser")
# list to store URLs
url_link = []
for link in main_soup.find_all('a'):
    url_link.append(str(link.get('href')))
# filter list to only those containing chatlogs
url_link = list(set(['http://perverted-justice.com'+i+'&nocomm=true' for i in url_link if i.startswith('./?archive=')]))
# export chatlog list
urlDF = pd.DataFrame(data=url_link)
urlDF.to_csv('C:/Users/Darren Cook/Documents/PhD Research/chat_logs/notes/chatlog_url.csv')

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup as soup

# empty dataframe to store slimyness scores
usernames = []

# load url page
result = requests.get('http://perverted-justice.com/?archive=byName')
src = result.content
page_soup = soup(src, "html.parser")
     
# locate usernames
names = page_soup.findAll('a', attrs={'id':'pedoLink'})

# isolate usernames
for i in names:
    i = str(i.text)
    usernames.append(i)

usersDF = pd.DataFrame(usernames, columns=['names'])

usersDF.to_csv('C:/Users/Darren Cook/Documents/PhD Research/chat_logs/results/speakers_soup.csv', index=False)

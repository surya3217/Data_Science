'''
Write a Python code to Scrap data and download data from given url using Beautiful Soup.
      url = "https://bidplus.gem.gov.in/bidlists"
      Make list and append given data:
          1. BID NO
'''

from bs4 import BeautifulSoup
import pandas as pd
import requests

url = "https://bidplus.gem.gov.in/bidlists?bidlists"
source = requests.get(url).text    ## requesting to the url

soup = BeautifulSoup(source,"lxml")  ## parsing the data with lxml

data= soup.find(id= "pagi_content")   ## finding required data using 'id'
print (data)

bid= data.findAll('a')   ## finding data using 'a' tag

bid_num= []
for i in range(len(bid)):    ## run loop alternetively to store bid no.
    if i%2==0:
        bid_num.append(bid[i].text) 

print(bid_num)






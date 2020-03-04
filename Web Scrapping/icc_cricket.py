
"""
Code Challenge
  Name: 
    Webscrapping ICC Cricket Page
  Filename: 
    icccricket.py
  Problem Statement:
    Write a Python code to Scrap data from ICC Ranking's 
    page and get the ranking table for ODI's (Men). 
    Create a DataFrame using pandas to store the information.
  Hint: 
    #https://www.icc-cricket.com/rankings/mens/team-rankings/odi 
    
    https://www.icc-cricket.com/rankings/mens/team-rankings/t20i
    #https://www.icc-cricket.com/rankings/mens/team-rankings/test
"""

from bs4 import BeautifulSoup
import requests

url= [ 'https://www.icc-cricket.com/rankings/mens/team-rankings/odi',
       'https://www.icc-cricket.com/rankings/mens/team-rankings/t20i',
       ]

file_name= ["icc_odi.csv","icc_t20.csv"]

for index, link in enumerate(url):
    
    #specify the url
    icc = link
    source = requests.get(icc).text
    
    soup = BeautifulSoup(source,"lxml")
    #print (soup.prettify())
    #all_tables=soup.find_all('table')
    
    right_table= soup.find('table', class_='table')  ## use class underscore
    print (right_table.tbody)
    
    #Generate lists
    pos=[]  ## Rank
    A=[]    ## Team
    B=[]    ## Weighted
    C=[]    ## Points
    D=[]    ## Rating
    
    for row in right_table.tbody.findAll('tr'):
        cells = row.findAll('td')
        pos.append(cells[0].text.strip())
        A.append(cells[1].text.strip())
        B.append(cells[2].text.strip())
        C.append(cells[3].text.strip())
        D.append(cells[4].text.strip())
    
    
    from collections import OrderedDict
    
    col_names = ["Rank","Team","Weighted Matches","Points","Rating"]
    col_data = OrderedDict(zip(col_names,[pos,A,B,C,D]))
    
    
    # If you want to store the data in a csv file
    import pandas as pd
    df2 = pd.DataFrame(col_data) 
    df2.to_csv(file_name[index])

print('All Done, You are Awesome')

##################################################################################


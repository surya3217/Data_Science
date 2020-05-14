"""
pip install or conda install the libraries
Create Mongo Atlas Account and Verify it via the email
https://cloud.mongodb.com

Create a cluster
Create Database
Give Database Access
Give Network Access
Create Collection 
Then Connect to get the python 3.4 sample code

# AWS N. Virginia Free Tier - MO Sandbox 
# Database Access - database access as Admin for new user with Read and Write Access
# Autopasssword generation should be used
# Security/Network Access - IP Whitelisting 0.0.0.0/0 ( Access from anywhere)
"""

'''
Insert the data to Mongodb Cloud
'''

import pymongo
import pandas as pd

cluster = pymongo.MongoClient("mongodb+srv://surajkumar9427:surya123@cluster0-xk6lv.mongodb.net/test?retryWrites=true&w=majority")
#client = pymongo.MongoClient("mongodb+srv://sylvesterferns:cooler%21%40%23123@cluster0-t6mku.mongodb.net/test?retryWrites=true&w=majority")
client= cluster['forsk_db']
collection= client['forsk_coll']


def add_employee(idd, first, last, pay):
    '''Adding the unique employees into the collection'''
    
    unique_employee = collection.find_one({"_id":idd})  ## finding the existing idd 
    if unique_employee:
        return "Employee already exists"
    else:
        collection.insert_one(         ## inserting the data
                {
                "_id" : idd,
                "first" : first,
                "last" : last,
                "pay" : pay
                })
        return "Employee added successfully"


def fetch_all_employee():
    '''reading the data from the mongodb cloud's collection'''
    user = collection.find()    ## finding all data from the collection
    for i in user:
        print (i)


collection.drop()

##Insert data in collection
add_employee(10,'Sylvester', 'Fernandes', 50000)
add_employee(2,'Yogendra', 'Singh', 70000)
add_employee(3,'Rohit', 'Mishra', 30000)
add_employee(4,'Kunal', 'Vaid', 30000)
add_employee(12,'Devendra', 'Shekhawat', 30000)

fetch_all_employee()

################################################################################

'''
Insert DataFrame to Mongodb Cloud using 'to_dict' Function
'''

import pymongo
import pandas as pd

cluster = pymongo.MongoClient("mongodb+srv://surajkumar9427:surya123@cluster0-xk6lv.mongodb.net/test?retryWrites=true&w=majority")
#client = pymongo.MongoClient("mongodb+srv://sylvesterferns:cooler%21%40%23123@cluster0-t6mku.mongodb.net/test?retryWrites=true&w=majority")
client= cluster['icc_db']
collection= client['odi']   ## creates new collection if doesn't exist

"reading data file"
df = pd.read_csv("Saved csv/icc_odi.csv") #csv file which you want to import
df= df.iloc[:,1:]
print(df)

"Inserting data to the mongodb collection"
records_ = df.to_dict(orient = 'records')
result = collection.insert_many(records_ )

"reading the data from the collection"
def fetch_all_employee():
    user = collection.find()    ## finding all data from the collection
    for i in user:
        print (i)

"calling the function"
fetch_all_employee()

###############################################################################

import pymongo
import pandas as pd

cluster = pymongo.MongoClient("mongodb+srv://surajkumar9427:surya123@cluster0-xk6lv.mongodb.net/test?retryWrites=true&w=majority")
client= cluster['icc_db']
collection= client['t20']   ## creates new collection if doesn't exist

"reading data file"
df = pd.read_csv("Saved csv/icc_t20.csv") #csv file which you want to import
df= df.iloc[:,1:]
print(df)

"Inserting data to the mongodb collection"
records_ = df.to_dict(orient = 'records')
result = collection.insert_many(records_ )

"reading the data from the collection"
def fetch_all_employee():
    user = collection.find()    ## finding all data from the collection
    for i in user:
        print (i)

"calling the function"
fetch_all_employee()






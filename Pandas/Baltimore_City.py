"""
Code Challenge
  Name: 
    Baltimore City Analysis
  Filename: 
    baltimore.py
  Problem Statement:
    Read the Baltimore_City_Employee_Salaries_FY2014.csv file 
    and perform the following task :

    0. Remove the dollar signs in the AnnualSalary field and assign it as a float
    1. Group the data on JobTitle and AnnualSalary, and aggregate with sum, mean, etc.
       Sort the data and display to show who get the highest salary
    2. Try to group on JobTitle only and sort the data and display
    3. How many employess are there for each JobRoles and Graph it
    4. Graph and show which Job Title spends the most
    5. List All the Agency ID and Agency Name 
    6. Find all the missing Gross data in the dataset 
"""

import pandas as pd
import numpy as np

data= pd.read_csv('pd_csv/Baltimore_City_Employee_Salaries_FY2014.csv')
data.info()

#1. Group the data on JobTitle and AnnualSalary, and aggregate with sum, mean, etc.
#   Sort the data and display to show who get the highest salary

grouped= data.groupby(['JobTitle', 'AnnualSalary'])
statistics =grouped.agg(['sum','mean','max','min'])

df_sorted = data.sort_values(by= ['AnnualSalary'], ascending = False ).reset_index(drop= True)
df_sorted.loc[0]
########################

#2. Try to group on JobTitle only and sort the data and display

df_sorted2 = data.sort_values(by= 'JobTitle')
df_sorted2.head()
########################

#3. How many employess are there for each JobRoles and Graph it

import matplotlib.pyplot as plt

df2= data['JobTitle'].value_counts()

plt.bar( range(20), df2[:20], align='center', alpha=1.0)
plt.xticks(range(20), df2.index[:20],  rotation= 80)
plt.xlabel('Job Roles')
plt.ylabel('Number of Employees')
plt.title('No. of Employees vs Job Roles')
plt.show()

'''OR'''

df3= data.groupby('JobTitle').size().reset_index(name='Freq')
df4= df3.sort_values(by= 'Freq', ascending= False).reset_index()

plt.bar( range(20), df4['Freq'][:20], align='center', alpha=1.0)
plt.xticks(range(20), df4['JobTitle'][:20],  rotation= 80)
plt.xlabel('Job Roles')
plt.ylabel('Number of Employees')
plt.title('No. of Employees vs Job Roles')
plt.show()
#####################

#4. List All the Agency ID and Agency Name 

data[['AgencyID', 'Agency']]
####################

#5. Find all the missing Gross data in the dataset 

data[ data['GrossPay'].isnull() ]



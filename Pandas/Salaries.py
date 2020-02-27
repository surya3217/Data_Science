
"""
Analysis of Salaries Data ( Hand On Activity )

1. Which Male and Female Professor has the highest and the lowest salaries
2. Which Professor takes the highest and lowest salaries.
3. Missing Salaries - should be mean of the matching salaries of those 
   whose service is the same
4. Missing phd - should be mean of the matching service 
5. How many are Male Staff and how many are Female Staff. 
   Show both in numbers and Graphically using Pie Chart.  
   Show both numbers and in percentage
6. How many are Prof, AssocProf and AsstProf. 
   Show both in numbers adn Graphically using a Pie Chart
7. Who are the senior and junior most employees in the organization.
8. Draw a histogram of the salaries divided into bin starting 
   from 50K and increment of 15K
"""

import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("pd_csv/Salaries.csv")
data.groupby(['rank', 'sex']).max()


#1. Which Male and Female Professor has the highest and the lowest salaries
male= data[(data['rank']=='Prof') & (data['sex']=='Male')]

male_max = male[ male['salary']== male['salary'].max() ]
print('Male Prof with highest salary:\n',male_max)
print()
male_min= male[male['salary']== male['salary'].min() ]
print('Male Prof with lowest salary:\n',min_salary)


###
female= data[(data['rank']=='Prof') & (data['sex']=='Female')]

max_salary= female[female['salary']== female['salary'].max() ]
print('Female Prof highest salary:\n',max_salary)
print()
min_salary= female[female['salary']== female['salary'].min() ]
print('Female Prof lowest salary:\n',min_salary)


#####################

#2. Which Professor takes the highest and lowest salaries.
prof= data[data['rank']== 'Prof']
print(prof['salary'].max() )
print(prof['salary'].min() )


#####################

#3. Missing Salaries - should be mean of the matching salaries of those 
#   whose service is the same

## first group all unique sercice years and then apply a funtion on its 
# salary column
data['salary'] = data.groupby('service')['salary'].apply(lambda x: x.fillna(x.mean()))

######################

# 3. Missing Salaries - should be mean of the matching salaries of those 
# whose discipline is the same
"""data['salary'] = data.groupby('discipline')['salary'].apply(lambda x: x.fillna(x.mean()))"""
    
# First Finding the mean of the salries according to the different discipline 
a = data['salary'][data['discipline'] == 'A'].mean()
b = data['salary'][data['discipline'] == 'B'].mean()
    
# Filling the mean salaries for the different categories of discipline
data['salary'][data['discipline'] == 'A'] = data['salary'].fillna(a)
data['salary'][data['discipline'] == 'B'] = data['salary'].fillna(b)


################################

#4. Missing phd - should be mean of the matching service 

data['phd'] = data.groupby('service')['phd'].apply(lambda x: x.fillna(round(x.mean()) ) )

###########################

# 4. Missing phd - should be mean of the matching discipline 
"""data['phd'] = data.groupby('discipline')['phd'].apply(lambda x: x.fillna(x.mean()))"""
    
# First Finding the mean of the phd according to the different discipline 
a1 = data['phd'][data['discipline'] == 'A'].mean()
b1 = data['phd'][data['discipline'] == 'B'].mean()
    
# Filling the mean phd by rounding its value for the different categories of discipline
data['phd'][data['discipline'] == 'A'] = data['phd'].fillna(round(a1))
data['phd'][data['discipline'] == 'B'] = data['phd'].fillna(round(b1)) 

##############################

#5. How many are Male Staff and how many are Female Staff. 
#   Show both in numbers and Graphically using Pie Chart.  
#   Show both numbers and in percentage

staff= dict(data['sex'].value_counts())

vis1 = plt.pie([staff['Male'], staff['Female']], explode=[0, 0], labels=['Male','Female'], autopct="%1.1f%%")
plt.axis('equal')   ## for equal radius i.e circle
plt.show(vis1)

##############################

#6. How many are Prof, AssocProf and AsstProf. 
#   Show both in numbers adn Graphically using a Pie Chart

rank= dict(data['rank'].value_counts())

vis1 = plt.pie([rank['Prof'], rank['AssocProf'], rank['AsstProf'] ] , explode=[0, 0, 0.1], labels=['Prof', 'AssocProf' ,'AsstProf'], autopct="%1.1f%%")
plt.axis('equal')
plt.show(vis1)

##############################

#7. Who are the senior and junior most employees in the organization.

print('Senior:\n', data[data['service']==data['service'].max()] )
print('Junior:\n', data[data['service']==data['service'].min()] )

#############################

#8. Draw a histogram of the salaries divided into bin starting 
#   from 50K and increment of 15K

plt.hist(data['salary'], bins=range(50000, 210000, 15000), facecolor='g')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.title('Salary distribution')
plt.grid(True)
plt.show()

############################

# hands on

df_rank= data.groupby(['rank'])

#df_rank.size()
#df_rank.count()
df_rank.groups


grouped= data.groupby(['rank','sex'])
for name,group in grouped:
    print(name)
    print(group)


import numpy as np
# find all rows of professor rank
grouped= data.groupby(['rank']) ## devide rank into its unique values and which will be the keys

print( grouped.get_group('Prof') )  # give group of key= prof
print( grouped['salary'].agg(np.max) )  









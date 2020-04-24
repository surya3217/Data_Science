"""
Code Challenge
  Name: 
    Thanks giving Analysis
  Filename: 
    Thanksgiving.py
  Problem Statement:
    Read the thanksgiving-2015-poll-data.csv file and 
    Perform the following task :

    1.Convert the column name to single word names   
    2.Discover regional and income-based patterns in what Americans eat for Thanksgiving dinner.
    3.Using the apply method to Gender column to convert Male & Female
    
    4.Using the apply method to clean up income
    (Range to a average number, X and up to X, Prefer not to answer to NaN)
    5.compare income between people who tend to eat homemade cranberry sauce for
      Thanksgiving vs people who eat canned cranberry sauce?
    
    6.find the average income for people who served each type of cranberry sauce
      for Thanksgiving (Canned, Homemade, None, etc).
    7.Find the number of people: 
      who live in each area type (Rural, Suburban, etc)
      who eat different kinds of main dishes for Thanksgiving
      
    8.Plotting the results of aggregation:      
    1.Do people in Suburban areas eat more Tofurkey than people in Rural areas?
    2.Where do people go to Black Friday sales most often?
    3.Is there a correlation between praying on Thanksgiving and income?
    4.What income groups are most likely to have homemade cranberry sauce?

    Verify a pattern:
        People who have Turducken and Homemade cranberry sauce seem to have 
        high household incomes.
        People who eat Canned cranberry sauce tend to have lower incomes, 
        but those who also have Roast Beef have the lowest incomes     
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("pd_csv/thanksgiving.csv", encoding='latin-1')
data.info()
data.plot.hist()

# 1.Convert the column name to single word names    
#df= pd.DataFrame(data)
data.columns= ['RespondentID', 
       'Celebrate Thanksgiving',
       'Main dish',
       'Main dish Other',
       'Cooked Process',
       'Cooked Process Other',
       'stuffing/dressing',
       'stuffing/dressing Other',
       'cranberry sauce type',
       'cranberry sauce type Other',
       'gravy',
       'side dishes - Brussel sprouts',
       'side dishes - Carrots',
       'side dishes - Cauliflower',
       'side dishes - Corn',
       'side dishes - Cornbread',
       'side dishes - Fruit salad',
       'side dishes - Green beans/green bean casserole',
       'side dishes - Macaroni and cheese',
       'side dishes - Mashed potatoes',
       'side dishes - Rolls/biscuits',
       'side dishes - Squash',
       'side dishes - Vegetable salad',
       'side dishes - Yams/sweet potato casserole',
       'side dishes - Other (please specify)',
       'side dishes - Other (please specify).1',
       'pie served - Apple',
       'pie served - Buttermilk',
       'pie served - Cherry',
       'pie served - Chocolate',
       'pie served - Coconut cream',
       'pie served - Key lime',
       'pie served - Peach',
       'pie served - Pecan',
       'pie served - Pumpkin',
       'pie served - Sweet Potato',
       'pie served - None',
       'pie served - Other (please specify)',
       'pie served - Other (please specify).1',
       'desserts - Apple cobbler',
       'desserts - Blondies',
       'desserts - Brownies',
       'desserts - Carrot cake',
       'desserts - Cheesecake',
       'desserts - Cookies',
       'desserts - Fudge',
       'desserts - Ice cream',
       'desserts - Peach cobbler',
       'desserts - None',
       'desserts - Other (please specify)',
       'desserts - Other (please specify).1',
       'pray',
       'travel for Thanksgiving',
       'watch any - Macys Parade',
       'age cutoff at your "kids table" ',
       'meet up with hometown friends',
       'attended any "Friendsgiving?"',
       'Black Friday sales',
       'work in retail',
       'employer work on Black Friday?',
       'area',
       'age',
       'gender',
       'income',
       'US Region']
data.info()
#############################################

#2.Discover regional and income-based patterns in what Americans eat for Thanksgiving 
#  dinner. 

region= data['US Region'].count()
print(region)

inc= data[['income','US Region' ]]
print(inc)
##########################################

#4.Using the apply method to clean up income
#  (Range to a average number, X and up to X, Prefer not to answer to NaN)

#The isinstance() function returns True if the specified object is of the specified type, otherwise False.
#If the type parameter is a tuple, this function will return True if the object is one of the types in the tuple.

import math
def clean_income(value):
    if value == "$200,000 and up":
        return 200000
    elif value == "Prefer not to answer":
        return np.nan
    elif isinstance(value, float) and math.isnan(value):
        return np.nan
    value = value.replace(",", "").replace("$", "")
    income_high, income_low = value.split(" to ")
    return (int(income_high) + int(income_low)) / 2

data['income']= data['income'].apply(clean_income)
data['income'].head()
##########################################

#5.Compare income between people who tend to eat "homemade cranberry sauce" for
#  Thanksgiving vs people who eat "canned cranberry sauce"?

sauce= data.groupby(['cranberry sauce type','income']).size()
print(sauce)
##########################################

#6.find the average income for people who served each type of cranberry sauce
#  for Thanksgiving (Canned, Homemade, None, etc).
canned= data[data['cranberry sauce type']=='Canned']['income'].mean()
homemade= data[data['cranberry sauce type']=='Homemade']['income'].mean() 
none= data[data['cranberry sauce type']=='None']['income'].mean()
other= data[data['cranberry sauce type']=='Other (please specify)']['income'].mean()

income_mean= canned, homemade, none, other
title= data['cranberry sauce type'].value_counts().index

plt.bar(range(4), income_mean, align='center', alpha=1.0)
plt.xticks(range(4), title,  rotation= 0)
plt.xlabel('Cranberry Sauce Type')
plt.ylabel('Incomes of People')
plt.title('Comparision of income who eat Cranberry Sauce')
plt.show()
########################################

#7.Find the number of people: 
#  who live in each area type (Rural, Suburban, etc)
df= pd.DataFrame(data['area'].value_counts() )
df.columns= ['No. of People']
print(df)

# visualization
ax = data['area'].value_counts().plot(kind='bar', figsize=(7,5), 
                                        color="coral", fontsize=11);
ax.set_alpha(0.8)
ax.set_title("People who live in each Area Type", fontsize=18)
ax.set_ylabel("Number of People", fontsize=15);

for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.1, i.get_height()+5, \
            str(round((i.get_height()), 2)), fontsize=11, color='black', rotation=0)

#another way: to print bar values
#for p in ax.patches:
#    ax.annotate(str(p.get_height()), (p.get_x()*1.005, p.get_height()*1.005))
    
'''OR'''

data['area'].value_counts().reset_index(name='No. of People')
#########################################

#  who eat different kinds of main dishes for Thanksgiving
df2= pd.DataFrame(data['Main dish'].value_counts() )
df2.columns= ['No. of Eaters']
print(df2)

# visualization
ax = data['Main dish'].value_counts().plot(kind='barh', figsize=(8,5),
                                                 color="slateblue", fontsize=12);
ax.set_alpha(0.8)
ax.set_title("who eat different kinds of Main Dishes", fontsize=15)
ax.set_xlabel("Number of Eaters", fontsize=15);

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.1, i.get_y()+.31, \
            str(round((i.get_width()), 2)), fontsize=13)
    
# invert for largest on top 
ax.invert_yaxis()
######################################
  
#1.Do people in Suburban areas eat more Tofurkey than people in Rural areas?

Tofurkey= []
dish_count= data.groupby(['area','Main dish']).size()
Tofurkey.append(dish_count['Rural']['Tofurkey'])
Tofurkey.append(dish_count['Suburban']['Tofurkey'])
Tofurkey.append(dish_count['Urban']['Tofurkey'])

if Tofurkey[1]> Tofurkey[0]:
    print('People in "Suburban" areas eat more Tofurkey than people in "Rural" areas.')
else:
    print('People in "Rural" areas eat more Tofurkey than people in "Suburban" areas')

plt.pie(Tofurkey, labels= ['Rural','Suburban','Urban'], autopct='%.0f%%')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Area wise no. of people who eat Tofurkey')
plt.show()
##########################################

#2.Where do people go to Black Friday sales most often?

a= data['travel for Thanksgiving'].value_counts()
print(a.index[0].split('--')[0].replace('is happening','happen').replace('my',''))

'''OR'''

c= dict( data['travel for Thanksgiving'].value_counts() )
#print(c)
d= max(c.values())
for key,val in c.items():
    if c[key]==d:
        print(key.split('--')[0])
##########################################

#3.Is there a correlation between praying on Thanksgiving and income?
'''
 Correlation: dependancy
How the height of basketball players is correlated to their shooting accuracy
Whether there’s a relationship between employee work experience and salary

Linear correlation(pearson method) measures the proximity of the mathematical relationship 
between variables or dataset features to a linear function. If the relationship 
between the two features is closer to some linear function, then their linear 
correlation is stronger and the absolute value of the correlation coefficient 
is higher.
'''

data.groupby(['pray','income']).size()

d= {'Yes':1, 'No':0}
data['pray'] = data['pray'].map(d)

data[['pray','income']].corr(method= 'pearson' )
print('''As values is -ive, so there is no dependency or correlation between "Pray" 
      and "income".''')

'''
                                pray before or after the meal?    income
pray before or after the meal?                        1.000000 -0.098634
income                                               -0.098634  1.000000
'''
'''OR'''

# visualization
import scipy.stats

data['pray'].corr(data['income'], method= 'pearson' )
x= data['pray']
y= data['income']
slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

print('It is shown from the graph that both columns are independent to each other.')

########################################
        
#4.What income groups are most likely to have homemade cranberry sauce?

sauce= data.groupby(['cranberry sauce type','income']).size()
print('Income groups:')
print(sauce['Homemade'])

# visualization
ax = sauce['Homemade'].plot(kind='barh', figsize=(8,5),color="slateblue", fontsize=10);
ax.set_alpha(0.8)
ax.set_title("Income groups which have homemade cranberry sauce", fontsize=13)
ax.set_xlabel("Number of People", fontsize=13);

# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.1, i.get_y()+.31, \
            str(round((i.get_width()), 2)), fontsize=13)
    
# invert for largest on top 
ax.invert_yaxis()


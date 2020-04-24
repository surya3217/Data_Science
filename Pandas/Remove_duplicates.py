'''
Removing duplicates value
'''
import pandas as pd

data= pd.read_csv('DL-FTP-2-Certificate-Form.csv')
data.info()

df= data.drop_duplicates(['Your Name', 'Your Mobile No.'], keep= "last" )
df.info()

df= df.reset_index(drop=True)

df2 = pd.DataFrame(df) 
df2.to_csv('2_DL-FTP-2-Certificate-Form.csv')


#df2 = data.groupby(['Your Name', 'Your Mobile No.']).size().reset_index(name='Freq')
#print(df2)
#repeat= df2[df2['Freq']>1 ]['Your Name']





#!/usr/bin/env python
# coding: utf-8

# ## Eploratory Data Analysis (EDA)

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Reading Data From CSV putting into a data frame
df = pd.read_csv(r'D:\Online Study\Projects For Data Analysis\JB Music Therapy Analysis\Music-and-Therapy-survey-1.csv')
df.head()


# In[3]:


# Looking at the number of columns and rows present in the csv 
df.shape


# In[4]:


df.info()


# In[5]:


# looking at null values
df.isnull().sum()


# In[6]:


# Describing Data in the data frame
df.describe()


# In[7]:


# looking for duplicate rows
duplicateRows = df[df.duplicated()]
duplicateRows


# In[8]:


# Looking at Unique Ages
Agegroup = df['Age'].unique()
print(Agegroup)
df['Age'].nunique()


# In[9]:


# Looking at Unique Primary Streaming Services 
pss = pd.value_counts(df['Primary streaming service']) 
print(pss)
df['Primary streaming service'].nunique()


# In[10]:


# Looking at Unique Hours Per Day listen Services
WW = pd.value_counts(df['Fav genre'])
count_unique = df['Fav genre'].nunique()
print(WW)
count_unique


# In[11]:


# Looking at peoples replay on different Music Types.
classic_music = pd.value_counts(df['Frequency [Classical]'])
country_music = pd.value_counts(df['Frequency [Country]'])
edm_music = pd.value_counts(df['Frequency [EDM]'])
Folk_music = pd.value_counts(df['Frequency [Folk]'])

print('Classic Music Listners \n',classic_music,'\n\n Country Music Listners\n',country_music,
      '\n\n EDM Music Listners\n',edm_music,'\n\n Folk Music Listners\n',Folk_music)


# In[12]:


# Looking at peoples replay on different Music Types.
Gospel_music = pd.value_counts(df['Frequency [Gospel]'])
hiphop_music = pd.value_counts(df['Frequency [Hip hop]'])
Jazz_music = pd.value_counts(df['Frequency [Jazz]'])
Kpop_music = pd.value_counts(df['Frequency [K pop]'])

print('Gospel Music Listners \n',Gospel_music,'\n\n Hip Hop Music Listners\n',hiphop_music,
      '\n\n Jazz Music Listners\n',Jazz_music,'\n\n K Pop Music Listners\n',Kpop_music)


# In[13]:


# Looking at peoples replay on different Music Types.
lofi_music = pd.value_counts(df['Frequency [Lofi]'])
metal_music = pd.value_counts(df['Frequency [Metal]'])
pop_music = pd.value_counts(df['Frequency [Pop]'])
rnb_music = pd.value_counts(df['Frequency [R&B]'])

print('Lofi Music Listners \n',lofi_music,'\n\n Metal Music Listners\n',metal_music,
      '\n\n Pop Music Listners\n',pop_music,'\n\n R & B Music Listners\n',rnb_music)


# In[14]:


# Looking at peoples replay on different Music Types.
Rap_music = pd.value_counts(df['Frequency [Rap]'])
Rock_music = pd.value_counts(df['Frequency [Rock]'])
VG_music = pd.value_counts(df['Frequency [Video game music]'])

print('Rap Music Listners \n',Rap_music,'\n\n Rock Music Listners\n',Rock_music,
      '\n\n Vidoe Game Music Listners\n',VG_music)


# In[15]:


# looking at the Anxiety column values
anxiety = pd.value_counts(df['Anxiety'])
print(anxiety)


# In[16]:


# looking at the Depression column values
depress = pd.value_counts(df['Depression'])
print(depress)


# In[17]:


# looking at the Insomnia column values
insomia = pd.value_counts(df['Insomnia'])
print(insomia)


# In[18]:


# looking at the Insomnia column values
ocd = pd.value_counts(df['OCD'])
print(ocd)


# In[19]:


# looking at the Music Effects column values
music_effects = pd.value_counts(df['Music effects'])
print(music_effects)


# In[20]:


# looking at the Permissions column values
permissions = pd.value_counts(df['Permissions'])
print(permissions)


# ## With this we now have understanding of what data we are working with, its time to clean data & create visuals
# 

# In[21]:


df.isnull().sum()


# In[22]:


# Note we have 736 rows and 33 columnns for better analysis we should not remove  null values which can be replaced, for better analysis
df.shape


# In[23]:


# Creating another Data Frame to work on for cleaning data. Also Excluding timestamp column as it woun't be required further
data = df.drop('Timestamp',axis=1)
data.head()


# In[24]:


data.shape


# In[25]:


data.isnull().sum()


# In[26]:


# Removing null values from Age & Primary streaming service
data = data.dropna(subset = ['Age','Primary streaming service'])


# In[27]:


data.isnull().sum()


# In[28]:


# Only 2 rows deleted where i had null values in column age & primary streaming service
data.shape


# In[29]:


# WE CAN ALO REMOVE Null values in columns while working, Instrumentalist, Composer & Foreign Languages
data = data.dropna(subset = ['While working','Instrumentalist','Composer','Foreign languages'])


# In[30]:


data.shape


# In[31]:


# Here we can see that only 2 columns contain null values now BPM & Music Effects
data.isnull().sum()


# ### We can not Remove all 104 rows where BPM is Null instead we will replace it. Removing it will have major effect on our dataset.

# In[32]:


data.BPM.value_counts()


# In[33]:


# Selecting Row where an unconsitent data is detected.
data.loc[data['BPM'] == 999999999.0 ] 


# In[34]:


# Replacing that unconsitent value with zero. 
data['BPM'].replace(999999999.0, 0, inplace = True)


# In[35]:


# AS now we can see at row 568 only the unstructured value has been updated
data.loc[data['BPM'] == 0]


# In[36]:


# The values in BPM now give more sense 
data.BPM.value_counts()


# In[37]:


# Now the average, mean gives more sense for the Column BPM
data.describe()


# In[38]:


data.isnull().sum()


# In[39]:


# Filling Null values of BPM column with mean value of BPM that will give us an average look which will be better for analysis
data.BPM.fillna(data.BPM.mean(),inplace = True)
data.isnull().sum()


# In[40]:


data.shape


# In[41]:


# We Will remove rows where music effect is null. 
data = data.dropna(subset = ['Music effects'])
data.isnull().sum()


# In[42]:


# WE have trimmed about 2.4% of dataset, to make sure we got clean data set. 
data.shape


# # Few Data Visuals

# In[43]:


# Importing Libraries for Data Visualization 
import matplotlib.pyplot as plt
import seaborn as sns


# In[44]:


# Heat Matrix For Correlation
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True, annot = True, cmap = 'mako' )


# In[45]:


# Bar chart Showing the amount of hours played on each streaming service. Spotify is being mostly used 
Music_Services = data['Primary streaming service']
Hpd = data['Hours per day']
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(9)
plt.bar(Music_Services,Hpd)
plt.title("Most Used Streaming Service")
plt.show()


# In[46]:


# exporting Clean Data to work on Data Visualization.
data.to_csv('Music_Therapy_Clean.csv')


# In[ ]:





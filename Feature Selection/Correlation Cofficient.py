#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read data
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Information gain\Census22_40QT.csv')
df.head()


# In[2]:


# Correlation cofficients
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


cor = df.corr()    # Plotting heatmap
plt.figure(figsize =(30,20))
sns.heatmap(cor, annot= True)


# In[19]:


df_numeric =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Correlation Cofficients\Numeric_Attributes.csv')
df_nominal =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Correlation Cofficients\Nominal_Attributes.csv')

df_numeric = df_numeric.iloc[ : , : ]
df_nominal = df_nominal.iloc[ : , : ]


# In[20]:


def display_corr_spearman(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(30,20))
    heatmap = sns.heatmap(df.corr(), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation")
    return(r)
display_corr_spearman(df_nominal)


# In[21]:


def display_corr_pearson(df):
    r = df.corr(method="pearson")
    plt.figure(figsize=(10,6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Pearson Correlation")
    return(r)
display_corr_pearson(df_numeric)


# In[ ]:





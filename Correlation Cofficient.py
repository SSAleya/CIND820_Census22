#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read data
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Information gain\Census22_40QT.csv')
df.head()


# In[7]:


# Correlation cofficients
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


cor = df.corr()    # Plotting heatmap
plt.figure(figsize =(30,20))
sns.heatmap(cor, annot= True)


# In[19]:


df_numeric =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Correlation Cofficients\Numeric_Attributes.csv')
df_nominal =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Correlation Cofficients\Nominal_Attributes.csv')

df_numeric = df_numeric.iloc[ : , : ]
df_nominal = df_nominal.iloc[ : , : ]


# In[20]:


cor = df_nominal.corr()    # Plotting heatmap
plt.figure(figsize =(30,20))
sns.heatmap(cor, annot= True)


# In[21]:


cor = df_numeric.corr()    # Plotting heatmap
plt.figure(figsize =(10,6))
sns.heatmap(cor, annot= True)


# In[ ]:





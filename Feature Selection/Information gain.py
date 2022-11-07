#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Read data
import pandas as pd
import numpy as np
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Preparing Target Variable\Census22_2QT.csv')
df.head()


# In[9]:


df.info()


# In[10]:


# Mutula information using full dataset
X = df.iloc[:, :-1]
y = df.iloc[:, -1:]
   
from sklearn.feature_selection import mutual_info_classif
mutual_info = mutual_info_classif(X,np.ravel(y,order="c"))
mutual_info


# In[11]:


mutual_info =pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values (ascending=False)


# In[12]:


mutual_info.sort_values (ascending=False).plot.bar(figsize=(40,15))


# In[ ]:





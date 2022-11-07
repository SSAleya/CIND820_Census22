#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing the libraries

import pandas as pd
from sklearn.preprocessing import normalize


# In[14]:


# Read data
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Variances Threshold\Census22_39.csv')
df.head()


# In[15]:


normalize = normalize(df)
data_scaled = pd.DataFrame(normalize)
data_scaled.var()


# In[16]:


#storing the variance and name of variables

variance = data_scaled.var()
columns = df.columns


# In[24]:


#saving the names of variables having variance more than a threshold value

variable = [ ]

for i in range(0,len(variance)):
    if variance[i]>=0.0001: 
        variable.append(columns[i])


# In[25]:


variable


# In[ ]:





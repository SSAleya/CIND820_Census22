#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install https://github.com/ydataai/pandas-profiling/archive/master.zip')

import pandas_profiling as pdpf


# In[2]:


import pandas as pd
import requests
import io
    
# Downloading the csv file from your GitHub account

#url = "https://raw.githubusercontent.com/SSAleya/CIND820_Census22/master/Census22.csv" 

# Reading the downloaded content and turning it into a pandas dataframe

#df = pd.read_csv(url)


# In[3]:


import pandas as pd
Census22_40 = pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Panda profiling\Census22_40.csv')


# In[4]:


from pandas_profiling import ProfileReport
profile1 = ProfileReport(Census22_40, title="Pandas_Profiling_Report1")


# In[5]:


profile1


# In[6]:


profile1.to_file("my_report1.html")


# In[ ]:





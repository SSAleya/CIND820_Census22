#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install https://github.com/ydataai/pandas-profiling/archive/master.zip')

import pandas_profiling as pdpf


# In[5]:


import pandas as pd
import requests
import io
    
# Downloading the csv file from your GitHub account

#url = "C:\\Users\sania\Data analytics\Course materials 820\Latest census\Data collectionCensus22_initial.csv" 

# Reading the downloaded content and turning it into a pandas dataframe

df = pd.read_csv("C:\\Users\sania\Data analytics\Course materials 820\Latest census\Data collection\Census22_initial.csv")


# In[11]:


# for the df in the question,

df['NOEMP']= df['NOEMP'].astype('str')
df['A_UNMEM'] = df['A_UNMEM'].astype('str')
df['FRMOTR'] = df['FRMOTR'].astype('str')
df['VET_YN']= df['VET_YN'].astype('str')
df['VET_QVA']= df['VET_QVA'].astype('str')
df['WKSWORK']= df['WKSWORK']. astype('int')
df['MIG_MTR3']= df['MIG_MTR3'].astype('str')
df['MIGSAME']= df['MIGSAME'].astype('str')
df['HHDREL']= df['HHDREL'].astype('str')
df['HHDFMX']= df['HHDFMX'].astype('str')
df['PARENT']= df['PARENT'].astype('str')
df['A_WKSTAT']= df['A_WKSTAT'].astype('str')
df['PRUNTYPE']= df['PRUNTYPE'].astype('str')
df['A_CLSWKR']= df['A_CLSWKR'].astype('str')
df['STATETAX_B']= df['STATETAX_B'].astype('int')
df['STATETAX_A']= df['STATETAX_A'].astype('int')
df['FILESTAT']= df['FILESTAT'].astype('str')
df['DIV_VAL']= df['DIV_VAL'].astype('int')
df['CAP_VAL']= df['CAP_VAL'].astype('int')
df['ERN_OTR']=  df['ERN_OTR'].astype('str')
df['A_HRSPAY']= df['A_HRSPAY'].astype('int')
df['A_MJOCC']= df['A_MJOCC'].astype('str')
df['A_MJIND']= df['A_MJIND'].astype('str')
df['A_HGA']= df['A_HGA'].astype('str')
df['A_MARITL'] = df['A_MARITL'].astype('str')
df['A_ENRLW']=  df['A_ENRLW'].astype('str')
df['A_SEX']= df['A_SEX'].astype('str')
df['PRDISFLG']= df['PRDISFLG'].astype('str')
df['PEMLR']= df['PEMLR'].astype('str')
df['PRCITSHP']= df['PRCITSHP'].astype('str')
df['PRDTRACE']= df['PRDTRACE'].astype('str')
df['PEFNTVTY']= df['PEFNTVTY'].astype('str')
df['PEMNTVTY']= df['PEMNTVTY'].astype('str')
df['PENATVTY']= df['PENATVTY'].astype('str')
df['PEHSPNON']= df['PEHSPNON'].astype('str')
df['OED_TYP3']= df['OED_TYP3'].astype('str')
df['OED_TYP2']= df['OED_TYP2'].astype('str')
df['OED_TYP1']= df['OED_TYP1'].astype('str')
df['PTOTVAL']= df['PTOTVAL'].astype('int')
df['A_AGE']= df['A_AGE'].astype('int')


# In[12]:


df.dtypes


# In[13]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, infer_dtypes=False, title="Pandas Profiling Report")


# In[14]:


profile


# In[15]:


profile.to_file("EDA_Final.html")


# In[ ]:





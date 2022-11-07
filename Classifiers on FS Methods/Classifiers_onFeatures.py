#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd


# In[10]:


# T/T Split: Accuracy

data = {'DT' : [0.79, 0.80, 0.78, 0.47, 0.79, 0.80],
    'RF' : [0.85, 0.85, 0.84, 0.52, 0.85, 0.85],
    'NB' : [0.76, 0.69, 0.63, 0.34, 0.69, 0.66],
    'LR' : [0.80, 0.80, 0.79, 0.44, 0.81, 0.81],
    'KNN' : [0.82, 0.82, 0.82, 0.50, 0.82, 0.82]}


df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("Accuracy")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[16]:


# T/T Split: MCC

data = {'DT' : [0.58, 0.60, 0.57, 0.34, 0.58, 0.59],
    'RF' : [0.69, 0.69, 0.68, 0.41, 0.70, 0.71],
    'NB' : [0.52, 0.44, 0.34, 0.19, 0.44, 0.37],
    'LR' : [0.61, 0.61, 0.59, 0.30, 0.62, 0.62],
    'KNN' : [0.63, 0.63, 0.63, 0.37, 0.64, 0.64]}

df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("MCC")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[17]:


# T/T Split: Time

data = {'DT' : [3.26, 2.79, 2.7, 0.66, 3.79, 3.28],
    'RF' : [25.71, 18.56, 20.12, 10.39, 26.24, 19.44],
    'NB' : [1.85, 1.79, 1.83, 0.16, 2.06, 1.56],
    'LR' : [4.31, 9.88, 5.64, 9.86, 10.38, 8.59],
    'KNN' : [6.02, 5.60, 5.44, 30.25, 7.77, 6.27]}

df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("Time")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[18]:


# RKF: Accuracy

data = {'DT' : [0.79, 0.80, 0.79, 0.48, 0.80, 0.80],
    'RF' : [0.83, 0.84, 0.83, 0.51, 0.84, 0.84],
    'NB' : [0.76, 0.69, 0.63, 0.35, 0.69, 0.66],
    'LR' : [0.80, 0.79, 0.78, 0.36, 0.79, 0.78],
    'KNN' : [0.82, 0.82, 0.81, 0.50, 0.82, 0.82]}

df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("Accuracy")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[20]:


# RKF: MCC

data = {'DT' : [0.58, 0.59, 0.57, 0.35,0.59, 0.60],
    'RF' : [0.67, 0.67, 0.65, 0.39, 0.63, 0.68],
    'NB' : [0.52, 0.45, 0.34, 0.20, 0.44, 0.38],
    'LR' : [0.59, 0.58, 0.57, 0.21, 0.58, 0.56],
    'KNN' : [0.63, 0.64, 0.63, 0.38, 0.65, 0.64]}

df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("MCC")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[21]:


# RKF: Time

data = {'DT' : [184.43, 102.37, 96.60, 65.92, 148.72, 146.74],
    'RF' : [251.81, 147.76, 152.42, 124.26, 211.85, 210.68],
    'NB' : [61.97, 52.96, 45.79, 10.28, 87.63, 76.11],
    'LR' : [2076.28, 1304.28, 1227.64, 1821.31, 1937.11, 1961.31],
    'KNN' : [550.84, 519.73, 502.94, 2169.55, 786.00, 788.92]}

df = pd.DataFrame(data,columns=['DT','RF','NB','LR','KNN'], index = ["Filter_MVR", "Filter_MIG", "Filter_LVF", "Filter_CC", "Wrapper_FS", "Embedded_SFS"])

df.plot.bar(figsize=(10, 5), rot=0)
plt.ylabel("Time")
plt.xlabel("Feature Selection Methods")
plt.title("Methods vs Classifiers")
plt.show()


# In[ ]:





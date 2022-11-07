#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Sequential forward selection\Census22_40.csv')


# In[2]:


def missing_zero_values_table(dff):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        zero_val_percent = 100 * zero_val / len(df)
        print(zero_val)
        print(zero_val_percent)

missing_zero_values_table(df)


# In[3]:


def missing_zero_values_table(dff):
        zero_val = (df1 == 0.00).astype(int).sum(axis=0)
        zero_val_percent = 100 * zero_val / len(df1)
        print(zero_val)
        print(zero_val_percent)

from sklearn import model_selection

df1 = df.iloc[:,[0,1,2,20,21,26,28,34,36,37]].copy()

missing_zero_values_table(df1)


# In[4]:


from tabulate import tabulate

Percentages = [["Attributes","% of Zeros"],
            ["OED_TYP1",97.05],
            ["OED_TYP2",97.05],
            ["OED_TYP3",97.05],
            ["CAP_VAL",94.08],
            ["DIV_VAL",84.33],
            ["PRUNTYPE",98.00],
            ["PARENT",97.77],
            ["VET_QVA",97.97],
            ["FRMOTR",93.21],
            ["A_UNMEM",90.07]]

table = tabulate(Percentages,headers='firstrow')
print(table)


# In[5]:


import matplotlib.pyplot as plt
Attributes = ["OED_TYP1","OED_TYP2","OED_TYP3", "CAP_VAL", "DIV_VAL","PRUNTYPE","PARENT","VET_QVA","FRMOTR","A_UNMEM"]
# This part is done in R
Zeros = [103395, 103395, 103395, 100234, 89844, 104396, 104159, 104367, 99296, 95961]
NonZeros = [3139, 3139, 3139, 6300, 16690, 2138, 2375, 2167, 7238, 10573]

w=0.8
plt.barh(Attributes,Zeros,w, label= "Zeros")
plt.barh(Attributes, NonZeros, left=Zeros, label="NonZeros")

plt.ylabel("Attributes")
plt.xlabel("Count")
plt.title("Zeros vs NonZeros")
plt.legend()
plt.show()


# In[ ]:





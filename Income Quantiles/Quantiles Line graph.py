#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Importing packages
import matplotlib.pyplot as plt

# T/T Split

# Define data values
Categories = [2,3,5,10]
DT_Acc = [0.79, 0.65, 0.49, 0.32]
RF_Acc = [0.84, 0.71, 0.54, 0.35]
NB_Acc = [0.69, 0.56, 0.40, 0.23]
LR_Acc = [0.81, 0.67, 0.49, 0.29]
KNN_Acc =[0.82, 0.68, 0.52, 0.34]

# Plot a simple line chart
plt.plot(Categories, DT_Acc, 'g', label='DT_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, RF_Acc, 'r', label='RF_Acc')

# Plot a simple line chart
plt.plot(Categories, NB_Acc, 'b', label='NB_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, LR_Acc, 'y', label='LR_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, KNN_Acc, 'slategray', label='KNN_Acc')

plt.title('Model Accuracy Vs Income Categories(T/T)', fontsize=14)
plt.xlabel('Income Categories', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.legend()
plt.show()


# In[10]:


# Importing packages
import matplotlib.pyplot as plt

# RKF

# Define data values
Categories = [2,3,5,10]
DT_Acc = [0.80, 0.66, 0.50, 0.33]
RF_Acc = [0.84, 0.72, 0.55, 0.36]
NB_Acc = [0.69, 0.40, 0.40, 0.23]
LR_Acc = [0.79, 0.62, 0.42, 0.24]
KNN_Acc =[0.82, 0.69, 0.52, 0.34]

# Plot a simple line chart
plt.plot(Categories, DT_Acc, 'g', label='DT_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, RF_Acc, 'r', label='RF_Acc')

# Plot a simple line chart
plt.plot(Categories, NB_Acc, 'b', label='NB_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, LR_Acc, 'y', label='LR_Acc')

# Plot another line on the same chart/graph
plt.plot(Categories, KNN_Acc, 'slategray', label='KNN_Acc')

plt.title('Model Accuracy Vs Income Categories(RKF)', fontsize=14)
plt.xlabel('Income Categories', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)

plt.legend()
plt.show()


# In[ ]:





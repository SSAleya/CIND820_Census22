#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Read data
import numpy as np
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Census22_2QT.csv')
# top16 =["PEHSPNON", "PEMLR", "A_SEX", "A_HGA", "A_AGE", "A_MJIND", "A_MJOCC","CAP_VAL","DIV_VAL", "STATETAX_A", "STATETAX_B", "A_CLSWKR","A_WKSTAT","HHDFMX","WKSWORK""NOEMP", "Income"]

df1 = df.loc[:,["PEHSPNON", "PEMLR", "A_SEX", "A_HGA", "A_AGE", "A_MJIND", "A_MJOCC","CAP_VAL","DIV_VAL", "STATETAX_A", "STATETAX_B", "A_CLSWKR","A_WKSTAT","HHDFMX","WKSWORK", "NOEMP", "Income"]]


# In[6]:


df1.head()


# In[7]:


# 2. Repeated K-Fold Cross-Validation

X = df1.values[:,:-1]
y = df1.values[:,-1:]

from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef

# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)


# In[10]:


### Decision tree
import timeit
start = timeit.default_timer()

from sklearn.tree import DecisionTreeClassifier
dt_clf1 = DecisionTreeClassifier()


# evaluate model
scores = cross_val_score(dt_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(dt_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(dt_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(dt_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
scores4 = cross_val_score(dt_clf1, X, y, scoring='matthews_corrcoef', cv=cv, n_jobs=-1)


stop = timeit.default_timer()
print('Time: ', stop - start)  

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))
print( 'MCC:" % 3f (%3f)' % (mean(scores4), std(scores4)))


# In[11]:


### Random Forest
import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10)

# evaluate model
scores = cross_val_score(RFC, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(RFC, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(RFC, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(RFC, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
scores4 = cross_val_score(RFC, X, y, scoring='matthews_corrcoef', cv=cv, n_jobs=-1)


stop = timeit.default_timer()
print('Time: ', stop - start)  

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))
print( 'MCC:" % 3f (%3f)' % (mean(scores4), std(scores4)))


# In[10]:


### LogisticRegression

import timeit
start = timeit.default_timer()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
lr_clf1 = LogisticRegression(C=1.0, solver='lbfgs', max_iter = 1000)

# evaluate model
scores = cross_val_score(lr_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(lr_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(lr_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(lr_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
scores4 = cross_val_score(lr_clf1, X, y, scoring='matthews_corrcoef', cv=cv, n_jobs=-1)


stop = timeit.default_timer()
print('Time: ', stop - start)  

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))
print( 'MCC:" % 3f (%3f)' % (mean(scores4), std(scores4)))


# In[9]:


### KNN

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn_clf1 = KNeighborsClassifier(n_neighbors = 11)

# evaluate model
scores = cross_val_score(knn_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(knn_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(knn_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(knn_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
scores4 = cross_val_score(knn_clf1, X, y, scoring='matthews_corrcoef', cv=cv, n_jobs=-1)


stop = timeit.default_timer()
print('Time: ', stop - start)  

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))
print( 'MCC:" % 3f (%3f)' % (mean(scores4), std(scores4)))


# In[14]:


### Naive Bayes

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.naive_bayes import GaussianNB
nb_clf1 = GaussianNB()

# evaluate model
scores = cross_val_score(nb_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(nb_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(nb_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(nb_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)
scores4 = cross_val_score(nb_clf1, X, y, scoring='matthews_corrcoef', cv=cv, n_jobs=-1)


stop = timeit.default_timer()
print('Time: ', stop - start)  

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))
print( 'MCC:" % 3f (%3f)' % (mean(scores4), std(scores4)))


# In[15]:


# 1.HoldOut Cross-validation or Train-Test Split
# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df1.values[:,:-1],
    df1.values[:,-1:],
    test_size=0.20,
    random_state=42)

from sklearn.metrics import classification_report
from sklearn import metrics


# In[16]:


### Decision tree

import timeit
start = timeit.default_timer()

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef
print( "MCC:", matthews_corrcoef(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[17]:


### Random Forest

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, np.squeeze(y_train))
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef
print( "MCC:", matthews_corrcoef(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[18]:


### Naive Bayes

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

nb_clf.fit(X_train, np.squeeze(y_train))
y_pred = nb_clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef
print( "MCC:", matthews_corrcoef(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[19]:


### KNN

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 11)

knn_clf.fit(X_train, np.squeeze(y_train))
y_pred = knn_clf.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef
print( "MCC:", matthews_corrcoef(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[20]:


### Multinomial logistic regression 

# Normalizing data

from sklearn import preprocessing
import pandas as pd

scaler = preprocessing.MinMaxScaler()

d_X_train = scaler.fit_transform(X_train)
scaled_X_train = pd.DataFrame(d_X_train)

d_X_test = scaler.fit_transform(X_test)
scaled_X_test = pd.DataFrame(d_X_test)

import timeit
start = timeit.default_timer()

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter = 1000)

# Train multinomial logistic regression model
mul_lr = model.fit(scaled_X_train, np.squeeze(y_train))
y_pred = mul_lr.predict(scaled_X_test)
print(classification_report(y_test, y_pred))

from sklearn.metrics import matthews_corrcoef
print( "MCC:", matthews_corrcoef(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)  


# In[40]:


# T/T Split
import matplotlib.pyplot as plt
import pandas as pd

data ={'Accuracy' : [0.79, 0.84, 0.63, 0.80, 0.82],
       'F1-score': [0.79, 0.84, 0.57, 0.80, 0.82],
       'Precision': [0.79, 0.84, 0.76, 0.80, 0.82],
       'Recall': [0.79, 0.84, 0.62, 0.80, 0.82],
        'MCC' : [0.59, 0.69, 0.35, 0.60, 0.64]}

df = pd.DataFrame(data,columns=['Accuracy','F1-score','Precision','Recall','MCC'], index = ['DT','RF','NB','LR','KNN'])

df.plot.bar(figsize=(15,10), rot=0)
plt.ylabel("Parameters")
plt.xlabel("Classifiers")
plt.title("Model Effectiveness")
plt.show()


# In[26]:


import matplotlib.pyplot as plt

Classifiers = ['DT','RF','NB','LR','KNN']       
Time= [1.21, 9.59, 0.83, 2.43, 3.60]

w=0.8
plt.bar(Classifiers, Time)

plt.ylabel("Time")
plt.xlabel("Classifiers")
plt.title("Model Efficiency")
plt.show()


# In[41]:


# RKF
import matplotlib.pyplot as plt
import pandas as pd

data ={'Accuracy' : [0.79, 0.83, 0.62, 0.78, 0.82],
       'F1-score':[0.79, 0.83, 0.57, 0.78, 0.82],
       'Precision':[0.79, 0.83, 0.75, 0.78, 0.82],
       'Recall': [0.79, 0.83, 0.62, 0.78, 0.82],
        'MCC' : [0.59, 0.67, 0.35, 0.57, 0.65]}

df = pd.DataFrame(data,columns=['Accuracy','F1-score','Precision','Recall','MCC'], index = ['DT','RF','NB','LR','KNN'])

df.plot.bar(figsize=(15,10), rot=0)
plt.ylabel("Parameters")
plt.xlabel("Classifiers")
plt.title("Model Effectiveness")
plt.show()


# In[21]:


import matplotlib.pyplot as plt

Classifiers = ['DT','RF','NB','LR','KNN']       
Time= [66.59, 118.12, 37.38, 1203.37, 458.96]

w=0.8
plt.bar(Classifiers, Time)

plt.ylabel("Time")
plt.xlabel("Classifiers")
plt.title("Model Efficiency")
plt.show()


# In[ ]:





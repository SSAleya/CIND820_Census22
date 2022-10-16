#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Read data
import numpy as np
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Information gain\Census22_28MI.csv')


# In[7]:


df.head()


# In[8]:


# 1.HoldOut Cross-validation or Train-Test Split
# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:,:-1],
    df.values[:,-1:],
    test_size=0.20,
    random_state=42)


# In[9]:


### Decision tree
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)


# In[10]:


from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = dt_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[11]:


### Random Forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, np.squeeze(y_train))
rf_clf.score(X_test, y_test)


# In[12]:


from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[13]:


### Naive Bayes
import numpy as np
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

nb_clf.fit(X_train, np.squeeze(y_train))
nb_clf.score(X_test, y_test)


# In[14]:


from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = nb_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[15]:


### KNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 11)

knn_clf.fit(X_train, np.squeeze(y_train))
knn_clf.score(X_test, y_test)


# In[16]:


from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = knn_clf.predict(X_test)
print(classification_report(y_test, y_pred))


# In[17]:


### Multinomial logistic regression 

# Normalizing data

from sklearn import preprocessing
import pandas as pd

scaler = preprocessing.MinMaxScaler()

d_X_train = scaler.fit_transform(X_train)
scaled_X_train = pd.DataFrame(d_X_train)

d_X_test = scaler.fit_transform(X_test)
scaled_X_test = pd.DataFrame(d_X_test)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', C=1.0, solver='lbfgs', max_iter = 1000)

# Train multinomial logistic regression model
mul_lr = model.fit(scaled_X_train, np.squeeze(y_train))

metrics.accuracy_score(y_test, mul_lr.predict(scaled_X_test))


# In[18]:


from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = mul_lr.predict(scaled_X_test)
print(classification_report(y_test, y_pred))


# In[19]:


# 2. Repeated K-Fold Cross-Validation

X = df.values[:,:-1]
y = df.values[:,-1:]

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

# prepare the cross-validation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)


# In[20]:


## Decision tree

from sklearn.tree import DecisionTreeClassifier

dt_clf1 = DecisionTreeClassifier()

# evaluate model
scores = cross_val_score(dt_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(dt_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(dt_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(dt_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))


# In[21]:


### Random Forest
import numpy as np

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=100)

# evaluate model
scores = cross_val_score(RFC, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(RFC, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(RFC, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(RFC, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))


# In[22]:


### LogisticRegression
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial', C=1.0, solver='lbfgs', max_iter = 1000)

# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(model, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(model, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(model, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))


# In[23]:


### KNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

knn_clf1 = KNeighborsClassifier(n_neighbors = 11)

# evaluate model
scores = cross_val_score(knn_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(knn_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(knn_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(knn_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))


# In[24]:


### Naive Bayes
import numpy as np
from sklearn.naive_bayes import GaussianNB

nb_clf1 = GaussianNB()

# evaluate model
scores = cross_val_score(nb_clf1, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
scores1 = cross_val_score(nb_clf1, X, y, scoring='f1_macro', cv=cv, n_jobs=-1)
scores2 = cross_val_score(nb_clf1, X, y, scoring='precision_macro', cv=cv, n_jobs=-1)
scores3 = cross_val_score(nb_clf1, X, y, scoring='recall_macro', cv=cv, n_jobs=-1)

# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
print('F1: %.3f (%.3f)' % (mean(scores1), std(scores1)))
print('Precision : %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('Recall : %.3f (%.3f)' % (mean(scores3), std(scores3)))


# In[ ]:





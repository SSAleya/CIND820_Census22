#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Read data
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Preparing Target Variable\Census22_10QT.csv')


# In[21]:


# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:,:-1],
    df.values[:,-1:],
    test_size=0.20,
    random_state=42)


# In[22]:


### Decision tree
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=0)

dt_clf.fit(X_train, y_train)
dt_clf.score(X_test, y_test)


# In[23]:


y_pred = dt_clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[24]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))


# In[25]:


### Random Forest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)

rf_clf.fit(X_train, np.squeeze(y_train))
rf_clf.score(X_test, y_test)


# In[26]:


y_pred = rf_clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[27]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))


# In[28]:


### Naive Bayes
import numpy as np
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

nb_clf.fit(X_train, np.squeeze(y_train))
nb_clf.score(X_test, y_test)


# In[29]:


y_pred = nb_clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[30]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))


# In[31]:


### KNN
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 11)

knn_clf.fit(X_train, np.squeeze(y_train))
knn_clf.score(X_test, y_test)


# In[32]:


y_pred = knn_clf.predict(X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[33]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))


# In[34]:


# Normalizing data

from sklearn import preprocessing
import pandas as pd
scaler = preprocessing.MinMaxScaler()

d_X_train = scaler.fit_transform(X_train)
scaled_X_train = pd.DataFrame(d_X_train)

d_X_test = scaler.fit_transform(X_test)
scaled_X_test = pd.DataFrame(d_X_test)


# In[35]:


### Multinomial logistic regression 
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Train multinomial logistic regression model
mul_lr = LogisticRegression(multi_class='multinomial', C=1.0, solver='lbfgs', max_iter = 1000)

mul_lr.fit(scaled_X_train, np.squeeze(y_train))
mul_lr.score(scaled_X_test, y_test)


# In[36]:


y_pred = mul_lr.predict(scaled_X_test)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[37]:


from sklearn.metrics import classification_report
from sklearn import metrics

print(classification_report(y_test, y_pred))


# In[38]:


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


# In[39]:


### Decision tree

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


# In[40]:


### Random Forest
import numpy as np

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=10)

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


# In[41]:


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


# In[42]:


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


# In[43]:


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





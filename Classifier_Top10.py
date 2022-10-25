#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Census22_40.csv')
col_select = select= [3,7,14,17,20,21,25,27,29,38,39]
df1 = df.iloc[:, col_select]  
#["PEHSPNON","PRDTRACE","A_HGA","A_MJOCC","CAP_VAL","DIV_VAL","A_CLSWKR","A_WKSTAT","HHDFMX","NOEMP"]


# In[2]:


df1.head()


# In[3]:


df1.columns


# In[4]:


# 1.HoldOut Cross-validation or Train-Test Split
# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df1.values[:,:-1],
    df1.values[:,-1:],
    test_size=0.20,
    random_state=42)


# In[5]:


### Decision tree

import timeit
start = timeit.default_timer()

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=0)
dt_clf.fit(X_train, y_train)

from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = dt_clf.predict(X_test)
print(classification_report(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start)


# In[6]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)


# In[7]:


### Random Forest

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)
rf_clf.fit(X_train, np.squeeze(y_train))

from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = rf_clf.predict(X_test)
print(classification_report(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[8]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)


# In[9]:


### Naive Bayes

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, np.squeeze(y_train))

from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = nb_clf.predict(X_test)
print(classification_report(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[10]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)


# In[11]:


### KNN

import timeit
start = timeit.default_timer()

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 11)
knn_clf.fit(X_train, np.squeeze(y_train))

from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = knn_clf.predict(X_test)
print(classification_report(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[12]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)


# In[13]:


### Multinomial logistic regression 

import timeit
start = timeit.default_timer()

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
from sklearn import metrics

# Train multinomial logistic regression model
mul_lr = LogisticRegression(multi_class='multinomial', C=1.0, solver='lbfgs', max_iter = 1000).fit(scaled_X_train, np.squeeze(y_train))
metrics.accuracy_score(y_test, mul_lr.predict(scaled_X_test))

from sklearn.metrics import classification_report
from sklearn import metrics
y_pred = mul_lr.predict(scaled_X_test)
print(classification_report(y_test, y_pred))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[14]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, y_pred)


# In[16]:


# 2. Repeated K-Fold Cross-Validation

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import brier_score_loss

df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Preparing Target Variable\Census22_Quantiles.csv')

df1 = df.loc[:,['PEHSPNON','PRDTRACE','A_HGA','A_MJOCC','CAP_VAL','DIV_VAL','A_CLSWKR','A_WKSTAT','HHDFMX','NOEMP','Income']]  

data = df1.iloc[:,:-1]
Income = df1["Income"]
class_names = np.unique(df1["Income"])
print(class_names)
labels, counts = np.unique(Income, return_counts=True)


# In[17]:


### Decision tree

import timeit
start = timeit.default_timer()

def evaluate_model_DT(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.tree import DecisionTreeClassifier
    dt_clf1 = DecisionTreeClassifier()
    classifier = dt_clf1.fit(train_x, train_y)

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes

predicted_income, actual_income = evaluate_model_DT(data, Income)
print(classification_report(predicted_income, actual_income))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[18]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(predicted_income, actual_income)


# In[22]:


### Random Forest

import timeit
start = timeit.default_timer()

def evaluate_model_RF(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=10, max_depth=4)
    classifier = RFC.fit(train_x, np.squeeze(train_y))

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes

predicted_income, actual_income = evaluate_model_RF(data, Income)
print(classification_report(predicted_income, actual_income))


stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[23]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(predicted_income, actual_income)


# In[34]:


### LogisticRegression

import timeit
start = timeit.default_timer()


def evaluate_model_LR(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    from sklearn import preprocessing
    import pandas as pd
    scaler = preprocessing.MinMaxScaler()

    d_X_train = scaler.fit_transform(train_x)
    scaled_X_train = pd.DataFrame(d_X_train)

    d_X_test = scaler.fit_transform(test_x)
    scaled_X_test = pd.DataFrame(d_X_test)    
        
    # Fit the classifier
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(multi_class='multinomial', C=1.0, solver='lbfgs', max_iter = 1000)
    classifier = LR.fit(scaled_X_train, np.squeeze(train_y))

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(scaled_X_test)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes

predicted_income, actual_income = evaluate_model_LR(data, Income)
print(classification_report(predicted_income, actual_income))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[35]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(predicted_income, actual_income)


# In[29]:


### KNN

import timeit
start = timeit.default_timer()

def evaluate_model_KNN(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf1 = KNeighborsClassifier(n_neighbors = 11)
    classifier = knn_clf1.fit(train_x, np.squeeze(train_y))

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes

predicted_income, actual_income = evaluate_model_KNN(data, Income)
print(classification_report(predicted_income, actual_income))

stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[30]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(predicted_income, actual_income)


# In[31]:


### Naive Bayes

import timeit
start = timeit.default_timer()

def evaluate_model_NB(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.naive_bayes import GaussianNB
    nb_clf1 = GaussianNB()
    classifier = nb_clf1.fit(train_x, np.squeeze(train_y))

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes

predicted_income, actual_income = evaluate_model_NB(data, Income)
print(classification_report(predicted_income, actual_income))


stop = timeit.default_timer()
print('Time: ', stop - start) 


# In[32]:


from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(predicted_income, actual_income)


# In[ ]:





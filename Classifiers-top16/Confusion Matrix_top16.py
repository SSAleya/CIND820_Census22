#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import brier_score_loss

df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Census22_2QT.csv')
# top16 =["PEHSPNON", "PEMLR", "A_SEX", "A_HGA", "A_AGE", "A_MJIND", "A_MJOCC","CAP_VAL","DIV_VAL", "STATETAX_A", "STATETAX_B", "A_CLSWKR","A_WKSTAT","HHDFMX","WKSWORK""NOEMP", "Income"]

df1 = df.loc[:,["PEHSPNON", "PEMLR", "A_SEX", "A_HGA", "A_AGE", "A_MJIND", "A_MJOCC","CAP_VAL","DIV_VAL", "STATETAX_A", "STATETAX_B", "A_CLSWKR","A_WKSTAT","HHDFMX","WKSWORK", "NOEMP", "Income"]]

data = df1.iloc[:,:-1]
Income = df1["Income"]
class_names = np.unique(df1["Income"])
print(class_names)
labels, counts = np.unique(Income, return_counts=True)
labels, counts


# In[2]:


def plot_confusion_matrix(predicted_labels_list, y_test_list):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix')
    plt.show()


# In[3]:


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    print('Confusion matrix')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Income')
    plt.xlabel('Predicted Income')

    return cnf_matrix


# In[4]:



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


# In[5]:


predicted_income, actual_income = evaluate_model_DT(data, Income)
plot_confusion_matrix(predicted_income, actual_income)


# In[6]:


def evaluate_model_RF(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.ensemble import RandomForestClassifier
    RFC = RandomForestClassifier(n_estimators=10)
    classifier = RFC.fit(train_x, train_y.ravel())

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes


# In[7]:


predicted_income, actual_income = evaluate_model_RF(data, Income)
plot_confusion_matrix(predicted_income, actual_income)


# In[12]:


def evaluate_model_LR(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.linear_model import LogisticRegression
    lr_clf1 = LogisticRegression(C=1.0, solver='lbfgs')
    classifier = lr_clf1.fit(train_x, train_y.ravel())

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes


# In[13]:


predicted_income, actual_income = evaluate_model_LR(data, Income)
plot_confusion_matrix(predicted_income, actual_income)


# In[14]:


def evaluate_model_KNN(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.neighbors import KNeighborsClassifier
    knn_clf1 = KNeighborsClassifier(n_neighbors = 11)
    classifier = knn_clf1.fit(train_x, train_y.ravel())

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes


# In[15]:


predicted_income, actual_income = evaluate_model_KNN(data, Income)
plot_confusion_matrix(predicted_income, actual_income)


# In[16]:


def evaluate_model_NB(data_x, data_y):
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)

    predicted_incomes = np.array([])
    actual_incomes = np.array([])

    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(df1.values[:,:-1], df1.values[:,-1:])
        
    # Fit the classifier
    from sklearn.naive_bayes import GaussianNB
    nb_clf1 = GaussianNB()
    classifier = nb_clf1.fit(train_x, train_y.ravel())

    # Predict the labels of the test set samples
    predicted_labels = classifier.predict(test_x)

    predicted_incomes = np.append(predicted_incomes, predicted_labels)
    actual_incomes = np.append(actual_incomes, test_y)

    return predicted_incomes, actual_incomes


# In[17]:


predicted_income, actual_income = evaluate_model_NB(data, Income)
plot_confusion_matrix(predicted_income, actual_income)


# In[25]:


# 1.HoldOut Cross-validation or Train-Test Split
# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df1.values[:,:-1],
    df1.values[:,-1:],
    test_size=0.20,
    random_state=42)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()

dt_clf.fit(X_train, y_train)
predicted_income = dt_clf.predict(X_test)
actual_income = y_test


plot_confusion_matrix(predicted_income, actual_income)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100)

rf_clf.fit(X_train, np.squeeze(y_train))
predicted_income = rf_clf.predict(X_test)
actual_income = y_test

plot_confusion_matrix(predicted_income, actual_income)


# In[29]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

nb_clf.fit(X_train, np.squeeze(y_train))
predicted_income = nb_clf.predict(X_test)
actual_income = y_test

plot_confusion_matrix(predicted_income, actual_income)


# In[30]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 11)

knn_clf.fit(X_train, np.squeeze(y_train))
predicted_income = knn_clf.predict(X_test)
actual_income = y_test

plot_confusion_matrix(predicted_income, actual_income)


# In[32]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0, solver='lbfgs')

# Train multinomial logistic regression model
mul_lr = model.fit(X_train, np.squeeze(y_train))
predicted_income = mul_lr.predict(X_test)
actual_income = y_test

plot_confusion_matrix(predicted_income, actual_income)


# In[ ]:





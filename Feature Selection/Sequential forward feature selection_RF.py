#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install mlxtend  ')


# In[3]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs


# Read data
import numpy as np
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Sequential forward selection\Census22_40.csv')
y = df["Income"]
X = df.iloc[ : , 0:-1]


# In[4]:


# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

### Theory : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=random+forest+classifier


# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=(1, 39),    # subset of features we are looking to select (k_features=23)
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)     # desired folds 10

# Perform SFFS
sfs1 = sfs1.fit(X, y)


# In[5]:


# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)


# In[15]:


# Read data
import numpy as np
import pandas as pd
df =pd.read_csv (r'C:\Users\sania\Data analytics\Course materials 820\Latest census\Initial Data analysis\Sequential forward selection\Census22_40.csv')

# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.values[:,:-1],
    df.values[:,-1:],
    test_size=0.25,
    random_state=42)


from sklearn.linear_model import Lasso
lasso = Lasso(alpha = 1)
lasso.fit(X_train, y_train)
y_pred1 = lasso.predict(X_test)
 
X_train = pd.DataFrame(X_train, columns = X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
    
mean_squared_error = np.mean((y_pred1 - y_test)**2)
print("MSE on test set", mean_squared_error)
lasso_coeff = pd.DataFrame()
lasso_coeff["Columns"] = X_train.columns
lasso_coeff['Coefficient Estimate'] = pd.Series(lasso.coef_)
 
print(lasso_coeff)


# In[ ]:





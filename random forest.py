# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 18:52:26 2018

@author: Blosher Brar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, 20].values


#convert strings to float
a = pd.Series(X[:,1])
b = pd.Series(X[:,4])
c = pd.Series(X[:,-2])
d = pd.Series(X[:,-1])
df = pd.DataFrame({'a':a,'b':b,'c':c,'d':d,})
df
cols_to_convert = ['a','b','c','d']
cols_to_convert
for col in cols_to_convert:
  df[col] = pd.to_numeric(df[col], errors='coerce')
df.dtypes
X1=df.values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X1[:,2:4])
X1[:,2:4] = imputer.transform(X1[:,2:4])

X[:,1]=X1[:,0]
X[:,4]=X1[:,1]
X[:,-2]=X1[:,-2]
X[:,-1]=X1[:,-1]
X

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,15] = labelencoder_X.fit_transform(X[:,15])

X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,7] = labelencoder_X.fit_transform(X[:,7])
X[:,8] = labelencoder_X.fit_transform(X[:,8])
X[:,9] = labelencoder_X.fit_transform(X[:,9])
X[:,10] = labelencoder_X.fit_transform(X[:,10])
X[:,11] = labelencoder_X.fit_transform(X[:,11])
X[:,12] = labelencoder_X.fit_transform(X[:,12])
X[:,13] = labelencoder_X.fit_transform(X[:,13])
X[:,14] = labelencoder_X.fit_transform(X[:,14])
X[:,16] = labelencoder_X.fit_transform(X[:,16])

onehotencoder = OneHotEncoder(categorical_features = [6,7,8,9,10,11,12,13,14,16])
X= onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 72, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{ 'n_estimators': [70, 71, 72,73,74,75,76,77,78,79]},
             ]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_










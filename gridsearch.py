
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
from decimal import Decimal
from sklearn import linear_model
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[3]:


def getData():
    dtype = {'BitRate': np.float64, 'FreezeRatio': np.float64, 'Quality': np.float64}
    df_all = pd.read_table('../data.txt', delim_whitespace=True, dtype = dtype)
    X, Y = np.array(df_all[['BitRate', 'FreezeRatio']]), np.array(df_all['Quality'])
    return X, Y
X, Y = getData() # Modify this function as per your need to get X and Y data.


# In[4]:


classifiers = {'Random Forest': RandomForestClassifier(random_state = 1), 
               'Nearest Neighbors': KNeighborsClassifier(),
               'Naive Bayers': GaussianNB(),
               'SVM': SVC(probability = True),
               'MLP': MLPClassifier(random_state = 1, max_iter = 10000),
               'AdaBoost': AdaBoostClassifier(random_state = 1)
              }

params = {'Random Forest': {'n_estimators': range(1, 21), 'criterion': ('gini', 'entropy')},
          'Nearest Neighbors': {'n_neighbors':range(1, 11)},
          'Naive Bayers': {},
          'SVM': {'kernel':('poly', 'rbf', 'sigmoid'), 'C':[0.1, 1, 10]},
          'MLP': {'hidden_layer_sizes': [(5,), (10,), (25,), (50,), (100,), (200,)], 
                  'alpha': [0.0001, 0.001],
                  'activation': ('logistic', 'tanh', 'relu'), 
                  'solver': ('lbfgs', 'sgd', 'adam')},
          'AdaBoost': {'n_estimators': [10, 25, 50, 100], 'learning_rate': [0.01, 0.1, 1, 10]}
         }


# In[7]:


fold = 10
accuracies = {}
confusion = {}
# function for obtaining best estimator using grid search
def grid_search(estimator, params):
    clf = GridSearchCV(estimator, params)
    clf.fit(X, Y)
    return (clf.best_estimator_, clf.best_score_)

# function for performing k-fold cross validation 
def k_Fold_CV(estimator, n):
    accuracies = []
    confusion = []
    kf = KFold(n_splits = n)
    for train, test in kf.split(X):
        pred = estimator.fit(X[train], Y[train]).predict(X[test])
        accuracies.append(estimator.score(X[test], Y[test]))
        confusion.append(confusion_matrix(Y[test], pred))
    return (accuracies, confusion)


# In[ ]:


best_classifiers = {}
for k in classifiers.keys():
    best_classifiers[k], _ = grid_search(classifiers[k], params[k])
    accuracies[k], confusion[k] = k_Fold_CV(best_classifiers[k], fold)
for k, v in accuracies.items():
    print(k)
    print("{}_fold accuracies: ".format(fold), np.around(v, decimals = 3))
    print("Average accuracy: {0:0.3f}".format(np.mean(v)))
    print("#################################")


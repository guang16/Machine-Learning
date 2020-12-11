import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from PreProcess import processFile

white_wines = processFile('winequality-white.csv')

# Define features X
X = np.asarray(white_wines.iloc[:, :-1])
# Define target y
y = np.asarray(white_wines['quality'])

# Standardlize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset to training 90% and testing 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Number of k from 1 to 26
k_range = range(1, 26)
k_scores = []
# Calculate cross validation score for every k number from 1 to 26
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # It’s 10 fold cross validation with ‘accuracy’ scoring
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())


# print k to find the largest k_score
# for k in k_scores:
#     print(k)

# Train the model and predict for k=17
def dowork(kvalue):
    knn = KNeighborsClassifier(n_neighbors=kvalue)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # classification report for test set
    print(metrics.classification_report(y_test, y_pred, digits=3, zero_division=1))
    # Calculate cv score with 'accuracy' scoring and 10 folds
    accuracy = cross_val_score(knn, X, y, scoring='accuracy', cv=10)
    print('cross validation score', accuracy.mean())
    # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
    accuracy = cross_val_score(knn, X, y, scoring='roc_auc_ovr', cv=10)
    print('cross validation score with roc_auc', accuracy.mean())
    # Calculate roc_auc score with multiclass parameter
    print('roc_auc_score', roc_auc_score(y_test, knn.predict_proba(X_test), multi_class='ovr'))


print('------------------- begin k = 17')
dowork(17)
print('-------------------- end k = 17')

print('------------------- begin k = 18')
dowork(18)
print('-------------------- end k = 18')

print('------------------- begin k = 25')
dowork(25)
print('-------------------- end k = 25')

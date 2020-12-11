import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

# import the CSV
white_wines = pd.read_csv('winequality-white.csv')

# pre-processing the data, include checking entries, features, checking null values.
# print(white_wines.shape)
# white_wines.head()
# white_wines.isnull().any()
# white_wines.describe()

# Check to see if data under residual sugar has outlier, because max = 65.8, min = 0.6 and avg = 6.39
# sns.boxplot(white_wines['residual sugar'])

# Use Z-Score to remove the outlier and check the remaining entries to be (4487)
z = np.abs(stats.zscore(white_wines))
white_wines = white_wines[(z < 3).all(axis=1)]
# print(white_wines.shape)

# Define features X
X = np.asarray(white_wines.iloc[:, :-1])
# Define target y
y = np.asarray(white_wines['quality'])

# Standardlize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset to training 90% and testing 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

def do_work():
    clf = MLPClassifier(random_state=1, max_iter=300)
    clf.fit(X_train, y_train)
    # Test out-of-sample test set
    y_pred = clf.predict(X_test)
    # Classification report
    print(metrics.classification_report(y_test, y_pred, digits=3))
    print('accuracy', accuracy_score(y_test, y_pred))
    # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
    accuracy = cross_val_score( clf, X, y, scoring='roc_auc_ovr', cv=10)
    print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
    # Calculate roc_auc score with multiclass parameter
    print('roc_auc_score', roc_auc_score(y_test,  clf.predict_proba(X_test), multi_class='ovr'))

do_work()





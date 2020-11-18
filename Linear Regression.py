import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def polynomial_regression(degree, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LogisticRegression(multi_class='multinomial', solver='newton-cg', **kwargs))


def do_work(degree):
    poly = polynomial_regression(degree)
    poly.fit(X_train, y_train)
    # Test out-of-sample test set
    y_pred = poly.predict(X_test)
    # Classification report
    print(metrics.classification_report(y_test, y_pred, digits=3))
    # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
    accuracy = cross_val_score(poly, X, y, scoring='roc_auc_ovr', cv=10)
    print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
    # Calculate roc_auc score with multiclass parameter
    print('roc_auc_score', roc_auc_score(y_test, poly.predict_proba(X_test), multi_class='ovr'))


# import the CSV
white_wines = pd.read_csv('winequality-white.csv')

# pre-processing the data, include checking entries, features, checking null values.
# #print(white_wines.shape)
# white_wines.head()
# white_wines.isnull().any()
# white_wines.describe()

# Check to see if data under residual sugar has outlier, because max = 65.8, min = 0.6 and avg = 6.39
# sns.boxplot(white_wines['residual sugar'])

# Use Z-Score to remove the outlier and check the remaining entries to be (4487)
# z = np.abs(stats.zscore(white_wines))
# white_wines = white_wines[(z < 3).all(axis=1)]
# #print(white_wines.shape)

# Check correlation between column.
# plt.subplots(figsize=(15, 10))
# sns.heatmap(white_wines.corr(), annot = True, cmap = 'coolwarm')

# Define features X
X = np.asarray(white_wines.iloc[:, :-1])
# Define target y
y = np.asarray(white_wines['quality'])

# Standardlize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset to training 90% and testing 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#print('Train set:', X_train.shape, y_train.shape)
#print('Test set:', X_test.shape, y_test.shape)

# Algorithm 1, Linear regression
# Train and fit model
print('Begin of Degree 1 Linear regression--------------------')
logreg = LogisticRegression(multi_class='multinomial', solver='newton-cg')
logreg.fit(X_train, y_train)
# Predict out-of-sample test set
y_pred = logreg.predict(X_test)
# classification report
print(metrics.classification_report(y_test, y_pred, digits=3, zero_division=1))
print('accuracy', accuracy_score(y_test, y_pred))
# Calculate cv score with ‘roc_auc_ovr’ scoring and 10 folds
accuracy = cross_val_score(logreg, X, y, scoring='roc_auc_ovr', cv=10)
print('cross validation score with roc_auc', accuracy.mean())
# Calculate roc_auc score with multiclass parameter
print('roc_auc_score', roc_auc_score(y_test, logreg.predict_proba(X_test), multi_class='ovr'))
print('End of Degree 1 Linear regression--------------------')

# Degree 2 Polynomial
print('Begin of Degree 2 Linear regression--------------------')
do_work(2)
print('End of Degree 2 Linear regression--------------------')
# Train and fit the 3rd degree polynomial regression model
# poly = polynomial_regression(2)
# poly.fit(X_train, y_train)
# # Test out-of-sample test set
# y_pred = poly.predict(X_test)
# # Classification report
# print(metrics.classification_report(y_test, y_pred, digits=3))
# # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
# accuracy = cross_val_score(poly, X, y, scoring='roc_auc_ovr', cv=10)
# print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
# # Calculate roc_auc score with multiclass parameter
# print('roc_auc_score', roc_auc_score(y_test, poly.predict_proba(X_test), multi_class='ovr'))

# Degree 3 Polynomial
print('Begin of Degree 3 Linear regression--------------------')
do_work(3)
print('End of Degree 3 Linear regression--------------------')
# Train and fit the 3rd degree polynomial regression model
# poly = polynomial_regression(3)
# poly.fit(X_train, y_train)
# # Test out-of-sample test set
# y_pred = poly.predict(X_test)
# # Classification report
# print(metrics.classification_report(y_test, y_pred, digits=3))
# # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
# accuracy = cross_val_score(poly, X, y, scoring='roc_auc_ovr', cv=10)
# print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
# # Calculate roc_auc score with multiclass parameter
# print('roc_auc_score', roc_auc_score(y_test, poly.predict_proba(X_test), multi_class='ovr'))

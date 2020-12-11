import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from PreProcess import processFile


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
    print('accuracy', accuracy_score(y_test, y_pred))
    # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
    accuracy = cross_val_score(poly, X, y, scoring='roc_auc_ovr', cv=10)
    print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
    # Calculate roc_auc score with multiclass parameter
    print('roc_auc_score', roc_auc_score(y_test, poly.predict_proba(X_test), multi_class='ovr'))


white_wines = processFile('winequality-white.csv')

# Define features X
X = np.asarray(white_wines.iloc[:, :-1])
# Define target y
y = np.asarray(white_wines['quality'])

# Standardlize the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset to training 90% and testing 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

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

# Degree 3 Polynomial
print('Begin of Degree 3 Linear regression--------------------')
do_work(3)
print('End of Degree 3 Linear regression--------------------')

import numpy as np
from sklearn import preprocessing, metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier

from PreProcess import processFile

white_wines = processFile('winequality-white.csv')
# Define features X
X = np.asarray(white_wines.iloc[:, :-1])
# Define target y
y = np.asarray(white_wines['quality'])

# Standardized the dataset
X = preprocessing.StandardScaler().fit(X).transform(X)

# Split the dataset to training 90% and testing 10%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


def do_work(hiddenLayer, activation):
    clf = MLPClassifier(hidden_layer_sizes=hiddenLayer, activation=activation, random_state=1, max_iter=3000)
    clf.fit(X_train, y_train)
    # Test out-of-sample test set
    y_pred = clf.predict(X_test)
    # Classification report
    print(metrics.classification_report(y_test, y_pred, digits=3))
    print('accuracy', accuracy_score(y_test, y_pred))
    # Calculate cv score with 'roc_auc_ovr' scoring and 10 folds
    accuracy = cross_val_score(clf, X, y, scoring='roc_auc_ovr', cv=10)
    print('cross validation score with roc_auc_ovr scoring', accuracy.mean())
    # Calculate roc_auc score with multiclass parameter
    print('roc_auc_score', roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))


print('1 layer, 100 hidden nodes, logistic activation function')
do_work(100, 'logistic')
print('End Scenario 1')

print('1 layer, 100 hidden nodes, RELU activation function')
do_work(100, 'relu')
print('End Scenario 2')

print('3 Layer, 100 hidden nodes, logistic activation function')
do_work((100, 100, 100), 'logistic')
print('End Scenario 3')

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 17:57:15 2018

@author: Alexandra Russell
"""
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import itertools

wine = pd.read_csv("winequality-red.csv")
data = wine.drop(['quality'], axis = 1)
target = wine['quality']

store = []
for i in range(10):
    model = DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size = 0.20)
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    store.append(metrics.accuracy_score(y_test,predicted))
    
print("The average accuracy score for the decision tree classifier is", sum(store)/len(store))

#function to create an image of the confusion matrix
#found here: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(title+".png")

#decision tree with parameters ['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
data = wine[['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
target = wine['quality']

model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size = 0.20)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print("Classify quality")
print("Accuracy score:", metrics.accuracy_score(y_test,predicted))
print("Classification report:")
print(metrics.classification_report(y_test,predicted))
confusion = (metrics.confusion_matrix(y_test,predicted))

plt.figure()
plot_confusion_matrix(confusion, classes=['3','4','5','6','7', '8'],
                      title='decisiontreeconfusion')

#decision tree with parameters 'volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'sulphates', 'alcohol' and a binary outcome
wine["binary quality"] = np.where(wine["quality"]<6, 'bad', 'good')
data = wine[['volatile acidity', 'residual sugar', 'chlorides', 'total sulfur dioxide', 'sulphates', 'alcohol']]
target = wine['binary quality']

model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size = 0.20)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
print("Classify binary quality")
print("Accuracy score:", metrics.accuracy_score(y_test,predicted))
print("Classification report:")
print(metrics.classification_report(y_test,predicted))
confusion = (metrics.confusion_matrix(y_test,predicted))

plt.figure()
plot_confusion_matrix(confusion, classes=['good', 'bad'],
                      title='decisiontreebinary')
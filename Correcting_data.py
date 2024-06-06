import numpy as np
import copy
import matplotlib.pyplot as plt
from Test_cases_v2 import *
from public_test import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Planer_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

shape_X = np.shape(X)
shape_Y = np.shape(Y)
m = np.shape(X[1])

# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);


# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")
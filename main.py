import numpy as np
from matplotlib import pyplot as plt

import Forward_Propagataion as fp
import Correcting_data as cd
import Predict_Model as pm
from Backward_Propagation import nn_model
from Planer_utils import plot_decision_boundary
import sklearn

# X, Y = cd.load_planar_dataset()
#
#
# # Visualize the data:
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
# plt.show()
#
# shape_X = np.shape(X)
# shape_Y = np.shape(Y)
# m = np.shape(X[1])
#
# clf = sklearn.linear_model.LogisticRegressionCV();
# clf.fit(X.T, Y.T);
#
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")
# plt.show()
#
# # Print accuracy
# LR_predictions = clf.predict(X.T)
# print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
#        '% ' + "(percentage of correctly labelled datapoints)")

# Build a model with a n_h-dimensional hidden layer
parameters = nn_model(cd.X, cd.Y, n_h = 4, num_iterations = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: pm.predict(parameters, x.T), cd.X, cd.Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# *********** Run the following code(it may take 1-2 minutes).
# Then, observe different behaviors of the model for various hidden layer sizes. **********************

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5]

# you can try with different hidden layer sizes
# but make sure before you submit the assignment it is set as "hidden_layer_sizes = [1, 2, 3, 4, 5]"
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(cd.X, cd.Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: pm.predict(parameters, x.T), cd.X, cd.Y)
    predictions = pm.predict(parameters, cd.X)
    accuracy = float((np.dot(cd.Y,predictions.T) + np.dot(1 - cd.Y, 1 - predictions.T)) / float(cd.Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    plt.show()
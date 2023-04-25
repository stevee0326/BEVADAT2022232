from sklearn import svm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
#print(cancer)

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer
print("Label types: ", cancer.target_names)

# print the cancer data (top 3 records)
print(cancer.data[:3])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target_names)
#print(cancer.target)

# Split dataset into training set and test set (sklearn train_test_split)
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# Create a svm Classifier: clf (https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
# Use linear kernel
classifier = SVC(kernel='linear')

# Train (fit) the model using the training sets
classifier.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = classifier.predict(X_test)

# Model Accuracy: how often is the classifier correct?
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


#Tananyag
# # Model Precision: what percentage of positive predictions are truly positive?
# 
# 
# # Model Recall: what percentage of positive datapoints are labelled as such?
# #### Tuning Hyperparameters
# * **Kernel:** The main function of the kernel is to transform the given dataset input data into the required form. There are various types of functions such as linear, polynomial, and radial basis function (RBF). Polynomial and RBF are useful for non-linear hyperplane. Polynomial and RBF kernels compute the separation line in the higher dimension. In some of the applications, it is suggested to use a more complex kernel to separate the classes that are curved or nonlinear. This transformation can lead to more accurate classifiers.
# * **Regularization:** Regularization parameter in python's Scikit-learn C parameter used to maintain regularization. Here C is the penalty parameter, which represents misclassification or error term. The misclassification or error term tells the SVM optimization how much error is bearable. This is how you can control the trade-off between decision boundary and misclassification term. A smaller value of C creates a small-margin hyperplane and a larger value of C creates a larger-margin hyperplane.
# * **Gamma:** A lower value of Gamma will loosely fit the training dataset, whereas a higher value of gamma will exactly fit the training dataset, which causes over-fitting. In other words, you can say a low value of gamma considers only nearby points in calculating the separation line, while the a value of gamma considers all the data points in the calculation of the separation line.
#(https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],     #C -> mennyire megengedő a modell! Mennyire megengedő, hogy a hipersíkhoz közel essenek
'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
'kernel': ['rbf']}

# Check how the function GridSearchCV works
svc = SVC()
# Use it with estimator SVC and with the above-given param_grid
grid_search = GridSearchCV(svc, param_grid)
print(clf)

# Set the verbose parameter to at least 1
grid_search = GridSearchCV(svc, param_grid, verbose=1)

# Fit the created grid model on your train data
grid_search.fit(X_train, y_train)

# Print best parameter after tuning 
# (your created grid model has a function named best_params_)
print("Best parameters:", grid_search.best_params_)

# Print how our model looks after hyper-parameter tuning
# (check the best_estimator_ function)
print("Best estimator:", grid_search.best_estimator_)

# Predict with the help of your new model: grid_predictions
# As usual, this model also has a 'predict' function
grid_predictions = grid_search.best_estimator_.predict(X_test)
print(grid_predictions)

# Evaluate your model: print its accuracy, precision and recall values
accuracy = metrics.accuracy_score(y_test, grid_predictions)
print("Accuracy:", accuracy)



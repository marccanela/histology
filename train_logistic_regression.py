'''
Created by @mcanela on Tuesday, 13/02/2024
'''

# Specify data
directory = '//folder/becell/Macro tests/List of images/ROIs to analyze/'
ratio = 1.55 # px/µm

# Select your ROIs
from master_script import create_dict_of_binary
dict_of_binary = create_dict_of_binary(directory)

# Train you model
from master_script import logistic_regression
log_reg, X_train, X_test, y_train, y_test = logistic_regression(dict_of_binary, ratio)

import pickle as pk
with open(directory + 'log_reg.pkl', 'wb') as file:
    pk.dump(log_reg, file)
with open(directory + 'train_data.pkl', 'wb') as file:
    pk.dump([X_train, X_test, y_train, y_test], file)




regressor = log_reg.named_steps['log_reg']

# Cross validation
from sklearn.model_selection import cross_val_score
cross_val_score(regressor, X_train, y_train, cv=3, scoring="accuracy")

# Confusion matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(regressor, X_train, y_train, cv=3)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train, y_train_pred)    

# Precision and recall
from sklearn.metrics import precision_score, recall_score
precision_score(y_train, y_train_pred)
recall_score(y_train, y_train_pred)

from sklearn.metrics import f1_score
f1_score(y_train, y_train_pred)

# ROC curve
from sklearn.metrics import roc_auc_score
y_scores = cross_val_predict(regressor, X_train, y_train, cv=3, method="decision_function")   
roc_auc_score(y_train, y_scores)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# Plot the decision boundary
import numpy as np
import matplotlib.pyplot as plt
def plot_boundary(log_reg, X=X_train, y=y_train):
    plt.figure(figsize=(8, 6))

    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap='Pastel1', alpha=0.8)
    
    # Scatter plot of data points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='Pastel1', marker='o')
    plt.title('Logistic Regression with PCA Decision Boundary')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    plt.show()


































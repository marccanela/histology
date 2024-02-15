'''
Created by @mcanela on Tuesday, 13/02/2024
'''

# Specify data
directory = '//folder/becell/Macro tests/List of images/ROIs to analyze/'
ratio = 1.55 # (1.55 px/µm at 10X)
layer = 'layer_2' # Select layer_0, layer_1, layer_2, etc.


# Select your ROIs
from master_script import create_dict_of_binary
dict_of_binary = create_dict_of_binary(directory, layer)

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
from master_script import plot_boundary
plot_boundary(log_reg, X_train, y_train)


































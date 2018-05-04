"""Description: This script allows the user to perform cross-validation to tune
a ridge regression model using imputed training data. The dictionary of best 
hyper-parameters from cross-validation will be saved.
"""
import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import pickle
# these are imported functions created for this package that involve splitting datasets or performing cross-validation 
# see data_split_tune_utils.py
from data_split_tune_utils import cross_validation_splits, X_y_site_split, cross_validation

np.random.seed(1)

train = pd.read_csv('../data/train_ridgeImp.csv')
#train = pd.read_csv('../data/train_rfImp.csv')

# drop rows with no monitor data response value
train = train.dropna(axis=0)

# model to tune in cross-validation
ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)

# hyper-parameters to test in cross-validation
parameter_grid_ridge = {'alpha' : [0.1, 0.01, 0.001, 0.0001, 0.00001]}

# run cross-validation
cv_r2, best_hyperparams = cross_validation(data=train, model=ridge, hyperparam_dict=parameter_grid_ridge, num_folds=4, y_var_name='MonitorData', site_var_name='site')
print('Cross-validation R^2: ' + str(cv_r2))
print('Best hyper-parameters: ' + str(best_hyperparams))

# save dictionary with best hyper-parameter combination
pickle.dump(best_hyperparams, open('best_ridge_hyperparams.pkl', 'wb'))
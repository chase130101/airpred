"""Description: This script allows the user to perform cross-validation to tune
a random forest regression model using imputed training data. The dictionary of best 
hyper-parameters from cross-validation will be saved.
"""
import pandas as pd
import numpy as np
import sklearn.ensemble
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
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)

# hyper-parameters to test in cross-validation
parameter_grid_rf = {'max_features' : [10, 15, 20, 25]}

# run cross-validation
cv_r2, best_hyperparams = cross_validation(data=train, model=rf, hyperparam_dict=parameter_grid_rf, num_folds=4, y_var_name='MonitorData', site_var_name='site')
print('Cross-validation R^2: ' + str(cv_r2))
print('Best hyper-parameters: ' + str(best_hyperparams))

# save dictionary with best hyper-parameter combination
pickle.dump(best_hyperparams, open('best_rf_hyperparams.pkl', 'wb'))
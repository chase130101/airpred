import pandas as pd
import numpy as np
from data_split_utils import cross_validation_splits, X_y_site_split
import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics

train = pd.read_csv('train_ridgeImp.csv')
test = pd.read_csv('test_ridgeImp.csv')

train = train.dropna(axis = 0)
test = test.dropna(axis = 0)

train.reset_index(inplace=True, drop = True)
cv_splits = cross_validation_splits(train, 5, site_var_name='site')

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

ridge = sklearn.linear_model.Ridge(random_state = 1, normalize=True)
parameter_grid_ridge = {'alpha' : [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}
grid_ridge = sklearn.model_selection.GridSearchCV(ridge, parameter_grid_ridge, cv = cv_splits, refit = 'r2', verbose=1)
grid_ridge.fit(train_x, train_y)
test_pred_ridge = grid_ridge.predict(test_x)
test_r2_ridge = sklearn.metrics.r2_score(test_y, test_pred_ridge)
print('Best regularization hyperparameter: ' + str(grid_ridge.best_params_))
print('Test R^2: ' + str(test_r2_ridge))
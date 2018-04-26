import pandas as pd
import numpy as np
from data_split_utils import X_y_site_split
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing

train = pd.read_csv('../data/train_ridgeImp.csv')
test = pd.read_csv('../data/test_ridgeImp.csv')

train = train.dropna(axis = 0)
test = test.dropna(axis = 0)

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

alpha = 0.001
ridge = sklearn.linear_model.Ridge(random_state = 1, normalize=True, alpha = alpha)
ridge.fit(train_x, train_y)
test_pred_ridge = ridge.predict(test_x)
test_r2_ridge = sklearn.metrics.r2_score(test_y, test_pred_ridge)
print('Regularization hyperparameter: ' + str(alpha))
print('Test R^2: ' + str(test_r2_ridge))
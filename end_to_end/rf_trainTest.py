import pandas as pd
import numpy as np
from data_split_utils import X_y_site_split
import sklearn.ensemble
import sklearn.metrics

train = pd.read_csv('../data/train_ridgeImp.csv')
test = pd.read_csv('../data/test_ridgeImp.csv')

train = train.dropna(axis = 0)
test = test.dropna(axis = 0)

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

max_features = 20
n_estimators = 500
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, criterion='mse', max_features=max_features, n_jobs=-1, random_state=1, verbose=1)
rf.fit(train_x, train_y)
test_pred_rf = rf.predict(test_x)
test_r2_rf = sklearn.metrics.r2_score(test_y, test_pred_rf)
print('Number of trees: ' + str(n_estimators))
print('Max features: ' + str(alpha))
print('Test R^2: ' + str(test_r2_rf))
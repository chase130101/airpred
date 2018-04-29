import pandas as pd
import numpy as np
import sklearn.ensemble
import sklearn.metrics
import pickle
from data_split_tune_utils import X_y_site_split

train = pd.read_csv('../data/train_ridgeImp.csv')
test = pd.read_csv('../data/test_ridgeImp.csv')
#train = pd.read_csv('../data/train_rfImp.csv')
#test = pd.read_csv('../data/test_rfImp.csv')
train = train.dropna(axis=0)
test = test.dropna(axis=0)
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

best_hyperparams = pickle.load(open('best_rf_hyperparams.pkl', 'rb'))
rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500, random_state=1, n_jobs=-1)
for key in list(best_hyperparams.keys()):
    setattr(rf, key, best_hyperparams[key])
    
rf.fit(train_x, train_y)
test_pred_rf = rf.predict(test_x)
test_r2_rf = sklearn.metrics.r2_score(test_y, test_pred_rf)
print('Test R^2: ' + str(test_r2_rf))
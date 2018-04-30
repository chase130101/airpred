import pandas as pd
import numpy as np
import sklearn.linear_model
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

best_hyperparams = pickle.load(open('best_ridge_hyperparams.pkl', 'rb'))
ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)
for key in list(best_hyperparams.keys()):
    setattr(ridge, key, best_hyperparams[key])
    
ridge.fit(train_x, train_y)
test_pred_ridge = ridge.predict(test_x)
test_r2_ridge = sklearn.metrics.r2_score(test_y, test_pred_ridge)
print('Test R^2: ' + str(test_r2_ridge))

test['MonitorData_pred'] = pd.Series(test_pred_ridge, index=test.index)
test.to_csv('../data/test_ridgePred.csv', index=False)
pickle.dump(ridge, open('ridge_final.pkl', 'wb'))
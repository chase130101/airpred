import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import pickle
from data_split_tune_utils import X_y_site_split

#train = pd.read_csv('../data/train_ridgeImp.csv')
#test = pd.read_csv('../data/test_ridgeImp.csv')
train = pd.read_csv('../data/train_rfImp.csv')
test = pd.read_csv('../data/test_rfImp.csv')

# drop rows with no monitor data response value
train = train.dropna(axis=0)
test = test.dropna(axis=0)

train = train.loc[:, ('site', 'MonitorData', 'Nearby_Peak2_PM25')]
test = test.loc[:, ('site', 'MonitorData', 'Nearby_Peak2_PM25')]

# split datasets into x, y, and site
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

# model to fit
ols = sklearn.linear_model.LinearRegression(normalize=True, fit_intercept=True)

# fit model on train data and make predictions on test data; compute test R^2
ols.fit(train_x, train_y)
test_pred_ols = ols.predict(test_x)
test_r2_ols = sklearn.metrics.r2_score(test_y, test_pred_ols)
print('Test R^2: ' + str(test_r2_ols))

# put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
test['MonitorData_pred'] = pd.Series(test_pred_ols, index=test.index)

# save test dataframe with predictions
test.to_csv('../data/test_olsPred_rfImp.csv', index=False)
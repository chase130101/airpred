from data_split_utils import train_test_split, X_y_site_split, cross_validation_splits
from predictiveImputer_mod import PredictiveImputer
import pandas as pd
import numpy as np
import pickle

np.random.seed(1)

data = pd.read_csv('../data/data_to_impute.csv')

train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

#https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
train_x['sin_time'] = np.sin(2*np.pi*train_x.month/12)
train_x['cos_time'] = np.cos(2*np.pi*train_x.month/12)
test_x['sin_time'] = np.sin(2*np.pi*test_x.month/12)
test_x['cos_time'] = np.cos(2*np.pi*test_x.month/12)

train_x = train_x.drop(['date', 'month'], axis=1)
test_x = test_x.drop(['date', 'month'], axis=1)

ridge_imputer = PredictiveImputer(max_iter=10, initial_strategy='mean', f_model='Ridge')
ridge_imputer.fit(train_x, alpha=0.01, fit_intercept=True, normalize=True, tol=0.001, random_state=1)

train_x_imp = ridge_imputer.transform(train_x)
test_x_imp = ridge_imputer.transform(test_x)

cols = ['site', 'MonitorData'] + list(train_x.columns)
train_imp_df = pd.DataFrame(np.concatenate([train_sites.values.reshape(len(train_sites), -1),\
                                              train_y.values.reshape(len(train_y), -1),\
                                              train_x_imp], axis=1),\
                                              columns = cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),\
                                              test_y.values.reshape(len(test_y), -1),\
                                              test_x_imp], axis=1),\
                                              columns = cols)

train_imp_df.to_csv('../data/train_ridgeImp.csv', index = False)
test_imp_df.to_csv('../data/test_ridgeImp.csv', index = False)
pickle.dump(ridge_imputer, open('ridge_imputer.pkl', 'wb'))
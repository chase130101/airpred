from data_split_utils import train_test_split, X_y_site_split, cross_validation_splits
from predictiveImputer_mod import PredictiveImputer
import pandas as pd
import numpy as np

np.random.seed(1)

data = pd.read_csv('../data/data_to_impute.csv', nrows=2000000)

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

rf_imputer = PredictiveImputer(max_iter=10, initial_strategy='mean', f_model='RandomForest')
rf_imputer.fit(train_x, max_features = 'sqrt', n_estimators = 50, n_jobs=-1, verbose=1, random_state=1)

train_x_imp = rf_imputer.transform(train_x)
test_x_imp = rf_imputer.transform(test_x)

cols = ['site', 'MonitorData'] + list(train_x.columns)
train_imp_df = pd.DataFrame(np.concatenate([train_sites.values.reshape(len(train_sites), -1),\
                                              train_y.values.reshape(len(train_y), -1),\
                                              train_x_imp], axis=1),\
                                              columns = cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),\
                                              test_y.values.reshape(len(test_y), -1),\
                                              test_x_imp], axis=1),\
                                              columns = cols)

train_imp_df.to_csv('train_rfImp.csv', index = False)
test_imp_df.to_csv('test_rfImp.csv', index = False)
pickle.dump(rf_imputer, open('rf_imputer.pkl', 'wb'))

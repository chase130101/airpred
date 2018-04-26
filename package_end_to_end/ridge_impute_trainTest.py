from data_split_utils import train_test_split, X_y_site_split
from predictiveImputer_mod import PredictiveImputer
import pandas as pd
import numpy as np
import pickle
import configparser


config = configparser.RawConfigParser()
config.read('config/py_config.ini')
config_section = "Ridge_Train_Test_Imputation"

np.random.seed(1)

data = pd.read_csv(config[config_section]["data_to_impute_path"])

train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')
train1, train2 = train_test_split(train, train_prop = 0.3, site_var_name = 'site')

train1_x, train1_y, train1_sites = X_y_site_split(train1, y_var_name='MonitorData', site_var_name='site')
train2_x, train2_y, train2_sites = X_y_site_split(train2, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

#https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
train1_x['sin_time'] = np.sin(2*np.pi*train1_x.month/12)
train1_x['cos_time'] = np.cos(2*np.pi*train1_x.month/12)
train2_x['sin_time'] = np.sin(2*np.pi*train2_x.month/12)
train2_x['cos_time'] = np.cos(2*np.pi*train2_x.month/12)
test_x['sin_time'] = np.sin(2*np.pi*test_x.month/12)
test_x['cos_time'] = np.cos(2*np.pi*test_x.month/12)

train1_x = train1_x.drop(['date', 'month'], axis=1)
train2_x = train2_x.drop(['date', 'month'], axis=1)
test_x = test_x.drop(['date', 'month'], axis=1)

ridge_imputer = PredictiveImputer(max_iter=10, initial_strategy='mean', f_model='Ridge')
ridge_imputer.fit(train1_x, alpha=0.001, fit_intercept=True, normalize=True, tol=0.001, random_state=1)

train1_x_imp = ridge_imputer.transform(train1_x)
train2_x_imp = ridge_imputer.transform(train2_x)
test_x_imp = ridge_imputer.transform(test_x)

cols = ['site', 'MonitorData'] + list(train1_x.columns)

train1_imp_df = pd.DataFrame(np.concatenate([train1_sites.values.reshape(len(train1_sites), -1),\
                                              train1_y.values.reshape(len(train1_y), -1),\
                                              train1_x_imp], axis=1),\
                                              columns = cols)

train2_imp_df = pd.DataFrame(np.concatenate([train2_sites.values.reshape(len(train2_sites), -1),\
                                              train2_y.values.reshape(len(train2_y), -1),\
                                              train2_x_imp], axis=1),\
                                              columns = cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),\
                                              test_y.values.reshape(len(test_y), -1),\
                                              test_x_imp], axis=1),\
                                              columns = cols)

train_imp_df = pd.concat([train1_imp_df, train2_imp_df])
train_imp_df = train_imp_df.reset_index().sort_values(['site', 'index'])
train_imp_df.drop('index', axis=1, inplace=True)
train_imp_df.reset_index(inplace=True, drop=True)

train_imp_df.to_csv(config[config_section]["train_imputed_ridge_path"], index = False)
test_imp_df.to_csv(config[config_section]["test_imputed_ridge_path"], index = False)
pickle.dump(ridge_imputer, open(config[config_section]["ridge_impute_model"], 'wb'))
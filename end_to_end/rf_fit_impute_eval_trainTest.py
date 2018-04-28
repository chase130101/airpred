import pandas as pd
import numpy as np
import pickle
from data_split_tune_utils import train_test_split, X_y_site_split
from predictiveImputer_mod import PredictiveImputer

np.random.seed(1)

data = pd.read_csv('../data/train.csv', nrows=100000)

train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')
train1, train2 = train_test_split(train, train_prop = 0.2, site_var_name = 'site')

train1_x, train1_y, train1_sites = X_y_site_split(train1, y_var_name='MonitorData', site_var_name='site')
train2_x, train2_y, train2_sites = X_y_site_split(train2, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

rf_imputer = PredictiveImputer(max_iter=10, initial_strategy='mean', f_model='RandomForest')
rf_imputer.fit(train1_x, max_features=15, n_estimators=50, n_jobs=-1, verbose=1, random_state=1)

train1_x_imp, train1_r2_scores = rf_imputer.transform(train1_x, evaluate = True)
train2_x_imp, train2_r2_scores = rf_imputer.transform(train2_x, evaluate = True)
test_x_imp, test_r2_scores = rf_imputer.transform(test_x, evaluate = True)

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

r2_scores_df = pd.DataFrame(np.concatenate([np.array(cols[2:]).reshape(len(cols)-2, -1),\
                                              np.array(train1_r2_scores).reshape(len(train1_r2_scores), -1),\
                                              np.array(train2_r2_scores).reshape(len(train2_r2_scores), -1),\
                                              np.array(test_r2_scores).reshape(len(test_r2_scores), -1)], axis=1),\
                                              columns = ['Variable', 'Train1_R2', 'Train2_R2', 'Test_R2'])

train_imp_df = pd.concat([train1_imp_df, train2_imp_df])
train_imp_df = train_imp_df.reset_index().sort_values(['site', 'index'])
train_imp_df.drop('index', axis=1, inplace=True)
train_imp_df.reset_index(inplace=True, drop=True)

r2_scores_df.to_csv('../data/r2_scores_rfImp.csv', index = False)
train_imp_df.to_csv('../data/train_rfImp.csv', index = False)
test_imp_df.to_csv('../data/test_rfImp.csv', index = False)
#pickle.dump(rf_imputer, open('rf_imputer.pkl', 'wb'))
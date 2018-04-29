import numpy as np
import pandas as pd
import pickle
from data_split_tune_utils import X_y_site_split

train = pd.read_csv('../data/trainV.csv')
val = pd.read_csv('../data/valV.csv')
test = pd.read_csv('../data/testV.csv')
ridge_imputer = pickle.load(open('ridge_imputer.pkl', 'rb'))

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
val_x, val_y, val_sites = X_y_site_split(val, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

train_x_imp, train_r2_scores = ridge_imputer.transform(train_x, evaluate = True, backup_impute_strategy = 'mean')
val_x_imp, val_r2_scores = ridge_imputer.transform(val_x, evaluate = True, backup_impute_strategy = 'mean')
test_x_imp, test_r2_scores = ridge_imputer.transform(test_x, evaluate = True, backup_impute_strategy = 'mean')

cols = ['site', 'MonitorData'] + list(train_x.columns)
train_imp_df = pd.DataFrame(np.concatenate([train_sites.values.reshape(len(train_sites), -1),\
                                              train_y.values.reshape(len(train_y), -1),\
                                              train_x_imp], axis=1),\
                                              columns = cols)

val_imp_df = pd.DataFrame(np.concatenate([val_sites.values.reshape(len(val_sites), -1),\
                                              val_y.values.reshape(len(val_y), -1),\
                                              val_x_imp], axis=1),\
                                              columns = cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),\
                                              test_y.values.reshape(len(test_y), -1),\
                                              test_x_imp], axis=1),\
                                              columns = cols)

r2_scores_df = pd.DataFrame(np.concatenate([np.array(cols[2:]).reshape(len(cols)-2, -1),\
                                              np.array(train_r2_scores).reshape(len(train_r2_scores), -1),\
                                              np.array(val_r2_scores).reshape(len(val_r2_scores), -1),\
                                              np.array(test_r2_scores).reshape(len(test_r2_scores), -1)], axis=1),\
                                              columns = ['Variable', 'Train_R2', 'Val_R2', 'Test_R2'])

r2_scores_df.loc[max(r2_scores_df.index)+1, :] = ['Mean_R2'] + list(np.nanmean(r2_scores_df.iloc[:, 1:].values.astype(np.float64), axis=0))

r2_scores_df.to_csv('../data/r2_scoresV_ridgeImp.csv', index = False)
train_imp_df.to_csv('../data/trainV_ridgeImp.csv', index = False)
val_imp_df.to_csv('../data/valV_ridgeImp.csv', index = False)
test_imp_df.to_csv('../data/testV_ridgeImp.csv', index = False)
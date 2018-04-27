import numpy as np
import pandas as pd
import pickle
from data_split_utils import X_y_site_split

train = pd.read_csv('../data/train.csv', index = False)
test = pd.read_csv('../data/test.csv', index = False)
ridge_imputer = pickle.load(open('ridge_imputer.pkl', 'rb'))

train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

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
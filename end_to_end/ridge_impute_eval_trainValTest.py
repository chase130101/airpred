import numpy as np
import pandas as pd
import pickle
from data_split_tune_utils import X_y_site_split

train = pd.read_csv('../data/trainV.csv')
val = pd.read_csv('../data/valV.csv')
test = pd.read_csv('../data/testV.csv')

# load fitted ridge imputer
ridge_imputer = pickle.load(open('ridge_imputer.pkl', 'rb'))

# split datasets into x, y, and site
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
val_x, val_y, val_sites = X_y_site_split(val, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

### make imputations on data matrices and create dataframes with imputation R^2 evaluations; computed weighted R^2 values
train_x_imp, train_r2_scores_df = ridge_imputer.transform(train_x, evaluate=True, backup_impute_strategy='mean')
train_r2_scores_df.columns = ['Train_R2', 'Train_num_missing']
train_r2_scores_df.loc[max(train_r2_scores_df.index)+1, :] = [np.average(train_r2_scores_df.loc[:, 'Train_R2'].values,
                                                                   weights=train_r2_scores_df.loc[:, 'Train_num_missing'].values,
                                                                   axis=0), np.mean(train_r2_scores_df.loc[:, 'Train_num_missing'].values)]

val_x_imp, val_r2_scores_df = ridge_imputer.transform(val_x, evaluate=True, backup_impute_strategy='mean')
val_r2_scores_df.columns = ['Val_R2', 'Val_num_missing']
val_r2_scores_df.loc[max(val_r2_scores_df.index)+1, :] = [np.average(val_r2_scores_df.loc[:, 'Val_R2'].values,
                                                                   weights=val_r2_scores_df.loc[:, 'Val_num_missing'].values,
                                                                   axis=0), np.mean(val_r2_scores_df.loc[:, 'Val_num_missing'].values)]

test_x_imp, test_r2_scores_df = ridge_imputer.transform(test_x, evaluate=True, backup_impute_strategy = 'mean')
test_r2_scores_df.columns = ['Test_R2', 'Test_num_missing']
test_r2_scores_df.loc[max(test_r2_scores_df.index)+1, :] = [np.average(test_r2_scores_df.loc[:, 'Test_R2'].values,
                                                                   weights=test_r2_scores_df.loc[:, 'Test_num_missing'].values,
                                                                   axis=0), np.mean(test_r2_scores_df.loc[:, 'Test_num_missing'].values)]

### convert imputed data matrices back into pandas dataframes with column names
cols = ['site', 'MonitorData'] + list(train_x.columns)
train_imp_df = pd.DataFrame(np.concatenate([train_sites.values.reshape(len(train_sites), -1),
                                              train_y.values.reshape(len(train_y), -1),
                                              train_x_imp], axis=1),
                                              columns=cols)

val_imp_df = pd.DataFrame(np.concatenate([val_sites.values.reshape(len(val_sites), -1),
                                              val_y.values.reshape(len(val_y), -1),
                                              val_x_imp], axis=1),
                                              columns=cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),
                                              test_y.values.reshape(len(test_y), -1),
                                              test_x_imp], axis=1),
                                              columns=cols)

# put R^2 evaluations for different datasets into same pandas dataframe
var_df = pd.DataFrame(np.array(cols[2:]+['Weighted_Mean_R2']).reshape(len(cols)-2+1, -1), columns=['Variable'])
r2_scores_df = pd.concat([var_df, train_r2_scores_df, val_r2_scores_df, test_r2_scores_df], axis=1)

# save evaluations and imputated datasets
r2_scores_df.to_csv('../data/r2_scoresV_ridgeImp.csv', index=False)
train_imp_df.to_csv('../data/trainV_ridgeImp.csv', index=False)
val_imp_df.to_csv('../data/valV_ridgeImp.csv', index=False)
test_imp_df.to_csv('../data/testV_ridgeImp.csv', index=False)
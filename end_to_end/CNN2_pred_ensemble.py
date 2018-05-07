import pandas as pd
import numpy as np
import torch
import pickle
from torch.autograd import Variable
import sklearn.preprocessing
import sklearn.metrics
# this imported function was created for this package to split datasets (see data_split_tune_utils.py)
from data_split_tune_utils import X_y_site_split
# these imported functions were created for this package for the purposes of data pre-processing for CNNs, training CNNs, 
# and evaluation of CNNs (see CNN_utils.py)
from CNN_utils import split_sizes_site, split_data, pad_stack_splits, get_monitorData_indices, r2, get_nonConst_vars, train_CNN
# this is an imported class that is one of our CNN architectures (see CNN_architecture.py)
from CNN_architecture import CNN2

# set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

### read in train, val, and test
train = pd.read_csv('../data/train_rfImp.csv')
test = pd.read_csv('../data/test_rfImp.csv')

### delete sites from datasets where all monitor outputs are nan
train_sites_all_nan_df = pd.DataFrame(np.isnan(train.groupby('site').sum()['MonitorData']))
train_sites_to_delete = list(train_sites_all_nan_df[train_sites_all_nan_df['MonitorData'] == True].index)
train = train[~train['site'].isin(train_sites_to_delete)]

test_sites_all_nan_df = pd.DataFrame(np.isnan(test.groupby('site').sum()['MonitorData']))
test_sites_to_delete = list(test_sites_all_nan_df[test_sites_all_nan_df['MonitorData'] == True].index)
test = test[~test['site'].isin(test_sites_to_delete)]

# split train, test into x, y, and sites
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

# get dataframes with non-constant features only
nonConst_vars = get_nonConst_vars(train, site_var_name='site', y_var_name='MonitorData', cutoff=1000)
train_x_nonConst = train_x.loc[:, nonConst_vars]
test_x_nonConst = test_x.loc[:, nonConst_vars]

# standardize all features
standardizer_all = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_all = standardizer_all.fit_transform(train_x)
test_x_std_all = standardizer_all.transform(test_x)

# standardize non-constant features
standardizer_nonConst = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_nonConst = standardizer_nonConst.fit_transform(train_x_nonConst)
test_x_std_nonConst = standardizer_nonConst.transform(test_x_nonConst)


# get split sizes for TEST data (splitting by site)
test_split_sizes = split_sizes_site(test_sites.values)

# get tuples by site
test_x_std_tuple_nonConst = split_data(torch.from_numpy(test_x_std_nonConst).float(), test_split_sizes, dim = 0)
test_x_std_tuple = split_data(torch.from_numpy(test_x_std_all).float(), test_split_sizes, dim = 0)
test_y_tuple = split_data(torch.from_numpy(test_y.values), test_split_sizes, dim = 0)

# get site sequences stacked into matrix to go through CNN
test_x_std_stack_nonConst = pad_stack_splits(test_x_std_tuple_nonConst, np.array(test_split_sizes), 'x')
test_x_std_stack_nonConst = Variable(torch.transpose(test_x_std_stack_nonConst, 1, 2))

# load trained model
cnn = pickle.load(open('cnn2_final_ridgeVImp.pkl', 'rb'))

# evaluate
batch_size = 128
test_r2, test_pred_cnn = r2(cnn, batch_size, test_x_std_stack_nonConst, test_x_std_tuple, test_y_tuple, get_pred=True)

print()
print('Test R^2: ' + str(test_r2))

# put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
test = test.dropna(axis=0)
test['MonitorData_pred'] = pd.Series(test_pred_cnn, index=test.index)

test.to_csv('../data/test_cnn2Pred_rfImp.csv', index=False)
import pandas as pd
import numpy as np
import torch
import pickle
from torch.autograd import Variable
import sklearn.preprocessing
import sklearn.metrics
from data_split_tune_utils import X_y_site_split
from CNN_utils import split_sizes_site, split_data, pad_stack_splits, get_monitorData_indices, r2, get_nonConst_vars, train_CNN
from CNN_architecture import CNN2

np.random.seed(1)
torch.manual_seed(1)

### read in train, val, and test
train = pd.read_csv('../data/trainV_ridgeImp.csv')
val = pd.read_csv('../data/valV_ridgeImp.csv')
test = pd.read_csv('../data/testV_ridgeImp.csv')

# combine train and validation sets into train set
train = pd.concat([train, val], axis=0, ignore_index=True)

### delete sites from datasets where all monitor outputs are NaN
train_sites_all_nan_df = pd.DataFrame(np.isnan(train.groupby('site').sum()['MonitorData']))
train_sites_to_delete = list(train_sites_all_nan_df[train_sites_all_nan_df['MonitorData'] == True].index)
train = train[~train['site'].isin(train_sites_to_delete)]

test_sites_all_nan_df = pd.DataFrame(np.isnan(test.groupby('site').sum()['MonitorData']))
test_sites_to_delete = list(test_sites_all_nan_df[test_sites_all_nan_df['MonitorData'] == True].index)
test = test[~test['site'].isin(test_sites_to_delete)]

### split train, val, and test into x, y, and sites
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

### get dataframes with non-constant features only
nonConst_vars = get_nonConst_vars(train, site_var_name='site', y_var_name='MonitorData', cutoff=1000)
train_x_nonConst = train_x.loc[:, nonConst_vars]
test_x_nonConst = test_x.loc[:, nonConst_vars]

### standardize all features
standardizer_all = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_all = standardizer_all.fit_transform(train_x)
test_x_std_all = standardizer_all.transform(test_x)

### standardize non-constant features
standardizer_nonConst = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_nonConst = standardizer_nonConst.fit_transform(train_x_nonConst)
test_x_std_nonConst = standardizer_nonConst.transform(test_x_nonConst)




### get split sizes for TRAIN data (splitting by site)
train_split_sizes = split_sizes_site(train_sites.values)

### get tuples by site
train_x_std_tuple_nonConst = split_data(torch.from_numpy(train_x_std_nonConst).float(), train_split_sizes, dim = 0)
train_x_std_tuple = split_data(torch.from_numpy(train_x_std_all).float(), train_split_sizes, dim = 0)
train_y_tuple = split_data(torch.from_numpy(train_y.values), train_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
train_x_std_stack_nonConst = pad_stack_splits(train_x_std_tuple_nonConst, np.array(train_split_sizes), 'x')
train_x_std_stack_nonConst = Variable(torch.transpose(train_x_std_stack_nonConst, 1, 2))


### get split sizes for TEST data (splitting by site)
test_split_sizes = split_sizes_site(test_sites.values)

### get tuples by site
test_x_std_tuple_nonConst = split_data(torch.from_numpy(test_x_std_nonConst).float(), test_split_sizes, dim = 0)
test_x_std_tuple = split_data(torch.from_numpy(test_x_std_all).float(), test_split_sizes, dim = 0)
test_y_tuple = split_data(torch.from_numpy(test_y.values), test_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
test_x_std_stack_nonConst = pad_stack_splits(test_x_std_tuple_nonConst, np.array(test_split_sizes), 'x')
test_x_std_stack_nonConst = Variable(torch.transpose(test_x_std_stack_nonConst, 1, 2))


num_epochs = 100
batch_size = 128
input_size_conv = train_x_std_nonConst.shape[1]
input_size_full = train_x_std_all.shape[1]

# CNN hyper-parameters
hidden_size_conv = 25
kernel_size = 3
padding = 1
hidden_size_full = 50
dropout_full = 0.1
hidden_size2_full = 50
dropout2_full = 0.1
lr = 0.1
weight_decay = 0.00001

# Loss function
mse_loss = torch.nn.MSELoss(size_average=True)

# instantiate CNN
cnn = CNN2(input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, 
           dropout_full, hidden_size2_full, dropout2_full)
                                
# instantiate optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)

print('Total number of variables: ' + str(input_size_full))
print('Total number of non-constant variables: ' + str(input_size_conv))
print('Hidden size conv: ' + str(hidden_size_conv))
print('Kernel size: ' + str(kernel_size))
print('Hidden size full: ' + str(hidden_size_full))
print('Dropout full: ' + str(dropout_full))
print('Hidden size 2 full: ' + str(hidden_size2_full))
print('Dropout 2 full: ' + str(dropout2_full))
print('Learning rate: ' + str(lr))
print('Weight decay: ' + str(weight_decay))
print()

train_CNN(train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, cnn, optimizer, mse_loss, num_epochs, batch_size)

train_r2 = r2(cnn, batch_size, train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, get_pred=False)
test_r2, test_pred_cnn = r2(cnn, batch_size, test_x_std_stack_nonConst, test_x_std_tuple, test_y_tuple, get_pred=True)

print()
print('Test R^2: ' + str(test_r2))

# put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
test = test.dropna(axis=0)
test['MonitorData_pred'] = pd.Series(test_pred_cnn, index=test.index)

# save test dataframe with predictions and final model
pickle.dump(cnn, open('cnn2_final_ridgeVImp.pkl', 'wb'))
test.to_csv('../data/test_cnn2Pred_ridgeVImp.csv', index=False)
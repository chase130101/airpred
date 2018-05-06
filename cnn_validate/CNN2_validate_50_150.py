import pandas as pd
import pandas as pd
import numpy as np
import torch
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

### delete sites from datasets where all monitor outputs are NaN
train_sites_all_nan_df = pd.DataFrame(np.isnan(train.groupby('site').sum()['MonitorData']))
train_sites_to_delete = list(train_sites_all_nan_df[train_sites_all_nan_df['MonitorData'] == True].index)
train = train[~train['site'].isin(train_sites_to_delete)]

val_sites_all_nan_df = pd.DataFrame(np.isnan(val.groupby('site').sum()['MonitorData']))
val_sites_to_delete = list(val_sites_all_nan_df[val_sites_all_nan_df['MonitorData'] == True].index)
val = val[~val['site'].isin(val_sites_to_delete)]

### split train, val, and test into x, y, and sites
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
val_x, val_y, val_sites = X_y_site_split(val, y_var_name='MonitorData', site_var_name='site')

### get dataframes with non-constant features only
nonConst_vars = get_nonConst_vars(train, site_var_name='site', y_var_name='MonitorData', cutoff=1000)
train_x_nonConst = train_x.loc[:, nonConst_vars]
val_x_nonConst = val_x.loc[:, nonConst_vars]

### standardize all features
standardizer_all = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_all = standardizer_all.fit_transform(train_x)
val_x_std_all = standardizer_all.transform(val_x)

### standardize non-constant features
standardizer_nonConst = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_nonConst = standardizer_nonConst.fit_transform(train_x_nonConst)
val_x_std_nonConst = standardizer_nonConst.transform(val_x_nonConst)




### get split sizes for TRAIN data (splitting by site)
train_split_sizes = split_sizes_site(train_sites.values)

### get tuples by site
train_x_std_tuple_nonConst = split_data(torch.from_numpy(train_x_std_nonConst).float(), train_split_sizes, dim = 0)
train_x_std_tuple = split_data(torch.from_numpy(train_x_std_all).float(), train_split_sizes, dim = 0)
train_y_tuple = split_data(torch.from_numpy(train_y.values), train_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
train_x_std_stack_nonConst = pad_stack_splits(train_x_std_tuple_nonConst, np.array(train_split_sizes), 'x')
train_x_std_stack_nonConst = Variable(torch.transpose(train_x_std_stack_nonConst, 1, 2))


### get split sizes for VALIDATION data (splitting by site)
val_split_sizes = split_sizes_site(val_sites.values)

### get tuples by site
val_x_std_tuple_nonConst = split_data(torch.from_numpy(val_x_std_nonConst).float(), val_split_sizes, dim = 0)
val_x_std_tuple = split_data(torch.from_numpy(val_x_std_all).float(), val_split_sizes, dim = 0)
val_y_tuple = split_data(torch.from_numpy(val_y.values), val_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
val_x_std_stack_nonConst = pad_stack_splits(val_x_std_tuple_nonConst, np.array(val_split_sizes), 'x')
val_x_std_stack_nonConst = Variable(torch.transpose(val_x_std_stack_nonConst, 1, 2))


num_epochs = 20
batch_size = 128
input_size_conv = train_x_std_nonConst.shape[1]
input_size_full = train_x_std_all.shape[1]
print('Total number of variables: ' + str(input_size_full))
print('Total number of non-constant variables: ' + str(input_size_conv))
print()

# CNN and optimizer hyper-parameters to test
hidden_size_conv_list = [25, 50]
kernel_size_list = [3, 5]
padding_list = [1, 2]
hidden_size_full_list = [150]
dropout_full_list = [0.1]
hidden_size2_full_list = [150]
dropout2_full_list = [0.1]
lr_list = [0.1]
weight_decay_list = [0.00001]

# Loss function
mse_loss = torch.nn.MSELoss(size_average=True)


best_val_r2 = -np.inf
for hidden_size_conv in hidden_size_conv_list:
    for kernel_size, padding in zip(kernel_size_list, padding_list):
        for hidden_size_full in hidden_size_full_list:
            for dropout_full in dropout_full_list:
                for hidden_size2_full in hidden_size2_full_list:
                    for dropout2_full in dropout2_full_list:
                        for lr in lr_list:
                            for weight_decay in weight_decay_list:
                                
                                # instantiate CNN
                                cnn = CNN2(input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, 
                                          dropout_full, hidden_size2_full, dropout2_full)
                                
                                # instantiate optimizer
                                optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)
                                
                                print('Hidden size conv: ' + str(hidden_size_conv))
                                print('Kernel size: ' + str(kernel_size))
                                print('Hidden size full: ' + str(hidden_size_full))
                                print('Dropout full: ' + str(dropout_full))
                                print('Hidden size 2 full: ' + str(hidden_size2_full))
                                print('Dropout 2 full: ' + str(dropout2_full))
                                print('Learning rate: ' + str(lr))
                                print('Weight decay: ' + str(weight_decay))

                                train_CNN(train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, cnn, optimizer, mse_loss, num_epochs, batch_size)
                                
                                val_r2 = r2(cnn, batch_size, val_x_std_stack_nonConst, val_x_std_tuple, val_y_tuple, get_pred=False)
                                print('Validation R^2: ' + str(val_r2))
                                print()
                                print()
                                
                                if val_r2 > best_val_r2:
                                    best_val_r2 = val_r2
                                    best_hidden_size_conv = hidden_size_conv
                                    best_kernel_size = kernel_size
                                    best_hidden_size_full = hidden_size_full
                                    best_dropout_full = dropout_full
                                    best_hidden_size2_full = hidden_size2_full
                                    best_dropout2_full = dropout2_full
                                    best_lr = lr
                                    best_weight_decay = weight_decay
                                    
print('Best validation R^2: ' + str(best_val_r2))
print('Best hidden size conv: ' + str(best_hidden_size_conv))
print('Best kernel size: ' + str(best_kernel_size))
print('Best hidden size full: ' + str(best_hidden_size_full))
print('Best dropout full: ' + str(best_dropout_full))
print('Best hidden size 2 full: ' + str(best_hidden_size2_full))
print('Best dropout 2 full: ' + str(best_dropout2_full))
print('Best learning rate: ' + str(best_lr))
print('Best weight decay: ' + str(best_weight_decay))                    
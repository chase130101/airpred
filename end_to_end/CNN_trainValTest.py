import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import sklearn.preprocessing
import sklearn.metrics
from data_split_tune_utils import X_y_site_split
from CNN_utils import split_sizes_site, split_data, pad_stack_splits, get_monitorData_indices, r2, get_nonConst_vars
from CNN_architecture import CNN 

np.random.seed(1)
torch.manual_seed(1)

### read in train, val, and test
train = pd.read_csv('../data/trainV_ridgeImp.csv')
val = pd.read_csv('../data/valV_ridgeImp.csv')
test = pd.read_csv('../data/testV_ridgeImp.csv')

### split train, val, and test into x, y, and sites
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
val_x, val_y, val_sites = X_y_site_split(val, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')

### get dataframes with non-constant features only
nonConst_vars = get_nonConst_vars(train, site_var_name='site', y_var_name='MonitorData', cutoff=1000)
train_x_nonConst = train_x.loc[:, nonConst_vars]
val_x_nonConst = val_x.loc[:, nonConst_vars]
test_x_nonConst = test_x.loc[:, nonConst_vars]

### standardize all features
standardizer_all = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_all = standardizer_all.fit_transform(train_x)
val_x_std_all = standardizer_all.transform(val_x)
test_x_std_all = standardizer_all.transform(test_x)

### standardize non-constant features
standardizer_nonConst = sklearn.preprocessing.StandardScaler(with_mean = True, with_std = True)
train_x_std_nonConst = standardizer_nonConst.fit_transform(train_x_nonConst)
val_x_std_nonConst = standardizer_nonConst.transform(val_x_nonConst)
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


### get split sizes for VALIDATION data (splitting by site)
val_split_sizes = split_sizes_site(val_sites.values)

### get tuples by site
val_x_std_tuple_nonConst = split_data(torch.from_numpy(val_x_std_nonConst).float(), val_split_sizes, dim = 0)
val_x_std_tuple = split_data(torch.from_numpy(val_x_std_all).float(), val_split_sizes, dim = 0)
val_y_tuple = split_data(torch.from_numpy(val_y.values), val_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
val_x_std_stack_nonConst = pad_stack_splits(val_x_std_tuple_nonConst, np.array(val_split_sizes), 'x')
val_x_std_stack_nonConst = Variable(torch.transpose(val_x_std_stack_nonConst, 1, 2))


### get split sizes for TEST data (splitting by site)
test_split_sizes = split_sizes_site(test_sites.values)

### get tuples by site
test_x_std_tuple_nonConst = split_data(torch.from_numpy(test_x_std_nonConst).float(), test_split_sizes, dim = 0)
test_x_std_tuple = split_data(torch.from_numpy(test_x_std_all).float(), test_split_sizes, dim = 0)
test_y_tuple = split_data(torch.from_numpy(test_y.values), test_split_sizes, dim = 0)

### get site sequences stacked into matrix to go through CNN
test_x_std_stack_nonConst = pad_stack_splits(test_x_std_tuple_nonConst, np.array(test_split_sizes), 'x')
test_x_std_stack_nonConst = Variable(torch.transpose(test_x_std_stack_nonConst, 1, 2))




# CNN parameters
input_size_conv = train_x_std_nonConst.shape[1]
hidden_size_conv = 20
kernel_size = 3
padding = 1
input_size_full = train_x_std_all.shape[1]
hidden_size_full = 30
hidden_size_combo = 30

# instantiate model
cnn = CNN(input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, hidden_size_combo)

# Loss function
mse_loss = torch.nn.MSELoss(size_average=True)

# Optimizer
lr = 0.0001
weight_decay = 0.000001
optimizer = torch.optim.SGD(cnn.parameters(), lr=lr, weight_decay=weight_decay)




num_epochs = 10000
batch_size = 500

# get number of batches
if train_x_std_stack_nonConst.size()[0] % batch_size != 0:
    num_batches = int(np.floor(train_x_std_stack_nonConst.size()[0]/batch_size) + 1)
else:
    num_batches = int(train_x_std_stack_nonConst.size()[0]/batch_size)
    
    
for epoch in range(num_epochs):
    epoch_loss = 0
    
    for batch in range(num_batches):
        # get x and y for this batch
        x_stack_batch_nonConst = train_x_std_stack_nonConst[batch_size * batch:batch_size * (batch+1)]
        x_tuple_batch = train_x_std_tuple[batch_size * batch:batch_size * (batch+1)]
        y_tuple_nans = train_y_tuple[batch_size * batch:batch_size * (batch+1)]
        
        # get indices for monitor data and actual monitor data
        y_by_site = []
        x_by_site = []
        y_ind_by_site = []
        for i in range(len(y_tuple_nans)):
            y_ind = get_monitorData_indices(y_tuple_nans[i])
            y_by_site.append(y_tuple_nans[i][y_ind])
            y_ind_by_site.append(y_ind)
            x_by_site.append(x_tuple_batch[i][y_ind])
        y_batch = Variable(torch.cat(y_by_site, dim=0)).float()
        x_batch = Variable(torch.cat(x_by_site, dim=0)).float()
        
        # get model output
        pred_batch = cnn(x_stack_batch_nonConst, x_batch, y_ind_by_site)
        
        # compute loss, backprop, and update parameters
        loss_batch = mse_loss(pred_batch, y_batch)
        loss_batch.backward()
        optimizer.step()
        
        # accumulate loss over epoch
        epoch_loss += loss_batch.data[0]
        
    print('Validation R^2 after epoch ' + str(epoch) + ': ' + str(r2(cnn, batch_size, val_x_std_stack_nonConst, val_x_std_tuple, val_y_tuple)))
    print('Epoch loss after epoch ' + str(epoch) + ': ' + str(epoch_loss))
    

print('Test R^2: ' + str(r2(cnn, batch_size, test_x_std_stack_nonConst, test_x_std_tuple, test_y_tuple)))
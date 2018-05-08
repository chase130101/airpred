"""Description: This script allows the user to train either one of our CNN architectures (see CNN_architecture.py)
on the full, imputed train set and evaluate the fitted model on the imputed test set using R^2. The predictions on the 
test set are saved in a csv as a column in the test data, excluding rows where there is no monitor data output.
"""
import argparse
import configparser
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
# these imported classes are the CNN architectures (see CNN_architecture.py)
from CNN_architecture import CNN1, CNN2


config = configparser.RawConfigParser()
config.read("config/py_config.ini")

parser = argparse.ArgumentParser()

parser.add_argument("cnn_type", 
    help = "Specify which of the two CNN architectures to use. The options are \"cnn_1\" and \"cnn_2\".",
    type=int,
    choices = ["cnn_1", "cnn_2"]
    default=4)

parser.add_argument("dataset",
    help = "Specify which imputed dataset to use. " + \
    "Options are ridge-imputed (\"ridgeImp\") and random-forest imputed (\"rfImp\").",
    choices=["ridgeImp", "rfImp"]) 


args = parser.parse_args()

# set seeds for reproducibility
np.random.seed(1)
torch.manual_seed(1)

# read in train, val, and test
train, val, test = None, None, None


if args.dataset == "ridgeImp":
    train = pd.read_csv(config["Ridge_Imputation"]["trainV"])
    val   = pd.read_csv(config["Ridge_Imputation"]["valV"])
    test  = pd.read_csv(config["Ridge_Imputation"]["testV"])


elif args.dataset == "rfImp":
    train = pd.read_csv(config["RF_Imputation"]["trainV"])
    val   = pd.read_csv(config["RF_Imputation"]["valV"])
    test  = pd.read_csv(config["RF_Imputation"]["testV"])


# combine train and validation sets into train set
train = pd.concat([train, val], axis=0, ignore_index=True)

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




# get split sizes for TRAIN data (splitting by site)
train_split_sizes = split_sizes_site(train_sites.values)

# get tuples by site
train_x_std_tuple_nonConst = split_data(torch.from_numpy(train_x_std_nonConst).float(), train_split_sizes, dim = 0)
train_x_std_tuple = split_data(torch.from_numpy(train_x_std_all).float(), train_split_sizes, dim = 0)
train_y_tuple = split_data(torch.from_numpy(train_y.values), train_split_sizes, dim = 0)

# get site sequences stacked into matrix to go through CNN
train_x_std_stack_nonConst = pad_stack_splits(train_x_std_tuple_nonConst, np.array(train_split_sizes), 'x')
train_x_std_stack_nonConst = Variable(torch.transpose(train_x_std_stack_nonConst, 1, 2))


# get split sizes for TEST data (splitting by site)
test_split_sizes = split_sizes_site(test_sites.values)

# get tuples by site
test_x_std_tuple_nonConst = split_data(torch.from_numpy(test_x_std_nonConst).float(), test_split_sizes, dim = 0)
test_x_std_tuple = split_data(torch.from_numpy(test_x_std_all).float(), test_split_sizes, dim = 0)
test_y_tuple = split_data(torch.from_numpy(test_y.values), test_split_sizes, dim = 0)

# get site sequences stacked into matrix to go through CNN
test_x_std_stack_nonConst = pad_stack_splits(test_x_std_tuple_nonConst, np.array(test_split_sizes), 'x')
test_x_std_stack_nonConst = Variable(torch.transpose(test_x_std_stack_nonConst, 1, 2))


# training parameters and model input sizes
num_epochs = 100
batch_size = 128
input_size_conv = train_x_std_nonConst.shape[1]
input_size_full = train_x_std_all.shape[1]

# train/test CNN1
if args.cnn_type == "cnn_1":
    hidden_size_conv  = config["CNN_hyperparam_1"]["hidden_size_conv"]
    kernel_size       = config["CNN_hyperparam_1"]["kernel_size"]
    padding           = config["CNN_hyperparam_1"]["padding"]
    hidden_size_full  = config["CNN_hyperparam_1"]["hidden_size_full"]
    dropout_full      = config["CNN_hyperparam_1"]["dropout_full"]
    hidden_size_combo = config["CNN_hyperparam_1"]["hidden_size_combo"]
    dropout_combo     = config["CNN_hyperparam_1"]["dropout_combo"]
    lr                = config["CNN_hyperparam_1"]["lr"]
    weight_decay      = config["CNN_hyperparam_1"]["weight_decay"]

    # Loss function
    mse_loss = torch.nn.MSELoss(size_average=True)

    # instantiate CNN
    cnn = CNN1(input_size_conv, hidden_size_conv, kernel_size, padding, input_size_full, hidden_size_full, 
               dropout_full, hidden_size_combo, dropout_combo)

    # instantiate optimizer
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=weight_decay)

    print('Total number of variables: ' + str(input_size_full))
    print('Total number of non-constant variables: ' + str(input_size_conv))
    print('Hidden size conv: ' + str(hidden_size_conv))
    print('Kernel size: ' + str(kernel_size))
    print('Hidden size full: ' + str(hidden_size_full))
    print('Dropout full: ' + str(dropout_full))
    print('Hidden size combo: ' + str(hidden_size_combo))
    print('Dropout combo: ' + str(dropout_combo))
    print('Learning rate: ' + str(lr))
    print('Weight decay: ' + str(weight_decay))
    print()

    # train
    train_CNN(train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, cnn, optimizer, mse_loss, num_epochs, batch_size)
    
    # evaluate
    test_r2, test_pred_cnn = r2(cnn, batch_size, test_x_std_stack_nonConst, test_x_std_tuple, test_y_tuple, get_pred=True)

    print()
    print('Test R^2: ' + str(test_r2))

    # put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
    test = test.dropna(axis=0)
    test['MonitorData_pred'] = pd.Series(test_pred_cnn, index=test.index)

    # save test dataframe with predictions and final model
    pickle.dump(cnn, open(config["Regression"]["cnn_1_model"], 'wb'))
    test.to_csv(config["Regression"]["cnn_1_pred"], index=False)


# train/test CNN2
else:
    hidden_size_conv  = config["CNN_hyperparam_2"]["hidden_size_conv"]
    kernel_size       = config["CNN_hyperparam_2"]["kernel_size"]
    padding           = config["CNN_hyperparam_2"]["padding"]
    hidden_size_full  = config["CNN_hyperparam_2"]["hidden_size_full"]
    dropout_full      = config["CNN_hyperparam_2"]["dropout_full"]
    hidden_size2_full = config["CNN_hyperparam_2"]["hidden_size2_full"]
    dropout2_full     = config["CNN_hyperparam_2"]["dropout2_full"]
    lr                = config["CNN_hyperparam_2"]["lr"]
    weight_decay      = config["CNN_hyperparam_2"]["weight_decay"]

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

    # train
    train_CNN(train_x_std_stack_nonConst, train_x_std_tuple, train_y_tuple, cnn, optimizer, mse_loss, num_epochs, batch_size)
    
    # evaluate
    test_r2, test_pred_cnn = r2(cnn, batch_size, test_x_std_stack_nonConst, test_x_std_tuple, test_y_tuple, get_pred=True)

    print()
    print('Test R^2: ' + str(test_r2))

    # put model predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
    test = test.dropna(axis=0)
    test['MonitorData_pred'] = pd.Series(test_pred_cnn, index=test.index)

    # save test dataframe with predictions and final model
    pickle.dump(cnn, open(config["Regression"]["cnn_2_model"], 'wb'))
    test.to_csv(config["Regression"]["cnn_2_pred"], index=False)
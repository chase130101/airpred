import numpy as np
import pandas as pd

def train_val_test_split(data, train_prop, test_prop, site_var_name='site'):
    """Splits data into train, validation, test sets by PM2.5 monitor site
    
    Arguments:
        data (pandas dataframe): Data to be split
        site_var_name (string): Site ID variable name
        train_prop (float): Proportion of sites to be put into train set
        test_prop (float): Proportion of non-train sites to be put into test set
    """
    # get sites for val/test data
    val_test_sites = np.random.choice(np.unique(data[site_var_name].values), round(len(np.unique(data[site_var_name].values))*(1-train_prop)), replace = False)
    
    # get sites for test data
    test_prop = test_prop/(1-train_prop)
    test_sites = np.random.choice(np.unique(val_test_sites), round(len(np.unique(val_test_sites))*test_prop), replace = False)
    
    # get train, val, and test
    train = data[~data[site_var_name].isin(val_test_sites)]
    val = data[(data[site_var_name].isin(val_test_sites)) & (~data[site_var_name].isin(test_sites))]
    test = data[data[site_var_name].isin(test_sites)]
    
    return train, val, test


def train_test_split(data, train_prop, site_var_name='site'):
    """Splits data into train test sets by PM2.5 monitor site
    
    Arguments:
        data (pandas dataframe): Data to be split
        site_var_name (string): Site ID variable name
        train_prop (float): Proportion of sites to be put into train set
    """
    # get sites for train data
    train_sites = np.random.choice(np.unique(data[site_var_name].values), round(len(np.unique(data[site_var_name].values))*train_prop), replace = False)
        
    # get train and test
    train = data[data[site_var_name].isin(train_sites)]
    test = data[~data[site_var_name].isin(train_sites)]
    
    return train, test


def X_y_site_split(data, y_var_name='MonitorData', site_var_name='site'):
    """Splits a dataframe into X, y, site ID
    
    Arguments:
        data (pandas dataframe): Data to be split
        y_var_name (string): Response variable name
        site_var_name (string): Site ID variable name
    """
    data_y = data.loc[:, y_var_name]
    data_sites = data.loc[:, site_var_name]
    data_x = data.drop([y_var_name, site_var_name], axis=1)
    
    return data_x, data_y, data_sites


def cross_validation_splits(data, num_folds, site_var_name='site'):
    """Returns indices for cross-validation train, test splits for use in GridSearchCV
    
    Arguments:
        data (pandas dataframe): Data to be split
        site_var_name (string): Site ID variable name
        num_folds (int): Number of cross_validation folds
    """
    # get site ids for each fold
    try:
        site_ids_by_fold = np.random.choice(np.unique(data[site_var_name].values), (num_folds, round(len(np.unique(data[site_var_name].values))/num_folds)), replace = False)
    except ValueError:
        site_ids_by_fold = np.random.choice(np.unique(data[site_var_name].values), (num_folds, round(len(np.unique(data[site_var_name].values))/(num_folds+1))), replace = False)
    site_ids_by_fold = [list(site_ids_by_fold[i]) for i in range(num_folds)]
    leftover_sites = list(np.unique(data[~data[site_var_name].isin(np.unique(site_ids_by_fold))][site_var_name]))
    site_ids_by_fold[0] += leftover_sites
    
    # get indices for each fold
    ind_by_fold = []
    for i in range(num_folds):
        ind_fold_i = list(data[data[site_var_name].isin(site_ids_by_fold[i])].index)
        ind_by_fold.append(ind_fold_i)
    
    # produce cross-validation train test splits based on indices
    train_test_splits = []
    for i in range(num_folds):
        test_i = np.array(ind_by_fold[i])
        train_i = np.array(np.concatenate(ind_by_fold[:i] + ind_by_fold[i+1:]).astype(int))
        train_test_splits.append((train_i, test_i))
                                 
    return np.array(train_test_splits)
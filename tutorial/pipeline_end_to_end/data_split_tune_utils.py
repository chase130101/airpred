"""Description: This is set of functions to be used for data pre-processing and model validation.
In particular, these functions are useful for datasets that are arranged into sequences, such as
the air pollution data. We do not, for example, want part of a site sequence to be in the train data
and the rest of the site sequence to be in the test data. The functions help the user avoid pitfalls 
like this. The other scripts in the repository and our tutorial will give the user more 
insight into how all of these functions can be used.

The cross-validation function was written because of the extreme memory usage of scikit-learn's 
GridSearchCV that is used for tuning models - our function requires much less memory and thus can
be used for large datasets such as the air pollution dataset.
"""
import numpy as np
import pandas as pd
import itertools
import sklearn.metrics

def train_val_test_split(data, train_prop, test_prop, site_var_name='site'):
    """Splits dataset into train, validation, and test sets
    - Site-days with same monitor site ID don't get split up
    -----------
    Inputs:
        - data (pandas.DataFrame): Dataset to be split
        - site_var_name (str): Site ID variable name
        - train_prop (float): Proportion of sites to be put into train set; must be between 0 and 1
        - test_prop (float): Proportion of sites to be put into test set; must be between 0 and 1
        -- Sum of train_prop and test_prop must be between 0 and 1
    -----------    
    Outputs:
        - train (pandas.DataFrame): train set
        - val (pandas.DataFrame): validation set
        - test (pandas.DataFrame): test set
    """
    # get sites for val/test data
    val_test_sites = np.random.choice(np.unique(data[site_var_name].values), 
                                      round(len(np.unique(data[site_var_name].values))*(1-train_prop)), 
                                      replace=False)
    
    # get sites for test data
    test_prop = test_prop/(1-train_prop)
    test_sites = np.random.choice(np.unique(val_test_sites), 
                                  round(len(np.unique(val_test_sites))*test_prop), 
                                  replace=False)
    
    # get train, val, and test
    train = data[~data[site_var_name].isin(val_test_sites)]
    val = data[(data[site_var_name].isin(val_test_sites)) & (~data[site_var_name].isin(test_sites))]
    test = data[data[site_var_name].isin(test_sites)]
    
    return train, val, test


def train_test_split(data, train_prop, site_var_name='site'):
    """Splits dataset into train and test sets
    - Site-days with same monitor site ID don't get split up
    -----------
    Inputs:
        - data (pandas.DataFrame): Dataset to be split
        - site_var_name (str): Site ID variable name
        - train_prop (float): Proportion of sites to be put into train set; must be between 0 and 1
    -----------
    Outputs:
        - train (pandas.DataFrame): train set
        - test (pandas.DataFrame): test set
    """
    # get sites for train data
    train_sites = np.random.choice(np.unique(data[site_var_name].values), 
                                   round(len(np.unique(data[site_var_name].values))*train_prop), 
                                   replace=False)
        
    # get train and test
    train = data[data[site_var_name].isin(train_sites)]
    test = data[~data[site_var_name].isin(train_sites)]
    
    return train, test


def X_y_site_split(data, y_var_name='MonitorData', site_var_name='site'):
    """Splits a dataset into features, response values, and monitor site IDs
    -----------
    Inputs:
        - data (pandas.DataFrame): Dataset to be split
        - y_var_name (str): Response variable name
        - site_var_name (str): Site ID variable name
    -----------
    Outputs:
        - X (pandas.DataFrame): Feature columns
        - y (pandas.Series): Response variable column
        - sites (pandas.Series): Site variable column
    """
    y = data.loc[:, y_var_name]
    sites = data.loc[:, site_var_name]
    X = data.drop([y_var_name, site_var_name], axis=1)
    
    return X, y, sites


def cross_validation_splits(data, num_folds, site_var_name='site'):
    """Returns indices for cross-validation train, test splits
    - Site-days with same monitor site ID don't get split up
    - This function is used within the cross_validation function below
    -----------
    Inputs:
        - data (pandas.DataFrame): Dataset to be split
        - site_var_name (str): Site ID variable name
        - num_folds (int): Number of cross-validation folds; must be less than the number 
        of unique sites
    -----------
    Outputs:
        - (numpy.array): Set of num_folds tuples; each tuple has its first value a numpy.array 
        of train indices and as its second value a numpy.array of test indices
    """
    # get site ids for each fold
    try:
        site_ids_by_fold = np.random.choice(np.unique(data[site_var_name].values), 
                                            (num_folds, round(len(np.unique(data[site_var_name].values))/num_folds)), 
                                            replace=False)
    except ValueError:
        site_ids_by_fold = np.random.choice(np.unique(data[site_var_name].values), 
                                            (num_folds, round(len(np.unique(data[site_var_name].values))/(num_folds+1))), 
                                            replace=False)
        
    site_ids_by_fold = [list(site_ids_by_fold[i]) for i in range(num_folds)]
    leftover_sites = list(np.unique(data[~data[site_var_name].isin(np.unique(site_ids_by_fold))][site_var_name]))
    site_ids_by_fold[0] += leftover_sites
    
    # get indices for each fold
    ind_by_fold = []
    for i in range(num_folds):
        ind_fold_i = list(data[data[site_var_name].isin(site_ids_by_fold[i])].index)
        ind_by_fold.append(ind_fold_i)
    
    # produce cross-validation train, test splits based on indices
    train_test_splits = []
    for i in range(num_folds):
        test_i = np.array(ind_by_fold[i])
        train_i = np.array(np.concatenate(ind_by_fold[:i] + ind_by_fold[i+1:]).astype(int))
        train_test_splits.append((train_i, test_i))
                                 
    return np.array(train_test_splits)


def cross_validation(data, model, hyperparam_dict, num_folds, y_var_name='MonitorData', site_var_name='site'):
    """Returns best cross-validation R^2 and dictionary of best model hyper-parameters
    - Site-days with same site ID don't get split up into different folds (this function uses
    the cross_validation_splits function)
    ----------- 
    Inputs:
        - data (pandas.DataFrame): To use for folds in cross-validation
        - model (sklearn model): Model to tune
        - hyperparam_dict (dict): Model hyper-parameters to grid search; keys should be 
        strings that are names of attributes of the sklearn model being tuned and values
        should be valid as model attributes
        - num_folds (int): Number of cross-validation folds; must be less than the number 
        of unique sites
        - y_var_name (str): Response variable name
        - site_var_name (str): Site ID variable name
    -----------
    Outputs:
        - best_mean_r2 (float): Best cross-validation R^2
        - best_combo_dict (dict): Best model hyper-parameters; keys are strings that are names 
        of attributes of the tuned sklearn model and values are valid as model attributes
    """
    # indices for cv train, test splits
    cv_splits = cross_validation_splits(data, num_folds, site_var_name=site_var_name)

    # split dataset into x, y, and site
    x, y, sites = X_y_site_split(data, y_var_name=y_var_name, site_var_name=site_var_name)
    
    # get all hyper-parameter combinations based on input dictionary
    hyperparam_lists = [hyperparam_dict[list(hyperparam_dict.keys())[i]] for i in range(len(hyperparam_dict.keys()))]
    hyperparam_combos = list(itertools.product(*hyperparam_lists))
    
    mean_r2_list = []
    for combo in hyperparam_combos:
        test_r2_list = []
        for train_ind, test_ind in cv_splits:
            
            train_x = x.loc[train_ind, :]
            train_y = y.loc[train_ind]

            test_x = x.loc[test_ind, :]
            test_y = y.loc[test_ind]
            
            # set model attributes based on current combination of hyper-parameters being tested
            for i, key in zip(np.arange(len(combo)), list(hyperparam_dict.keys())):
                setattr(model, key, combo[i])

            model.fit(train_x, train_y)
            model_test_pred = model.predict(test_x)
            model_test_r2 = sklearn.metrics.r2_score(test_y, model_test_pred)
            
            test_r2_list.append(model_test_r2)
            
        mean_r2 = np.mean(test_r2_list)
        mean_r2_list.append(mean_r2)
        
        # keep track of best hyper-parameter combination and best mean R^2
        if combo == hyperparam_combos[0]:
            best_mean_r2 = np.mean(test_r2_list)
            best_combo = combo
        elif mean_r2 > best_mean_r2:
            best_mean_r2 = np.mean(test_r2_list)
            best_combo = combo
    
    # put best hyper-parameter combination into dictionary
    best_combo_dict = {}
    for i, key in zip(np.arange(len(best_combo)), list(hyperparam_dict.keys())):
        best_combo_dict[key] = best_combo[i]
        
    return best_mean_r2, best_combo_dict
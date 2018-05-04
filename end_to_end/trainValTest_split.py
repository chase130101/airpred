"""Description: This script splits a dataset into train/validation/test sets
and saves the new, separated datasets as csv files. The data splits will be 
such that site-days for the same site do not get split up.
"""
import numpy as np
import pandas as pd
# this is an imported functions created for this package that split datasets (see data_split_tune_utils.py)
from data_split_tune_utils import train_val_test_split

# set seed for reproducibility
np.random.seed(1)

# split data into train, validation, and test sets and save
data = pd.read_csv('../data/data_to_impute.csv')
train, val, test = train_val_test_split(data, train_prop = 0.7, test_prop = 0.15, site_var_name = 'site')
train.to_csv('../data/trainV.csv', index = False)
val.to_csv('../data/valV.csv', index = False)
test.to_csv('../data/testV.csv', index = False)
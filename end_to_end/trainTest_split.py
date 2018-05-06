"""Description: This script splits a dataset into either train/test sets
and saves the new, separated datasets as csv files. The data splits will 
be such that site-days for the same site do not get split up.
"""
import numpy as np
import pandas as pd
# this is an imported functions created for this package that split datasets (see data_split_tune_utils.py)
from data_split_tune_utils import train_test_split

# set seed for reproducibility
np.random.seed(1)

# split data into train and test sets and save
data = pd.read_csv('../data/data_to_impute.csv')
train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')
train.to_csv('../data/train.csv', index = False)
test.to_csv('../data/test.csv', index = False)
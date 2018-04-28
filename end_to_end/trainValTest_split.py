import numpy as np
import pandas as pd
from data_split_tune_utils import train_val_test_split

np.random.seed(1)
data = pd.read_csv('../data/data_to_impute.csv')
train, val, test = train_val_test_split(data, train_prop = 0.7, test_prop = 0.15, site_var_name = 'site')
train.to_csv('../data/trainV.csv', index = False)
val.to_csv('../data/valV.csv', index = False)
test.to_csv('../data/testV.csv', index = False)
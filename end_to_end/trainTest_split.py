import numpy as np
import pandas as pd
from data_split_tune_utils import train_test_split

np.random.seed(1)
data = pd.read_csv('../data/data_to_impute.csv')
train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')
train.to_csv('../data/train.csv', index = False)
test.to_csv('../data/test.csv', index = False)
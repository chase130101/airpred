import pandas as pd
import numpy as np
import pickle
from data_split_tune_utils import train_test_split, X_y_site_split
from predictiveImputer_mod import PredictiveImputer

np.random.seed(1)

train = pd.read_csv('../data/train.csv')

# split train up; only fit imputer on part of train set due to memory/time
train1, train2 = train_test_split(train, train_prop=0.3, site_var_name='site')

# split part of train set to fit imputer on into x, y, and site
train1_x, train1_y, train1_sites = X_y_site_split(train1, y_var_name='MonitorData', site_var_name='site')

# create imputer and fit on part of train set
ridge_imputer = PredictiveImputer(max_iter=10, initial_strategy='mean', f_model='Ridge')
ridge_imputer.fit(train1_x, alpha=0.0001, fit_intercept=True, normalize=True, random_state=1)

# save fitted imputer
pickle.dump(ridge_imputer, open('ridge_imputer.pkl', 'wb'))
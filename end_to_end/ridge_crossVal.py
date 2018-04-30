import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.metrics
import pickle
from data_split_tune_utils import cross_validation_splits, X_y_site_split, cross_validation

np.random.seed(1)

train = pd.read_csv('../data/train_ridgeImp.csv')
#train = pd.read_csv('../data/train_rfImp.csv')
train = train.dropna(axis=0)

ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)
parameter_grid_ridge = {'alpha' : [0.1, 0.01, 0.001, 0.0001, 0.00001]}
cv_r2, best_hyperparams = cross_validation(data=train, model=ridge, hyperparam_dict=parameter_grid_ridge, num_folds=4, y_var_name='MonitorData', site_var_name='site')
print('Cross-validation R^2: ' + str(cv_r2))
print('Best hyper-parameters: ' + str(best_hyperparams))
pickle.dump(best_hyperparams, open('best_ridge_hyperparams.pkl', 'wb'))
import pandas as pd
import numpy as np
import xgboost as xgb
import sklearn.metrics
import pickle
from data_split_tune_utils import cross_validation_splits, X_y_site_split, cross_validation

np.random.seed(1)

train = pd.read_csv('../data/train_ridgeImp.csv')
# train = pd.read_csv('../data/train_rfImp.csv')
train = train.dropna(axis=0)


xgboost = xgb.XGBRegressor(random_state=1, n_jobs=-1)

#Information on gradient boosting parameter tuning
#https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
parameter_grid_xgboost = {'learning_rate': [0.001, 0.01, 0.05, 0.1], 'max_depth': [4, 6, 8, 10], 'n_estimators': [100, 250, 500, 750, 1000]}

cv_r2, best_hyperparams = cross_validation(data=train, model=xgboost, hyperparam_dict=parameter_grid_xgboost, num_folds=4, y_var_name='MonitorData', site_var_name='site')
print('Cross-validation R^2: ' + str(cv_r2))
print('Best hyper-parameters: ' + str(best_hyperparams))
pickle.dump(best_hyperparams, open('best_xgboost_hyperparams.pkl', 'wb'))

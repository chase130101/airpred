import argparse
import configparser
import numpy as np
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.ensemble
import sklearn.metrics
import sys
import xgboost as xgb

from data_split_tune_utils import cross_validation_splits, X_y_site_split, cross_validation

models = ["ridge", "rf", "xgb"]
datasets = ["ridgeImp", "rfImp"]


config = configparser.RawConfigParser()
config.read('config/py_config.ini')


parser = argparse.ArgumentParser()
parser.add_argument("model", 
    help = "Specify which model to evaluate via cross-validation. " +\
    "Options are Ridge (\"ridge\"), Random Forest (\"rf\"), and XGBoost (\"xgb\").")


parser.add_argument("dataset",
    help = "Specify which dataset to use. " + \
    "Options are ridge-imputed (\"ridgeImp\") and random-forest imputed (\"rfImp\").") 

args = parser.parse_args()

if args.model not in models:
    print("Invalid regression model!")
    sys.exit()


if args.dataset not in datasets:
    print("Invalid dataset!")
    sys.exit()


train = None

if args.dataset == "ridgeImp":
    train = pd.read_csv(config["Ridge_Imputation"]["train"])


elif args.dataset == "rfImp":
    train = pd.read_csv(config["RF_Imputation"]["train"])


if train == None: # failsafe
    print("Invalid dataset!")
    sys.exit()

train = train.dropna(axis=0)

print("Training model \"{}\" on dataset \"{}\"...)".format(args.model, args.dataset))

if args.model == "ridge":
    ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)
    parameter_grid_ridge = {'alpha' : [0.1, 0.01, 0.001, 0.0001, 0.00001]}
    cv_r2, best_hyperparams = cross_validation(data=train, model=ridge, hyperparam_dict=parameter_grid_ridge, num_folds=4, y_var_name='MonitorData', site_var_name='site')
    print('Cross-validation R^2: ' + str(cv_r2))
    print('Best hyper-parameters: ' + str(best_hyperparams))
    pickle.dump(best_hyperparams, open(config["Reg_Best_Hyperparams"]["ridge"], 'wb'))

elif args.model == "rf":
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
    parameter_grid_rf = {'max_features' : [10, 15, 20, 25]}
    cv_r2, best_hyperparams = cross_validation(data=train, model=rf, hyperparam_dict=parameter_grid_rf, num_folds=4, y_var_name='MonitorData', site_var_name='site')
    print('Cross-validation R^2: ' + str(cv_r2))
    print('Best hyper-parameters: ' + str(best_hyperparams))
    pickle.dump(best_hyperparams, open(config["Reg_Best_Hyperparams"]["rf"], 'wb'))

elif args.model == "xgb":
    xgboost = xgb.XGBRegressor(random_state=1, n_jobs=-1)

    #Information on gradient boosting parameter tuning
    #https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
    parameter_grid_xgboost = {'learning_rate': [0.001, 0.01, 0.05, 0.1], 'max_depth': [4, 6, 8, 10], 'n_estimators': [100, 250, 500, 750, 1000]}

    cv_r2, best_hyperparams = cross_validation(data=train, model=xgboost, hyperparam_dict=parameter_grid_xgboost, num_folds=4, y_var_name='MonitorData', site_var_name='site')
    print('Cross-validation R^2: ' + str(cv_r2))
    print('Best hyper-parameters: ' + str(best_hyperparams))
    pickle.dump(best_hyperparams, open(config["Reg_Best_Hyperparams"]["xgb"], 'wb'))


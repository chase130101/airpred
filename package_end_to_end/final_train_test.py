"""Description: This script allows the user to train either a ridge regression, random forest,
or XGBoost model on the full, imputed train set using the best hyper-parameters from cross-validation 
and evaluate the fitted model on the imputed test set using R^2. The predictions on the test set are saved 
in a csv as a column in the test data, excluding rows where there is no monitor data output. The feature
importances from random forest or XGBoost are saved when those methods are used.
"""
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
# this is an imported functions created for this package that splits datasets (see data_split_tune_utils.py)
from data_split_tune_utils import X_y_site_split

models = ["ridge", "rf", "xgb"]
datasets = ["ridgeImp", "rfImp"]


config = configparser.RawConfigParser()
config.read("config/py_config.ini")


parser = argparse.ArgumentParser()
parser.add_argument("model", 
    help = "Specify which model to evaluate a train/test split on. " +\
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
test  = None


if args.dataset == "ridgeImp":
    train = pd.read_csv(config["Ridge_Imputation"]["train"])
    test  = pd.read_csv(config["Ridge_Imputation"]["test"])


elif args.dataset == "rfImp":
    train = pd.read_csv(config["RF_Imputation"]["train"])
    test  = pd.read_csv(config["RF_Imputation"]["test"])


if train == None or test == None: # failsafe
    print("Invalid dataset!")
    sys.exit()


# drop rows with no monitor data response value
train = train.dropna(axis=0)
test = test.dropna(axis=0)

# split train and test datasets into x, y, and site
train_x, train_y, train_sites = X_y_site_split(train, y_var_name="MonitorData", site_var_name="site")
test_x, test_y, test_sites = X_y_site_split(test, y_var_name="MonitorData", site_var_name="site")


print("Training model \"{}\" on dataset \"{}\"...)".format(args.model, args.dataset))

if args.model == "ridge":
    # load dictionary with best ridge hyper-parameters from cross-validation
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["ridge"], "rb"))
    
    # instantiate ridge
    ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)
    
    # set ridge attributes to best combination of hyper-parameters from cross-validation
    for key in list(best_hyperparams.keys()):
        setattr(ridge, key, best_hyperparams[key])
    
    # fit ridge on train data and make predictions on test data; compute test R^2
    ridge.fit(train_x, train_y)
    test_pred_ridge = ridge.predict(test_x)
    test_r2_ridge = sklearn.metrics.r2_score(test_y, test_pred_ridge)
    print("Test R^2: " + str(test_r2_ridge))

    # put ridge predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
    # save ridge predictions and fitted ridge model
    test["MonitorData_pred"] = pd.Series(test_pred_ridge, index=test.index)
    test.to_csv(config["Regression"]["ridge_pred"], index=False)
    pickle.dump(ridge, open(config["Regression"]["ridge_final"], "wb"))


elif args.model == "rf":
    # load dictionary with best random forest hyper-parameters from cross-validation
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["rf"], "rb"))
    
    # instantiate random forest
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500, random_state=1, n_jobs=-1)
    
    # set random forest attributes to best combination of hyper-parameters from cross-validation
    for key in list(best_hyperparams.keys()):
        setattr(rf, key, best_hyperparams[key])
    
    # fit random forest on train data and make predictions on test data; compute test R^2
    rf.fit(train_x, train_y)
    test_pred_rf = rf.predict(test_x)
    test_r2_rf = sklearn.metrics.r2_score(test_y, test_pred_rf)
    print("Test R^2: " + str(test_r2_rf))

    # put random forest predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
    # save random forest predictions; don't save fitted random forest model due to memory
    test["MonitorData_pred"] = pd.Series(test_pred_rf, index=test.index)
    test.to_csv("../data/test_rfPred.csv", index=False)
    #pickle.dump(rf, open("rf_final.pkl", "wb"))

    # create dataframe of random forest feature importances and save
    feature_importance_df = pd.DataFrame(rf.feature_importances_.reshape(len(rf.feature_importances_), -1), columns=["RF_Feature_Importance"])
    feature_importance_df["Variable"] = pd.Series(train_x.columns, index=feature_importance_df.index)
    feature_importance_df.to_csv(config["Regression"]["rf_ftImp"], index=False)                                      

elif args.model == "xgb":
    # load dictionary with best xgboost hyper-parameters from cross-validation
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["xgb"], "rb"))

    # instantiate xgboost
    xgboost = xgb.XGBRegressor(random_state=1, n_jobs=-1)
    
    # set xgboost attributes to best combination of hyper-parameters from cross-validation
    for key in list(best_hyperparams.keys()):
        setattr(xgboost, key, best_hyperparams[key])
    
    # fit xgboost on train data and make predictions on test data; compute test R^2
    xgboost.fit(train_x, train_y)
    test_pred_xgboost = xgboost.predict(test_x)
    test_r2_xgboost = sklearn.metrics.r2_score(test_y, test_pred_xgboost)
    print("Test R^2: " + str(test_r2_xgboost))
    
    # put xgboost predictions into test dataframe (note that these predictions do not include those for rows where there is no response value)
    # save xgboost predictions; don't save fitted xgboost model due to memory
    test["MonitorData_pred"] = pd.Series(test_pred_xgboost, index=test.index)
    test.to_csv(config["Regression"]["xgb_pred"], index=False)
    #pickle.dump(xgboost, open("xgboost_final.pkl", "wb"))

    # create dataframe of xgboost feature importances and save
    feature_importance_df = pd.DataFrame(xgboost.feature_importances_.reshape(len(xgboost.feature_importances_), -1), columns=["XGBoost_Feature_Importance"])
    feature_importance_df["Variable"] = pd.Series(train_x.columns, index=feature_importance_df.index)
    feature_importance_df.to_csv(config["Regression"]["xgb_ftImp"], index=False)
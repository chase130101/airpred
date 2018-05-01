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


train = train.dropna(axis=0)
test = test.dropna(axis=0)
train_x, train_y, train_sites = X_y_site_split(train, y_var_name='MonitorData', site_var_name='site')
test_x, test_y, test_sites = X_y_site_split(test, y_var_name='MonitorData', site_var_name='site')


print("Training model \"{}\" on dataset \"{}\"...)".format(args.model, args.dataset))

if args.model == "ridge":
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["ridge"], 'rb'))
    ridge = sklearn.linear_model.Ridge(random_state=1, normalize=True, fit_intercept=True)
    for key in list(best_hyperparams.keys()):
        setattr(ridge, key, best_hyperparams[key])
        
    ridge.fit(train_x, train_y)
    test_pred_ridge = ridge.predict(test_x)
    test_r2_ridge = sklearn.metrics.r2_score(test_y, test_pred_ridge)
    print('Test R^2: ' + str(test_r2_ridge))

    test['MonitorData_pred'] = pd.Series(test_pred_ridge, index=test.index)
    test.to_csv(config["Regression"]["ridge_pred"], index=False)
    pickle.dump(ridge, open(config["Regression"]["ridge_final"], 'wb'))


elif args.model == "rf":
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["rf"], 'rb'))
    rf = sklearn.ensemble.RandomForestRegressor(n_estimators=500, random_state=1, n_jobs=-1)
    for key in list(best_hyperparams.keys()):
        setattr(rf, key, best_hyperparams[key])
        
    rf.fit(train_x, train_y)
    test_pred_rf = rf.predict(test_x)
    test_r2_rf = sklearn.metrics.r2_score(test_y, test_pred_rf)
    print('Test R^2: ' + str(test_r2_rf))

    test['MonitorData_pred'] = pd.Series(test_pred_rf, index=test.index)
    test.to_csv('../data/test_rfPred.csv', index=False)
    #pickle.dump(rf, open('rf_final.pkl', 'wb'))

    feature_importance_df = pd.DataFrame(rf.feature_importances_.reshape(len(rf.feature_importances_), -1), columns=['RF_Feature_Importance'])
    feature_importance_df['Variable'] = pd.Series(train_x.columns, index=feature_importance_df.index)
    feature_importance_df.to_csv(config["Regression"]['rf_ft'], index=False)                                      


elif args.model == "xgb":
    best_hyperparams = pickle.load(open(config["Reg_Best_Hyperparams"]["xgb"], 'rb'))

    xgboost = xgb.XGBRegressor(random_state=1, n_jobs=-1)
    for key in list(best_hyperparams.keys()):
        setattr(xgboost, key, best_hyperparams[key])
        
    xgboost.fit(train_x, train_y)
    test_pred_xgboost = xgboost.predict(test_x)
    test_r2_xgboost = sklearn.metrics.r2_score(test_y, test_pred_xgboost)
    print('Test R^2: ' + str(test_r2_xgboost))

    test['MonitorData_pred'] = pd.Series(test_pred_xgboost, index=test.index)
    test.to_csv(config["Regression"]['xgb_pred'], index=False)
    #pickle.dump(xgboost, open('xgboost_final.pkl', 'wb'))

    feature_importance_df = pd.DataFrame(xgboost.feature_importances_.reshape(len(xgboost.feature_importances_), -1), columns=['XGBoost_Feature_Importance'])
    feature_importance_df['Variable'] = pd.Series(train_x.columns, index=feature_importance_df.index)
    feature_importance_df.to_csv(config["Regression"]['xgb_ft'], index=False)  

    #Variable importance plot
    #xgb.plot_importance(model, max_num_features=20)
    #pyplot.show()


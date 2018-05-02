import argparse
import configparser
import pandas as pd
import numpy as np
import pickle
from data_split_tune_utils import train_test_split, X_y_site_split, train_val_test_split
from predictiveImputer_mod import PredictiveImputer


parser = argparse.ArgumentParser()


# add optional validation set argument
parser.add_argument("--val", 
    help="Train using a validation set in addition to train and test sets",
    action="store_true")


# add optional training data split proportion argument
parser.add_argument("--impute_split", 
    help="Specify what proportion of the training set to fit the imputer on. " + \
         "Must lie between 0 and 1. Default value is 0.23.",
    type=float,
    default=0.23)


# add optional initial imputation stragegy argument
parser.add_argument("--initial_strategy", 
    help="Determines how to impute columns prior to first iteration of method. Default is mean imputation.",
    choices=["mean", "median"],
    default="mean")


# add optional backup imputation stragegy argument
parser.add_argument("--backup_strategy", 
    help="Determines how to impute columns should ridge imputer perform poorly. Default is mean imputation.",
    choices=["mean", "median"],
    default="mean")


# add optional maximum iteration argument for Predictive Imputer
parser.add_argument("--max_iter", 
    help="Determines the maximum number of iterations Predictive Imputer can run for, " + \
         "if the stopping criterion is not reached. Default is 10.",
    type=int,
    default=10)


# add optional maximum features argument for Predictive Imputer
parser.add_argument("-- max_features", 
    help="Determines the number of features randomly chosen at each node of the trees in " + \
	     "PredictiveImputer's random forest. Default is 15.",
    type=int,
    default=15)


# add optional number of estimators argument random forest in Predictive Imputer
parser.add_argument("--n_estimators", 
    help="Determines the number of trees to use in PredictiveImputer's random forest. " + \
     "Default is 25.",
    type=int,
    default=25)


args = parser.parse_args()

config = configparser.RawConfigParser()
config.read("config/py_config.ini")

train, val, test = None, None, None


if args.val:
    train = pd.read_csv(config["data"]["trainV"])
    val   = pd.read_csv(config["data"]["valV"])
    test  = pd.read_csv(config["data"]["testV"])

else:
    train = pd.read_csv(config["data"]["train"])
    test  = pd.read_csv(config["data"]["test"])

np.random.seed(1)

train1, train2 = train_test_split(train, train_prop=args.impute_split, site_var_name="site")

train1_x, train1_y, train1_sites = X_y_site_split(train1, y_var_name="MonitorData", site_var_name="site")
train2_x, train2_y, train2_sites = X_y_site_split(train2, y_var_name="MonitorData", site_var_name="site")
test_x, test_y, test_sites = X_y_site_split(test, y_var_name="MonitorData", site_var_name="site")

rf_imputer = PredictiveImputer(max_iter=args.max_iter, initial_strategy=args.initial_strategy, f_model="RandomForest")
rf_imputer.fit(train1_x, max_features=args.max_features, n_estimators=args.n_estimators, n_jobs=-1, verbose=0, random_state=1)

train1_x_imp, train1_r2_scores_df = rf_imputer.transform(train1_x, evaluate=True, backup_impute_strategy=args.backup_strategy)
train1_r2_scores_df.columns = ["Train1_R2", "Train1_num_missing"]
train1_r2_scores_df.loc[max(train1_r2_scores_df.index)+1, :] = [np.average(train1_r2_scores_df.loc[:, "Train1_R2"].values,
                                                                   weights=train1_r2_scores_df.loc[:, "Train1_num_missing"].values,
                                                                   axis=0), np.mean(train1_r2_scores_df.loc[:, "Train1_num_missing"].values)]
    
train2_x_imp, train2_r2_scores_df = rf_imputer.transform(train2_x, evaluate = True, backup_impute_strategy = "mean")
train2_r2_scores_df.columns = ["Train2_R2", "Train2_num_missing"]
train2_r2_scores_df.loc[max(train2_r2_scores_df.index)+1, :] = [np.average(train2_r2_scores_df.loc[:, "Train2_R2"].values,
                                                                   weights=train2_r2_scores_df.loc[:, "Train2_num_missing"].values,
                                                                   axis=0), np.mean(train2_r2_scores_df.loc[:, "Train2_num_missing"].values)]


test_x_imp, test_r2_scores_df = rf_imputer.transform(test_x, evaluate = True, backup_impute_strategy = "mean")
test_r2_scores_df.columns = ["Test_R2", "Test_num_missing"]
test_r2_scores_df.loc[max(test_r2_scores_df.index)+1, :] = [np.average(test_r2_scores_df.loc[:, "Test_R2"].values,
                                                                   weights = test_r2_scores_df.loc[:, "Test_num_missing"].values,
                                                                   axis=0), np.mean(test_r2_scores_df.loc[:, "Test_num_missing"].values)]

cols = ["site", "MonitorData"] + list(train1_x.columns)
train1_imp_df = pd.DataFrame(np.concatenate([train1_sites.values.reshape(len(train1_sites), -1),
                                              train1_y.values.reshape(len(train1_y), -1),
                                              train1_x_imp], axis=1),
                                              columns=cols)

train2_imp_df = pd.DataFrame(np.concatenate([train2_sites.values.reshape(len(train2_sites), -1),
                                              train2_y.values.reshape(len(train2_y), -1),
                                              train2_x_imp], axis=1),
                                              columns=cols)

test_imp_df = pd.DataFrame(np.concatenate([test_sites.values.reshape(len(test_sites), -1),
                                              test_y.values.reshape(len(test_y), -1),
                                              test_x_imp], axis=1),
                                              columns=cols)

var_df = pd.DataFrame(np.array(cols[2:] + ["Weighted_Mean_R2"]).reshape(len(cols)-2+1, -1), columns=["Variable"])

train_imp_df = pd.concat([train1_imp_df, train2_imp_df])
train_imp_df = train_imp_df.reset_index().sort_values(["site", "index"])
train_imp_df.drop("index", axis=1, inplace=True)
train_imp_df.reset_index(inplace=True, drop=True)
train_imp_df.to_csv(config["RF_Imputation"]["train"], index=False)
test_imp_df.to_csv(config["RF_Imputation"]["test"], index=False)
#pickle.dump(rf_imputer, open("rfV_imputer.pkl", "wb"))


if args.val:
    val_x, val_y, val_sites = X_y_site_split(val, y_var_name="MonitorData", site_var_name="site")
    val_x_imp, val_r2_scores_df = rf_imputer.transform(val_x, evaluate = True, backup_impute_strategy = "mean")
    val_r2_scores_df.columns = ["Val_R2", "Val_num_missing"]

    val_r2_scores_df.loc[max(val_r2_scores_df.index)+1, :] = [np.average(val_r2_scores_df.loc[:, "Val_R2"].values,
                                                                       weights = val_r2_scores_df.loc[:, "Val_num_missing"].values,
                                                                       axis=0), np.mean(val_r2_scores_df.loc[:, "Val_num_missing"].values)]

    val_imp_df = pd.DataFrame(np.concatenate([val_sites.values.reshape(len(val_sites), -1),
                                                  val_y.values.reshape(len(val_y), -1),
                                                  val_x_imp], axis=1),
                                                  columns=cols)

    val_imp_df.to_csv(config["RF_Imputation"]["val"], index=False)

    r2_scores_df = pd.concat([var_df, train1_r2_scores_df, train2_r2_scores_df, val_r2_scores_df, test_r2_scores_df], axis=1)

    r2_scores_df.to_csv(config["RF_Imputation"]["r2_scores"], index=False)


else:
    r2_scores_df = pd.concat([var_df, train1_r2_scores_df, train2_r2_scores_df, test_r2_scores_df], axis=1)
    r2_scores_df.to_csv(config["RF_Imputation"]["r2_scores"], index=False)



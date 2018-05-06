"""Description: This script fits ridge regression models for imputing missing data on a 
proportion of the training data that is specified by the user using the MissForest algorithm
(see PredictiveImputer in predictiveImputer_mod.py for more details). The fitted imputer is saved 
and can be used to impute other datasets (e.g., the test set).
"""
import argparse
import configparser
import pandas as pd
import numpy as np
import pickle
# these are imported functions created for this package that split datasets (see data_split_tune_utils.py)
from data_split_tune_utils import train_test_split, X_y_site_split
# this is the PredictiveImputer class inspired by the MissForest algorithm (see predictiveImputer_mod.py)
from predictiveImputer_mod import PredictiveImputer

config = configparser.RawConfigParser()
config.read("config/py_config.ini")

parser = argparse.ArgumentParser()

# add optional validation set argument
parser.add_argument("--val", 
    help="Use the \"trainV\" dataset (generated via train-test-validation split) as opposed" + \
         " to the \"train\" dataset (train-test only) to fit",
    action="store_true" )


# add optional training data split proportion argument
parser.add_argument("--impute_split", 
    help="Specify what proportion of the training set to fit the imputer on. " + \
         "Must lie between 0 and 1. Default value is 0.3.",
    type=float,
    default=0.3)


# add optional maximum iteration argument for Predictive Imputer
parser.add_argument("--max_iter", 
    help="Determines the maximum number of iterations PredictiveImputer can run for, " + \
         "if the stopping criterion is not reached. Default is 10.",
    type=int,
    default=10)


# add optional initial strategy argument
parser.add_argument("--initial_strategy", 
    help="Determines how to impute columns prior to first iteration of method. Default is mean imputation.",
    choices=["mean", "median"],
    default="mean")


# add optional training data split proportion argument
parser.add_argument("--alpha", 
    help="Ridge imputation regularization parameter. Default is 0.0001.",
    type=float,
    default=0.0001)


args = parser.parse_args()

if args.impute_split < 0 or args.impute_split > 1:
    print("Impute split value out of range!")
    sys.exit()


train = None
if args.val:
    train = pd.read_csv(config["data"]["trainV"])

else:
    train = pd.read_csv(config["data"]["train"])


# set seed for reproducibility
np.random.seed(1)

# split train up; only fit imputer on part of train set due to memory/time
train1, train2 = train_test_split(train, train_prop=args.impute_split, site_var_name="site")

# split part of train set to fit imputer on into x, y, and site
train1_x, train1_y, train1_sites = X_y_site_split(train1, y_var_name="MonitorData", site_var_name="site")

# create imputer and fit on part of train set
ridge_imputer = PredictiveImputer(max_iter=args.max_iter, initial_strategy=args.initial_strategy, f_model="Ridge")
ridge_imputer.fit(train1_x, alpha=args.alpha, fit_intercept=True, normalize=True, random_state=1)

# save fitted imputer
pickle.dump(ridge_imputer, open(config["Ridge_Imputation"]["model"], "wb"))
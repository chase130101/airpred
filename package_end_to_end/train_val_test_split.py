"""Description: This script splits a dataset into either train/test sets or train/validation/test sets
and saves the new, separated datasets as csv files. The data splits will be such that site-days for the
same site do not get split up.
"""
import argparse
import numpy as np
import pandas as pd
import sys
# these are imported functions created for this package that split datasets (see data_split_tune_utils.py)
from data_split_tune_utils import train_test_split, train_val_test_split

config = configparser.RawConfigParser()
config.read("config/py_config.ini")


parser = argparse.ArgumentParser()

# add optional validation set argument
parser.add_argument("--val", 
    help="Create a validation set in addition to train and test sets",
    action="store_true" )


# add optional training data split proportion argument
parser.add_argument("--train_split", 
    help="Specify how much of train-test(-val) split is training data. " + \
         "Must lie between 0 and 1.",
    type=float,
    default=0.7)


# add optional validation data split proportion argument
parser.add_argument("--val_split", 
    help="Specify how much of train-test-val split is validation data. " + \
         "Must lie between 0 and 1. By default, test and val are split 50/50 between the non-training data. " + \
         "This option is valid only when the --val flag is enabled!",
    type=float,
    default=0)


args = parser.parse_args()


if (not args.val) and args.val_split != 0:
    print("Validation split specified without validation flag!")
    sys.exit()


if args.train_split < 0 or args.train_split > 1 or \
   args.val_split < 0 or args.val_split > 1:
    print("Split value out of range!")
    sys.exit()


if args.train_split + args.val_split > 1:
    print("Invalid train/validation split ratio! Must fall under a total of 1.")
    sys.exit()


test_split = 1. - args.train_split - args.val_split


data = pd.read_csv(config["data"]["data_to_impute"])

# set seed for reproducibility
np.random.seed(1)

if args.val: # split data into train, validation, and test sets and save
    train, val, test = train_val_test_split(data, train_prop=args.train_split, test_prop=test_split, site_var_name="site")
    train.to_csv(config["data"]["trainV"], index=False)
    val.to_csv(  config["data"]["valV"], index=False)
    test.to_csv( config["data"]["testV"], index=False)


else: # split data into train and test sets and save
    train, test = train_test_split(data, train_prop=args.train_split, site_var_name="site")
    train.to_csv(config["data"]["train"], index=False)
    test.to_csv(config["data"]["test"], index=False)
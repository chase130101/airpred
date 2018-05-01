import numpy as np
import pandas as pd
from data_split_tune_utils import train_test_split, train_val_test_split
import argparse

parser = argparse.ArgumentParser()

# add optional validation set argument
parser.add_argument("--val", 
    help="Create a validation set in addition to train and test sets",
    action="store_true" )

args = parser.parse_args()

config = configparser.RawConfigParser()
config.read('config/py_config.ini')

data = pd.read_csv(config["data"]["data_to_impute"])

np.random.seed(1)


if args.val: # create validation set
    train, val, test = train_val_test_split(data, train_prop = 0.7, test_prop = 0.15, site_var_name = 'site')
    train.to_csv(config["data"]["trainV"], index = False)
    val.to_csv(  config["data"]["valV"], index = False  )
    test.to_csv( config["data"]["testV"], index = False )


else:
    train, test = train_test_split(data, train_prop = 0.8, site_var_name = 'site')
    train.to_csv(config["data"]["train"], index = False)
    test.to_csv(config["data"]["test"], index = False)

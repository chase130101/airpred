import argparse
from predictive_imputer import predictive_imputer
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "Path of input file")
args = parser.parse_args()

data = pd.read_csv(args.filename)

to_remove = ["site", "date", "MonitorData"]

data = data.drop(to_remove, axis = 1) # drop columns

impute = predictive_imputer.PredictiveImputer()
impute.fit(data)
print(impute.transform(data.as_matrix()))

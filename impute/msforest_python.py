import argparse
from predictive_imputer import predictive_imputer
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("filename", help = "Path of input file")
parser.add_argument("output_file", help = "Path of output file")
args = parser.parse_args()

data = pd.read_csv(args.filename)

to_remove = ["site", "date", "MonitorData"]  
# edit this to drop columns

cols = [n for n in to_remove if n in data.columns]

data = data.drop(cols, axis = 1) # drop columns as applicable

impute = predictive_imputer.PredictiveImputer()
impute.fit(data)

#df = pd.DataFrame(impute.transform(data.as_matrix()), columns = data.columns)
#df.to_csv(args.output_file)
np.savetxt(args.output_file, impute.transform(data.as_matrix()), 
delimiter=",",header=','.join(data.columns), comments="")


import argparse
import numpy as np
import pandas as pd
import pickle
import time
from predictiveImputer_mod import PredictiveImputer
     

parser = argparse.ArgumentParser()
parser.add_argument("subset_file", help = "Path of subset data file")
parser.add_argument("full_file", help = "Path of full data file")
parser.add_argument("output_file", help = "Path of output file")
args = parser.parse_args()

subset_data = pd.read_csv(args.subset_file)
full_data = pd.read_csv(args.full_file)

subset_data = pd.get_dummies(subset_data, columns = ["month"])
full_data = pd.get_dummies(full_data, columns = ["month"])


to_remove = ["site", "date", "MonitorData"]  
# edit this to drop columns

cols_to_remove = [n for n in to_remove if n in subset_data.columns]

#Remove columns not needed for imputation
full_data_to_impute = full_data.drop(cols_to_remove, axis = 1) 
subset_data_to_impute = subset_data.drop(cols_to_remove, axis = 1) 

other_data = full_data.loc[:,cols_to_remove]

impute = PredictiveImputer(f_model = "RandomForest")
print("Beginning fit()")
start = time.time()
impute.fit(subset_data_to_impute) #Train model
print("Time elapsed for fit(): {} seconds".format(time.time() - start))

pickle.dump(impute, open('miss_forest.pkl', 'wb')) #Save model

print("Converting to pandas dataframe...")
start = time.time()
full_imputed_data = pd.DataFrame(impute.transform(full_data_to_impute), columns = full_data_to_impute.columns)
full_data_complete = pd.concat([other_data, full_imputed_data], axis = 1) #Add previously removed columns back
print("Time elapsed on pandas df: {}".format(time.time() - start))

full_data_complete.to_csv(args.output_file)

#np.savetxt(args.output_file, impute.transform(data.as_matrix()), 
#delimiter=",",header=','.join(data.columns), comments="")
import random
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri

# filename = "testrds.rds"  # for testing on smaller RDS file
filename = "data.rds"

pandas2ri.activate()
print("pandas2ri activated...")
readRDS = robjects.r['readRDS']
print("Reading RDS file...")
df = readRDS(filename)

print("Done reading file.")
print(df.colnames)

num_rows = df.nrow

print("Dataframe has {} rows".format(num_rows))

sample_size = int(num_rows * 0.005)
print("Taking a 0.5% sample ({} elements)...".format(sample_size))

sample = random.sample(range(1, num_rows), sample_size)
rows_i = robjects.IntVector(sample)
print("Generated random indices. Sampling dataframe...")
subdata = df.rx(rows_i, True)
print("Constructed subset of data. Writing to CSV...")
subdata.to_csvfile("random_subset.csv")
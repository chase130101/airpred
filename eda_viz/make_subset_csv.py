import random
import pandas as pd

#filename = "~/airpred/data/assembled_data.rds"
filename = "~/airpred/data/assembled_data.csv"
#filename = "test.csv"

df = pd.read_csv(filename)

print("Done reading file.")

num_rows = len(df)

print("Dataframe has {} rows".format(num_rows))

percent_sample = 10.
sample_size = int(num_rows * percent_sample / 100)
print("Taking a {}% sample ({} elements)...".format(percent_sample, sample_size))

print("Generated random indices. Sampling dataframe...")

subdata = df.sample(frac = percent_sample / 100.)

print("Constructed subset of data. Writing to CSV...")
subdata.to_csv("~/airpred/eda/random_subset_10pct.csv")

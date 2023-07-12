'''
Randomly sample 10% of the original documents as the validation set.
For each topic modelling method, I will manually check the accuracy on this validation set and compare the performance.
'''

import pandas as pd
from config import *

# Load the whole dataset
df = pd.read_csv(data_path)

# Sample 10%
df_sample = df.sample(frac = 0.1).reset_index(drop = True)

# Save output
df_sample.to_csv(sample_path, index = False)
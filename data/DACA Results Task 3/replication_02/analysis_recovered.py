# Code recovered from run_log_02.md
# Task 3, Replication 02
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)

# --- Code block 2 ---
FT = b0 + b1*ELIGIBLE + b2*AFTER + b3*(ELIGIBLE*AFTER) + covariates + error


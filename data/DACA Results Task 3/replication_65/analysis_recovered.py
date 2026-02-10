# Code recovered from run_log_65.md
# Task 3, Replication 65
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)


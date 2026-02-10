# Code recovered from run_log_05.md
# Task 3, Replication 05
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# --- Code block 2 ---
FT = α + β₁(ELIGIBLE) + β₂(AFTER) + β₃(ELIGIBLE × AFTER) + ε

# --- Code block 3 ---
model.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})


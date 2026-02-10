# Code recovered from run_log_45.md
# Task 3, Replication 45
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# --- Code block 2 ---
# Interaction term for DiD
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Female indicator (IPUMS: SEX=1 Male, SEX=2 Female)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Married indicator (IPUMS: MARST=1 or 2 = married)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# --- Code block 3 ---
model = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + AGE + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(STATEFIP)',
                data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP'].values})


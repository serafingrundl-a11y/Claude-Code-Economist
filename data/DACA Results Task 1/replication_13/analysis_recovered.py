# Code recovered from run_log_13.md
# Task 1, Replication 13
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

cols_to_keep = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
                'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
                'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP', 'LABFORCE']

chunks = []
for chunk in pd.read_csv('data/data.csv', usecols=cols_to_keep, chunksize=1000000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
# Total: 991,261 observations
# Non-citizens (CITIZEN == 3): 701,347
# Working age (18-64): 603,425

# --- Code block 2 ---
df_noncit['age_at_arrival'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
df_noncit['daca_eligible'] = ((df_noncit['age_at_arrival'] < 16) &
                               (df_noncit['BIRTHYR'] >= 1982) &
                               (df_noncit['YRIMMIG'] <= 2007))

# DACA-eligible (working age): 77,038 (12.8%)
# DACA-ineligible (working age): 526,387 (87.2%)

# --- Code block 3 ---
df_working['fulltime'] = (df_working['UHRSWORK'] >= 35).astype(int)
df_working['employed'] = (df_working['EMPSTAT'] == 1).astype(int)
df_working['post'] = (df_working['YEAR'] >= 2013).astype(int)
df_working['eligible'] = df_working['daca_eligible'].astype(int)
df_working['female'] = (df_working['SEX'] == 2).astype(int)
df_working['married'] = (df_working['MARST'] <= 2).astype(int)
df_working['age_sq'] = df_working['AGE'] ** 2
df_working['less_hs'] = (df_working['EDUC'] < 6).astype(int)
df_working['some_college'] = (df_working['EDUC'].isin([7, 8, 9])).astype(int)
df_working['college_plus'] = (df_working['EDUC'] >= 10).astype(int)
df_working['eligible_post'] = df_working['eligible'] * df_working['post']

# --- Code block 4 ---
# Preferred specification: Weighted regression with state and year FE, clustered SE by state
model = smf.wls('fulltime ~ eligible + post + eligible_post + AGE + age_sq + female + married + less_hs + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                data=df_working, weights=df_working['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_working['STATEFIP']})


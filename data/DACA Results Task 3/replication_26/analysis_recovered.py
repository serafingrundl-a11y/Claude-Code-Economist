# Code reconstructed from run_log_26.md and replication_report_26.tex
# Task 3, Replication 26
# This file was reconstructed from the run log and replication report
# (no code blocks were present in the run log).

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --- Data Loading ---
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
# Shape: (17382, 105)

# --- Variable Construction ---
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] <= 2).astype(int)

# --- Model 1: Basic OLS ---
m1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
# DiD = 0.0643, SE = 0.0153, p < 0.001, N = 17,382

# --- Model 2: WLS with State-Clustered SE (PREFERRED) ---
m2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
              data=df, weights=df['PERWT']).fit(
              cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# DiD = 0.0748, SE = 0.0203, p < 0.001, 95% CI: [0.035, 0.115], N = 17,382

# --- Model 3: With Demographic Controls ---
m3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + C(EDUC_RECODE) + AGE',
              data=df, weights=df['PERWT']).fit(
              cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# DiD = 0.0640, SE = 0.0217, p = 0.003

# --- Model 4: With Year Fixed Effects ---
m4 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + NCHILD + C(EDUC_RECODE) + AGE + C(YEAR)',
              data=df, weights=df['PERWT']).fit(
              cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
# DiD = 0.0612, SE = 0.0210, p = 0.004

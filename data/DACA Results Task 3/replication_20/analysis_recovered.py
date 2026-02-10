# Code recovered from run_log_20.md
# Task 3, Replication 20
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(df.shape)  # (17382, 105)
print(df['YEAR'].value_counts().sort_index())

# --- Code block 2 ---
# Manual calculation
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
# Treated change: 0.6658 - 0.6263 = 0.0394
# Control change: 0.6449 - 0.6697 = -0.0248
# DiD = 0.0394 - (-0.0248) = 0.0643

# --- Create interaction term ---
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['SEX_female'] = (df['SEX'] == 2).astype(int)

# --- Code block 3 ---
import statsmodels.formula.api as smf

# Model 1: Basic DiD
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 2: With Year FE
model2 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 3: With Year + State FE
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')

# Model 4: Full unweighted
model4 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + SEX_female + AGE + MARRIED + NCHILD + C(EDUC_RECODE)', data=df).fit(cov_type='HC1')

# Model 5: Weighted basic
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 6: Weighted full (PREFERRED)
model6 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + SEX_female + AGE + MARRIED + NCHILD + C(EDUC_RECODE)', data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Model 7: Clustered SE
model7 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# --- Code block 4 ---
import matplotlib.pyplot as plt
# Created parallel_trends.png and event_study.png


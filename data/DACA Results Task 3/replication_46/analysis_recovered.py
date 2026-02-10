# Code recovered from run_log_46.md
# Task 3, Replication 46
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# --- Code block 2 ---
# Weighted means by group
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
)

# Treatment: 0.637 (pre) -> 0.686 (post), diff = +0.049
# Control: 0.689 (pre) -> 0.663 (post), diff = -0.026
# DiD = 0.049 - (-0.026) = 0.075 (7.5 pp)

# --- Code block 3 ---
import statsmodels.api as sm
from statsmodels.api import WLS

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic unweighted
X1 = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model1 = sm.OLS(df['FT'], X1).fit(cov_type='HC1')

# Model 2: Basic weighted
model2 = WLS(df['FT'], X1, weights=df['PERWT']).fit(cov_type='HC1')

# Models 3-6: Add year FE, demographics, education, state FE progressively

# --- Code block 4 ---
# Create ELIGIBLE x Year interactions (ref: 2011)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year)

# --- Code block 5 ---
df_pre = df[df['AFTER'] == 0]
df_pre['FAKE_AFTER'] = (df_pre['YEAR'] >= 2010).astype(float)
df_pre['FAKE_TREAT'] = df_pre['ELIGIBLE'] * df_pre['FAKE_AFTER']

# --- Code block 6 ---
model.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})


# Code recovered from run_log_24.md
# Task 3, Replication 24
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# --- Code block 2 ---
# Unweighted means
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()

# Weighted means
def wmean(group, col='FT', wt='PERWT'):
    return np.average(group[col], weights=group[wt])
weighted_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(lambda x: wmean(x))

# --- Code block 3 ---
import statsmodels.api as sm

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Basic weighted OLS
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']])
model = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 4 ---
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True).astype(float)
X_year = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER']], year_dummies], axis=1)
model_year = sm.WLS(df['FT'], X_year, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 5 ---
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True).astype(float)
X_both = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER']], year_dummies, state_dummies], axis=1)
model_both = sm.WLS(df['FT'], X_both, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 6 ---
df['FEMALE'] = (df['SEX'] == 2).astype(float)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True).astype(float)
marst_dummies = pd.get_dummies(df['MARST'], prefix='MARST', drop_first=True).astype(float)

X_full = pd.concat([const, df[['ELIGIBLE', 'ELIGIBLE_AFTER', 'AGE', 'FEMALE']],
                    educ_dummies, marst_dummies, year_dummies, state_dummies], axis=1)
model_full = sm.WLS(df['FT'], X_full, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 7 ---
# Males only
df_male = df[df['SEX'] == 1]
# [run regression on subset]

# --- Code block 8 ---
# Create year-eligible interactions
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(float)

# --- Code block 9 ---
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['PLACEBO_AFTER'] = (df_pre['YEAR'] >= 2010).astype(float)
df_pre['ELIGIBLE_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_AFTER']


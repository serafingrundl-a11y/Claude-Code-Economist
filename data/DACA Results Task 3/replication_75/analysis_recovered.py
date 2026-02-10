# Code recovered from run_log_75.md
# Task 3, Replication 75
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Dataset shape:', df.shape)  # (17382, 105)

# --- Code block 2 ---
# Simple means calculation
ft_eligible_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # 0.6263
ft_eligible_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()   # 0.6658
ft_control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()   # 0.6697
ft_control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()    # 0.6449

# DiD = (0.6658 - 0.6263) - (0.6449 - 0.6697) = 0.0643

# --- Code block 3 ---
import statsmodels.formula.api as smf
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
# DiD coefficient: 0.0643 (SE: 0.015, p<0.001)

# --- Code block 4 ---
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df,
                  weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0748 (SE: 0.018, p<0.001)

# --- Code block 5 ---
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['EDUC_HS'] = (df['EDUC'] == 7).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC'] == 8).astype(int)
df['EDUC_BA_PLUS'] = (df['EDUC'].isin([10, 11])).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_BA_PLUS'
model3 = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0617 (SE: 0.017, p<0.001)

# --- Code block 6 ---
formula = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_BA_PLUS + C(YEAR) + C(STATEFIP)'
model4 = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0585 (SE: 0.017, p<0.001)

# --- Code block 7 ---
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

# Results (relative to 2008):
# 2009: 0.018 (p=0.553) - pre-trend OK
# 2010: -0.013 (p=0.676) - pre-trend OK
# 2011: 0.065 (p=0.045) - borderline
# 2013: 0.079 (p=0.012) - post-DACA effect
# 2014: 0.051 (p=0.118)
# 2015: 0.054 (p=0.092)
# 2016: 0.124 (p<0.001) - strong effect

# --- Code block 8 ---
df_pre = df[df['AFTER'] == 0].copy()
df_pre['AFTER_FAKE'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_AFTER_FAKE'] = df_pre['ELIGIBLE'] * df_pre['AFTER_FAKE']
# Placebo coefficient: 0.017 (p=0.443) - not significant as expected

# --- Code block 9 ---
# Males: DiD = 0.061 (SE: 0.020, p=0.002)
# Females: DiD = 0.053 (SE: 0.028, p=0.056)

# --- Code block 10 ---
# With DRIVERSLICENSES, INSTATETUITION, EVERIFY, SECURECOMMUNITIES
# DiD coefficient: 0.060 (p<0.001) - robust to policy controls


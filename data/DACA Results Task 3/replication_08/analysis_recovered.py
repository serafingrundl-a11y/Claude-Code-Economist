# Code recovered from run_log_08.md
# Task 3, Replication 08
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)

# --- Code block 2 ---
df['female'] = (df['SEX'] == 'Female').astype(int)
df['married'] = (df['MARST'] == 'Married, spouse present').astype(int)
df['has_children'] = (df['NCHILD'] > 0).astype(int)
df['interaction'] = df['ELIGIBLE'] * df['AFTER']

# Education dummies (reference: Less than High School)
df['educ_hs'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['educ_somecollege'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_2yr'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# --- Code block 3 ---
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER', data=df).fit(cov_type='HC1')

# --- Code block 4 ---
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 5 ---
formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE:AFTER + female + married + has_children + educ_hs + educ_somecollege + educ_2yr + educ_ba'
model3 = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 6 ---
# Reference year: 2011
years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'eligible_x_year_{year}'] = df['ELIGIBLE'] * df[f'year_{year}']


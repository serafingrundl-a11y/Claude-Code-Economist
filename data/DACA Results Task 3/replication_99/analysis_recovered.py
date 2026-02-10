# Code recovered from run_log_99.md
# Task 3, Replication 99
# This file was reconstructed from the run log.

# --- Code block 1 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)

# --- Code block 2 ---
# Treatment interaction
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Demographics
df['FEMALE'] = (df['SEX'] == 2).astype(float)  # SEX=2 is female per IPUMS
df['MARRIED'] = (df['MARST'] <= 2).astype(float)  # MARST 1-2 is married

# Education dummies (HS as reference)
df['educ_somecoll'] = (df['EDUC_RECODE'] == 'Some College').astype(float)
df['educ_twoyear'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(float)
df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(float)

# Fixed effects
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)

# --- Code block 3 ---
for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_Y{y}'] = (df['ELIGIBLE'] * (df['YEAR'] == y)).astype(float)

# --- Code block 4 ---
# Created parallel_trends.pdf showing FT rates by group over time
plt.savefig('figures/parallel_trends.pdf')

# --- Code block 5 ---
# Created event_study.pdf showing year-specific coefficients
plt.savefig('figures/event_study.pdf')


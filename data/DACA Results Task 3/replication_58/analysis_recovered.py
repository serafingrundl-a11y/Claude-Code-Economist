# Code recovered from run_log_58.md
# Task 3, Replication 58
# This file was reconstructed from the run log.

# --- Code block 1 ---
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]

# --- Code block 2 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)  # (17382, 105)

# --- Code block 3 ---
pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
# AFTER        0     1    All
# ELIGIBLE
# 0         3294  2706   6000
# 1         6233  5149  11382
# All       9527  7855  17382

# --- Code block 4 ---
import statsmodels.api as sm
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']
X = sm.add_constant(df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER']])
model = sm.WLS(df['FT'], X, weights=df['PERWT']).fit(cov_type='HC1')

# --- Code block 5 ---
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)
df['educ_some_college'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_two_year'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba_plus'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# --- Code block 6 ---
for y in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'year_{y}'] = (df['YEAR'] == y).astype(int)

# --- Code block 7 ---
model = sm.WLS(y, X, weights=w).fit(cov_type='cluster', cov_kwds={'groups': groups})

# --- Code block 8 ---
for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_x_{y}'] = df['ELIGIBLE'] * (df['YEAR'] == y).astype(int)

# --- Code block 9 ---
import matplotlib.pyplot as plt
# Plot weighted FT rates by year for eligible and control groups
plt.savefig('figure1_trends.png', dpi=150)

# --- Code block 10 ---
# Plot year-specific treatment effects with 95% CIs
plt.savefig('figure2_event_study.png', dpi=150)


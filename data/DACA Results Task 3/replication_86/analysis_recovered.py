# Code recovered from run_log_86.md
# Task 3, Replication 86
# This file was reconstructed from the run log.

# --- Code block 1 ---
from docx import Document
doc = Document('replication_instructions.docx')
[print(p.text) for p in doc.paragraphs]

# --- Code block 2 ---
import pandas as pd
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print('Shape:', df.shape)
# Output: Shape: (17382, 105)

# --- Code block 3 ---
import pandas as pd
import numpy as np

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# 2x2 DiD calculation
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

# Results:
# Pre-treatment: Treated=0.637, Control=0.689
# Post-treatment: Treated=0.686, Control=0.663
# DiD = (0.686-0.637) - (0.663-0.689) = 0.049 - (-0.026) = 0.075

# --- Code block 4 ---
import statsmodels.api as sm

df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Model 1: Basic OLS
X1 = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER']]
X1 = sm.add_constant(X1)
y = df['FT']
model1 = sm.OLS(y, X1).fit(cov_type='HC1')
# DiD coefficient: 0.0643 (SE: 0.0153)

# Model 2: WLS with survey weights
model2 = sm.WLS(y, X1, weights=df['PERWT']).fit(cov_type='HC1')
# DiD coefficient: 0.0748 (SE: 0.0181)

# Model 3: WLS with covariates
# Includes: FEMALE, MARRIED, NCHILD, education dummies
# DiD coefficient: 0.0641 (SE: 0.0167)

# Model 4: WLS with covariates + year FE + state FE
# DiD coefficient: 0.0608 (SE: 0.0166)

# --- Code block 5 ---
pre_df = df[df['AFTER'] == 0].copy()
pre_df['YEAR_CENTERED'] = pre_df['YEAR'] - 2008
pre_df['ELIGIBLE_X_YEAR'] = pre_df['ELIGIBLE'] * pre_df['YEAR_CENTERED']

X_pre = pre_df[['ELIGIBLE', 'YEAR_CENTERED', 'ELIGIBLE_X_YEAR']]
X_pre = sm.add_constant(X_pre)
model_pre = sm.WLS(pre_df['FT'], X_pre, weights=pre_df['PERWT']).fit(cov_type='HC1')

# ELIGIBLE_X_YEAR coefficient: 0.0174 (SE: 0.011, p=0.113)
# Conclusion: Cannot reject parallel trends assumption

# --- Code block 6 ---
# Reference year: 2011 (last pre-treatment year)
# Coefficients relative to 2011:
# 2008: -0.068 (SE: 0.035)
# 2009: -0.050 (SE: 0.036)
# 2010: -0.082 (SE: 0.036) **
# 2013: +0.016 (SE: 0.038)
# 2014: +0.000 (SE: 0.038)
# 2015: +0.001 (SE: 0.038)
# 2016: +0.074 (SE: 0.038) *

# --- Code block 7 ---
# Males: DiD = 0.0716 (SE: 0.0199) ***
# Females: DiD = 0.0527 (SE: 0.0281) *

# --- Code block 8 ---
import matplotlib.pyplot as plt

trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trends.index, trends[1], 'b-o', label='Treated (Ages 26-30)')
ax.plot(trends.index, trends[0], 'r--s', label='Control (Ages 31-35)')
ax.axvline(x=2012, color='gray', linestyle=':', label='DACA Implementation')
plt.savefig('figure1_trends.png', dpi=150)

# --- Code block 9 ---
# Plotted event study coefficients with 95% CIs
plt.savefig('figure2_eventstudy.png', dpi=150)

# --- Code block 10 ---
# Simple 2-point DiD visualization with counterfactual
plt.savefig('figure3_did.png', dpi=150)


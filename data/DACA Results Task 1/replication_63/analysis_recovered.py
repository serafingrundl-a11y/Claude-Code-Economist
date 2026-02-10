# Code recovered from run_log_63.md and replication_report_63.tex
# Task 1, Replication 63
# This file was reconstructed from the run log and report.

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Step 1: Load data ---
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'LABFORCE', 'UHRSWORK']

df = pd.read_csv('data/data.csv', usecols=cols_needed)

# --- Step 2: Filter to Mexican-born Hispanic non-citizens ---
df = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)]

# --- Step 3: Define DACA eligibility ---
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_young'] = df['age_at_immig'] < 16
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1996) &
                   df['arrived_young'] & df['in_us_since_2007']).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981) &
                 df['arrived_young'] & df['in_us_since_2007']).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control'] == 1)]

# --- Step 4: Create outcome and other variables ---
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Exclude 2012
df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)

# DiD interaction
df['did'] = df['treatment'] * df['post']

# Demographics
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['age_sq'] = df['AGE'] ** 2

# Education dummies (less than HS is reference)
df['hs'] = (df['EDUC'] == 6).astype(int)
df['some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['college_plus'] = (df['EDUC'] >= 10).astype(int)

print(f"Analysis sample: {len(df)} observations")
print(f"  Treatment: {df['treatment'].sum()}")
print(f"  Control: {df['control'].sum()}")

# --- Step 5: Summary statistics ---
print("\n--- Summary Statistics ---")
for grp_name, grp in [('Treat Pre', df[(df['treatment']==1) & (df['post']==0)]),
                       ('Treat Post', df[(df['treatment']==1) & (df['post']==1)]),
                       ('Ctrl Pre', df[(df['control']==1) & (df['post']==0)]),
                       ('Ctrl Post', df[(df['control']==1) & (df['post']==1)])]:
    print(f"\n{grp_name} (N={len(grp)}):")
    print(f"  Fulltime: {grp['fulltime'].mean():.3f}")
    print(f"  Age: {grp['AGE'].mean():.1f}")
    print(f"  Female: {grp['female'].mean():.3f}")

# --- Step 6: Difference-in-Differences Regressions ---

# Model 1: Basic DiD
model1 = smf.ols('fulltime ~ treatment + post + did', data=df).fit(cov_type='HC1')
print(f"\nModel 1 (Basic DiD): {model1.params['did']:.4f} (SE={model1.bse['did']:.4f})")

# Model 2: + Demographics
model2 = smf.ols('fulltime ~ treatment + post + did + AGE + age_sq + female + married + hs + some_college + college_plus',
                  data=df).fit(cov_type='HC1')
print(f"Model 2 (+ Demographics): {model2.params['did']:.4f} (SE={model2.bse['did']:.4f})")

# Model 3: + Year FE
model3 = smf.ols('fulltime ~ treatment + did + AGE + age_sq + female + married + hs + some_college + college_plus + C(YEAR)',
                  data=df).fit(cov_type='HC1')
print(f"Model 3 (+ Year FE): {model3.params['did']:.4f} (SE={model3.bse['did']:.4f})")

# Model 4: + State FE (PREFERRED)
model4 = smf.ols('fulltime ~ treatment + did + AGE + age_sq + female + married + hs + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                  data=df).fit(cov_type='HC1')
print(f"Model 4 (+ State FE, PREFERRED): {model4.params['did']:.4f} (SE={model4.bse['did']:.4f})")

# Model 5: + Survey weights
model5 = smf.wls('fulltime ~ treatment + did + AGE + age_sq + female + married + hs + some_college + college_plus + C(YEAR) + C(STATEFIP)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Model 5 (+ Weights): {model5.params['did']:.4f} (SE={model5.bse['did']:.4f})")

print(f"\n*** Preferred Estimate: {model4.params['did']:.4f} (SE={model4.bse['did']:.4f}) ***")
print(f"*** N = {int(model4.nobs)} ***")

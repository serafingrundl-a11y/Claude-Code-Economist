"""
DACA Replication Study - Analysis Script
Replication 97

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for the Deferred
Action for Childhood Arrivals (DACA) program on the probability of full-time
employment (35+ hours per week)?

Author: Replication Study
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_97")

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. Load and Prepare Data
# =============================================================================
print("\n1. Loading data...")

# Read data in chunks due to large file size
chunks = []
chunksize = 1000000

# Define dtypes to reduce memory usage
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'PERWT': 'float32'
}

# Columns we need
usecols = ['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK', 'PERWT']

print("Reading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols,
                                       dtype=dtypes, chunksize=chunksize)):
    # Filter to Hispanic-Mexican born in Mexico
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total Hispanic-Mexican individuals born in Mexico: {len(df):,}")

# =============================================================================
# 2. Create Variables
# =============================================================================
print("\n2. Creating variables...")

# Remove 2012 (cannot distinguish pre/post DACA within year)
df = df[df['YEAR'] != 2012]
print(f"  After removing 2012: {len(df):,}")

# Create post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Create year of immigration (handle missing/zero values)
df['yrimmig'] = df['YRIMMIG'].replace(0, np.nan)

# Calculate age at arrival
df['age_at_arrival'] = df['YEAR'] - df['yrimmig'] - df['AGE'] + (df['YEAR'] - df['yrimmig'])
# More accurate: if someone is AGE in YEAR and immigrated in YRIMMIG
# Their age at arrival would be approximately AGE - (YEAR - YRIMMIG)
df['age_at_arrival'] = df['AGE'] - (df['YEAR'] - df['yrimmig'])

# Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Working age restriction (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
print(f"  Working age (16-64): {len(df):,}")

# =============================================================================
# 3. Define DACA Eligibility
# =============================================================================
print("\n3. Defining DACA eligibility...")

"""
DACA Eligibility Criteria:
1. Arrived unlawfully in US before 16th birthday
2. Not yet 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 and undocumented

For our analysis:
- Age at arrival < 16: age_at_arrival < 16
- Under 31 in 2012: BIRTHYR >= 1982 (conservative, ensures < 31)
- In US since 2007: YRIMMIG <= 2007
- Undocumented: CITIZEN == 3 (not a citizen)
"""

# Create eligibility indicator based on time-invariant characteristics
# We assume non-citizens without naturalization are potentially undocumented
df['undocumented'] = (df['CITIZEN'] == 3).astype(int)

# Age criteria for DACA (under 31 as of June 2012)
df['age_eligible'] = (df['BIRTHYR'] >= 1982).astype(int)

# Arrived before age 16
df['arrival_age_eligible'] = (df['age_at_arrival'] < 16).astype(int)

# Continuous presence since 2007
df['presence_eligible'] = (df['yrimmig'] <= 2007).astype(int)

# DACA eligibility (all criteria met)
df['daca_eligible'] = (
    (df['undocumented'] == 1) &
    (df['age_eligible'] == 1) &
    (df['arrival_age_eligible'] == 1) &
    (df['presence_eligible'] == 1)
).astype(int)

print(f"  DACA eligible: {df['daca_eligible'].sum():,} ({100*df['daca_eligible'].mean():.1f}%)")

# =============================================================================
# 4. Sample Statistics
# =============================================================================
print("\n4. Sample Statistics")
print("-" * 50)

# Full sample
print(f"\nFull Sample (Hispanic-Mexican born in Mexico, ages 16-64):")
print(f"  Total observations: {len(df):,}")
print(f"  Unique years: {sorted(df['YEAR'].unique())}")

# By eligibility
print(f"\nBy DACA Eligibility:")
for elig in [0, 1]:
    sub = df[df['daca_eligible'] == elig]
    print(f"  {'Eligible' if elig else 'Not Eligible'}: N={len(sub):,}")
    print(f"    Mean age: {sub['AGE'].mean():.1f}")
    print(f"    Male: {100*sub['SEX'].eq(1).mean():.1f}%")
    print(f"    Full-time employed: {100*sub['fulltime'].mean():.1f}%")

# By period
print(f"\nBy Period:")
for period in ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']:
    if 'Pre' in period:
        sub = df[df['post'] == 0]
    else:
        sub = df[df['post'] == 1]
    print(f"  {period}: N={len(sub):,}")
    print(f"    Full-time employed: {100*sub['fulltime'].mean():.1f}%")

# =============================================================================
# 5. Create Control Variables
# =============================================================================
print("\n5. Creating control variables...")

# Age squared
df['age_sq'] = df['AGE'] ** 2

# Male indicator
df['male'] = (df['SEX'] == 1).astype(int)

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'],
                        bins=[-1, 2, 6, 10, 11, 15],
                        labels=['less_hs', 'some_hs', 'hs_grad', 'some_college', 'college_plus'])

# Marital status (married = 1 or 2)
df['married'] = df['MARST'].isin([1, 2]).astype(int)

# Create education dummies
educ_dummies = pd.get_dummies(df['educ_cat'], prefix='educ', drop_first=True)
df = pd.concat([df, educ_dummies], axis=1)

# State dummies (for fixed effects)
df['state'] = df['STATEFIP'].astype(str)

print("  Control variables created.")

# =============================================================================
# 6. Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 80)
print("6. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Store results
results_dict = {}

# 6a. Simple 2x2 DiD (means)
print("\n6a. Simple 2x2 DiD Table (Unweighted Means)")
print("-" * 50)

did_table = df.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
did_table.index = ['Not Eligible', 'Eligible']
did_table.columns = ['Pre-DACA', 'Post-DACA']
print(did_table.round(4))

# Calculate DiD manually
pre_elig = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
post_elig = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
pre_inelig = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
post_inelig = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

did_estimate = (post_elig - pre_elig) - (post_inelig - pre_inelig)
print(f"\nDiD Estimate (raw): {did_estimate:.4f}")
print(f"  Change for Eligible: {post_elig - pre_elig:.4f}")
print(f"  Change for Not Eligible: {post_inelig - pre_inelig:.4f}")

results_dict['simple_did'] = {
    'estimate': did_estimate,
    'pre_eligible': pre_elig,
    'post_eligible': post_elig,
    'pre_ineligible': pre_inelig,
    'post_ineligible': post_inelig
}

# 6b. Weighted 2x2 DiD
print("\n6b. Weighted 2x2 DiD Table")
print("-" * 50)

def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

weighted_table = df.groupby(['daca_eligible', 'post']).apply(weighted_mean).unstack()
weighted_table.index = ['Not Eligible', 'Eligible']
weighted_table.columns = ['Pre-DACA', 'Post-DACA']
print(weighted_table.round(4))

# Weighted DiD
w_pre_elig = weighted_mean(df[(df['daca_eligible']==1) & (df['post']==0)])
w_post_elig = weighted_mean(df[(df['daca_eligible']==1) & (df['post']==1)])
w_pre_inelig = weighted_mean(df[(df['daca_eligible']==0) & (df['post']==0)])
w_post_inelig = weighted_mean(df[(df['daca_eligible']==0) & (df['post']==1)])

w_did_estimate = (w_post_elig - w_pre_elig) - (w_post_inelig - w_pre_inelig)
print(f"\nWeighted DiD Estimate: {w_did_estimate:.4f}")

results_dict['weighted_did'] = {
    'estimate': w_did_estimate,
    'pre_eligible': w_pre_elig,
    'post_eligible': w_post_elig,
    'pre_ineligible': w_pre_inelig,
    'post_ineligible': w_post_inelig
}

# =============================================================================
# 7. Regression Analysis
# =============================================================================
print("\n" + "=" * 80)
print("7. REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df['eligible_x_post'] = df['daca_eligible'] * df['post']

# 7a. Basic DiD regression (no controls)
print("\n7a. Model 1: Basic DiD (no controls)")
print("-" * 50)

model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post', data=df).fit()
print(f"DiD Coefficient: {model1.params['eligible_x_post']:.4f}")
print(f"Std Error: {model1.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model1.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model1.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['eligible_x_post', 0]:.4f}, {model1.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {int(model1.nobs):,}")

results_dict['model1'] = {
    'coef': model1.params['eligible_x_post'],
    'se': model1.bse['eligible_x_post'],
    'tstat': model1.tvalues['eligible_x_post'],
    'pvalue': model1.pvalues['eligible_x_post'],
    'ci_low': model1.conf_int().loc['eligible_x_post', 0],
    'ci_high': model1.conf_int().loc['eligible_x_post', 1],
    'rsq': model1.rsquared,
    'n': int(model1.nobs)
}

# 7b. DiD with demographic controls
print("\n7b. Model 2: DiD with demographic controls")
print("-" * 50)

model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + male + married',
                 data=df).fit()
print(f"DiD Coefficient: {model2.params['eligible_x_post']:.4f}")
print(f"Std Error: {model2.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model2.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model2.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['eligible_x_post', 0]:.4f}, {model2.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {int(model2.nobs):,}")

results_dict['model2'] = {
    'coef': model2.params['eligible_x_post'],
    'se': model2.bse['eligible_x_post'],
    'tstat': model2.tvalues['eligible_x_post'],
    'pvalue': model2.pvalues['eligible_x_post'],
    'ci_low': model2.conf_int().loc['eligible_x_post', 0],
    'ci_high': model2.conf_int().loc['eligible_x_post', 1],
    'rsq': model2.rsquared,
    'n': int(model2.nobs)
}

# 7c. DiD with demographic + education controls
print("\n7c. Model 3: DiD with demographics + education")
print("-" * 50)

# Check which education dummies exist
educ_cols = [c for c in df.columns if c.startswith('educ_')]
formula3 = 'fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + male + married + ' + ' + '.join(educ_cols)
model3 = smf.ols(formula3, data=df).fit()

print(f"DiD Coefficient: {model3.params['eligible_x_post']:.4f}")
print(f"Std Error: {model3.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model3.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model3.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['eligible_x_post', 0]:.4f}, {model3.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

results_dict['model3'] = {
    'coef': model3.params['eligible_x_post'],
    'se': model3.bse['eligible_x_post'],
    'tstat': model3.tvalues['eligible_x_post'],
    'pvalue': model3.pvalues['eligible_x_post'],
    'ci_low': model3.conf_int().loc['eligible_x_post', 0],
    'ci_high': model3.conf_int().loc['eligible_x_post', 1],
    'rsq': model3.rsquared,
    'n': int(model3.nobs)
}

# 7d. Full model with year fixed effects
print("\n7d. Model 4: Full model with year fixed effects")
print("-" * 50)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)
year_cols = [c for c in df.columns if c.startswith('year_')]

formula4 = 'fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + male + married + ' + ' + '.join(educ_cols) + ' + ' + ' + '.join(year_cols)
model4 = smf.ols(formula4, data=df).fit()

print(f"DiD Coefficient: {model4.params['eligible_x_post']:.4f}")
print(f"Std Error: {model4.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model4.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model4.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['eligible_x_post', 0]:.4f}, {model4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs):,}")

results_dict['model4'] = {
    'coef': model4.params['eligible_x_post'],
    'se': model4.bse['eligible_x_post'],
    'tstat': model4.tvalues['eligible_x_post'],
    'pvalue': model4.pvalues['eligible_x_post'],
    'ci_low': model4.conf_int().loc['eligible_x_post', 0],
    'ci_high': model4.conf_int().loc['eligible_x_post', 1],
    'rsq': model4.rsquared,
    'n': int(model4.nobs)
}

# 7e. Full model with state fixed effects
print("\n7e. Model 5: Full model with state + year fixed effects")
print("-" * 50)

# Use C() for categorical state fixed effects
formula5 = 'fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + male + married + ' + ' + '.join(educ_cols) + ' + C(YEAR) + C(STATEFIP)'
model5 = smf.ols(formula5, data=df).fit()

print(f"DiD Coefficient: {model5.params['eligible_x_post']:.4f}")
print(f"Std Error: {model5.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model5.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model5.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['eligible_x_post', 0]:.4f}, {model5.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

results_dict['model5'] = {
    'coef': model5.params['eligible_x_post'],
    'se': model5.bse['eligible_x_post'],
    'tstat': model5.tvalues['eligible_x_post'],
    'pvalue': model5.pvalues['eligible_x_post'],
    'ci_low': model5.conf_int().loc['eligible_x_post', 0],
    'ci_high': model5.conf_int().loc['eligible_x_post', 1],
    'rsq': model5.rsquared,
    'n': int(model5.nobs)
}

# =============================================================================
# 8. Robust Standard Errors (Clustered by State)
# =============================================================================
print("\n" + "=" * 80)
print("8. PREFERRED SPECIFICATION WITH CLUSTERED STANDARD ERRORS")
print("=" * 80)

# Fit model with clustered standard errors
model_robust = smf.ols(formula5, data=df).fit(cov_type='cluster',
                                               cov_kwds={'groups': df['STATEFIP']})

print(f"\nPreferred Model: Full specification with state-clustered SEs")
print("-" * 50)
print(f"DiD Coefficient: {model_robust.params['eligible_x_post']:.4f}")
print(f"Clustered Std Error: {model_robust.bse['eligible_x_post']:.4f}")
print(f"t-statistic: {model_robust.tvalues['eligible_x_post']:.2f}")
print(f"p-value: {model_robust.pvalues['eligible_x_post']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['eligible_x_post', 0]:.4f}, {model_robust.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"R-squared: {model_robust.rsquared:.4f}")
print(f"N: {int(model_robust.nobs):,}")

results_dict['preferred'] = {
    'coef': model_robust.params['eligible_x_post'],
    'se': model_robust.bse['eligible_x_post'],
    'tstat': model_robust.tvalues['eligible_x_post'],
    'pvalue': model_robust.pvalues['eligible_x_post'],
    'ci_low': model_robust.conf_int().loc['eligible_x_post', 0],
    'ci_high': model_robust.conf_int().loc['eligible_x_post', 1],
    'rsq': model_robust.rsquared,
    'n': int(model_robust.nobs)
}

# =============================================================================
# 9. Year-by-Year Effects (Event Study)
# =============================================================================
print("\n" + "=" * 80)
print("9. EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year interactions with eligibility (relative to 2011)
df['year_rel'] = df['YEAR'].copy()
years = sorted(df['YEAR'].unique())

print("\nYear-by-year effects (reference year: 2011)")
print("-" * 50)

event_results = {}
for yr in years:
    if yr != 2011:  # Reference year
        df[f'elig_x_{yr}'] = (df['daca_eligible'] * (df['YEAR'] == yr)).astype(int)

# Create formula for event study
event_vars = [f'elig_x_{yr}' for yr in years if yr != 2011]
formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(event_vars) + ' + AGE + age_sq + male + married + ' + ' + '.join(educ_cols) + ' + C(YEAR) + C(STATEFIP)'

model_event = smf.ols(formula_event, data=df).fit(cov_type='cluster',
                                                   cov_kwds={'groups': df['STATEFIP']})

print(f"{'Year':<8} {'Coef':>10} {'SE':>10} {'95% CI':>24} {'p-value':>10}")
print("-" * 62)
for yr in years:
    if yr == 2011:
        print(f"{yr:<8} {'0.0000':>10} {'(ref)':>10} {'':>24} {'':>10}")
    else:
        var = f'elig_x_{yr}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci = model_event.conf_int().loc[var]
        pval = model_event.pvalues[var]
        print(f"{yr:<8} {coef:>10.4f} {se:>10.4f} [{ci[0]:>8.4f}, {ci[1]:>8.4f}] {pval:>10.4f}")
        event_results[yr] = {'coef': coef, 'se': se, 'ci_low': ci[0], 'ci_high': ci[1], 'pvalue': pval}

results_dict['event_study'] = event_results

# =============================================================================
# 10. Heterogeneity Analysis
# =============================================================================
print("\n" + "=" * 80)
print("10. HETEROGENEITY ANALYSIS")
print("=" * 80)

# By gender
print("\n10a. By Gender")
print("-" * 50)

for sex, label in [(1, 'Male'), (2, 'Female')]:
    sub = df[df['SEX'] == sex].copy()
    model_sub = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + married + ' +
                        ' + '.join(educ_cols) + ' + C(YEAR) + C(STATEFIP)',
                        data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['STATEFIP']})
    print(f"{label}: coef={model_sub.params['eligible_x_post']:.4f}, se={model_sub.bse['eligible_x_post']:.4f}, n={int(model_sub.nobs):,}")
    results_dict[f'het_{label.lower()}'] = {
        'coef': model_sub.params['eligible_x_post'],
        'se': model_sub.bse['eligible_x_post'],
        'n': int(model_sub.nobs)
    }

# By education
print("\n10b. By Education Level")
print("-" * 50)

df['low_educ'] = (df['EDUC'] <= 6).astype(int)  # Less than high school
for educ, label in [(1, 'Less than HS'), (0, 'HS or more')]:
    sub = df[df['low_educ'] == educ].copy()
    model_sub = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + male + married + C(YEAR) + C(STATEFIP)',
                        data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['STATEFIP']})
    print(f"{label}: coef={model_sub.params['eligible_x_post']:.4f}, se={model_sub.bse['eligible_x_post']:.4f}, n={int(model_sub.nobs):,}")
    results_dict[f'het_{label.lower().replace(" ", "_").replace("/", "_")}'] = {
        'coef': model_sub.params['eligible_x_post'],
        'se': model_sub.bse['eligible_x_post'],
        'n': int(model_sub.nobs)
    }

# By age group
print("\n10c. By Age Group")
print("-" * 50)

df['age_group'] = pd.cut(df['AGE'], bins=[15, 24, 34, 64], labels=['16-24', '25-34', '35-64'])
for age_grp in ['16-24', '25-34', '35-64']:
    sub = df[df['age_group'] == age_grp].copy()
    if len(sub) > 1000:
        model_sub = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + male + married + C(YEAR) + C(STATEFIP)',
                            data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub['STATEFIP']})
        print(f"Age {age_grp}: coef={model_sub.params['eligible_x_post']:.4f}, se={model_sub.bse['eligible_x_post']:.4f}, n={int(model_sub.nobs):,}")
        results_dict[f'het_age_{age_grp}'] = {
            'coef': model_sub.params['eligible_x_post'],
            'se': model_sub.bse['eligible_x_post'],
            'n': int(model_sub.nobs)
        }

# =============================================================================
# 11. Summary Statistics for Report
# =============================================================================
print("\n" + "=" * 80)
print("11. SUMMARY STATISTICS FOR REPORT")
print("=" * 80)

# Sample sizes by year and eligibility
print("\n11a. Sample sizes by year and DACA eligibility")
print("-" * 50)
sample_by_year = df.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_by_year.columns = ['Not Eligible', 'Eligible']
print(sample_by_year)
results_dict['sample_by_year'] = sample_by_year.to_dict()

# Mean characteristics by eligibility
print("\n11b. Mean characteristics by DACA eligibility")
print("-" * 50)

char_vars = ['AGE', 'male', 'married', 'EDUC', 'fulltime']
char_stats = df.groupby('daca_eligible')[char_vars].mean()
char_stats.index = ['Not Eligible', 'Eligible']
print(char_stats.round(3))
results_dict['char_stats'] = char_stats.to_dict()

# =============================================================================
# 12. Save Results
# =============================================================================
print("\n" + "=" * 80)
print("12. SAVING RESULTS")
print("=" * 80)

# Save numerical results to JSON
with open('results_97.json', 'w') as f:
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    json.dump(convert_types(results_dict), f, indent=2)

print("Results saved to results_97.json")

# =============================================================================
# 13. PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE SUMMARY")
print("=" * 80)
print(f"""
Research Question: Effect of DACA eligibility on full-time employment
Sample: Hispanic-Mexican individuals born in Mexico, ages 16-64
Method: Difference-in-Differences with state and year fixed effects

PREFERRED ESTIMATE:
  Effect size: {results_dict['preferred']['coef']:.4f}
  Standard error (clustered by state): {results_dict['preferred']['se']:.4f}
  95% Confidence Interval: [{results_dict['preferred']['ci_low']:.4f}, {results_dict['preferred']['ci_high']:.4f}]
  Sample size: {results_dict['preferred']['n']:,}

Interpretation: DACA eligibility is associated with a {results_dict['preferred']['coef']*100:.2f} percentage point
{'increase' if results_dict['preferred']['coef'] > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if results_dict['preferred']['pvalue'] < 0.05 else 'not statistically significant'} at the 5% level (p = {results_dict['preferred']['pvalue']:.4f}).
""")

print("\nAnalysis complete!")

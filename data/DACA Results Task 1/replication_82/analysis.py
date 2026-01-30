"""
DACA Replication Study - Analysis Script
Replication ID: 82

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals (2013-2016)

Author: [Anonymous for replication]
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_82")

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD AND FILTER DATA
# =============================================================================
print("\n[1] Loading data...")

# Load data in chunks due to large size
chunksize = 1000000
chunks = []

# Define columns to use
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
           'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN',
           'YRIMMIG', 'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK']

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunksize):
    # Filter to Hispanic-Mexican born in Mexico
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"   Loaded {len(df):,} Hispanic-Mexican individuals born in Mexico")

# =============================================================================
# STEP 2: DEFINE DACA ELIGIBILITY
# =============================================================================
print("\n[2] Defining DACA eligibility criteria...")

# Filter to non-citizens only (CITIZEN == 3)
df_noncit = df[df['CITIZEN'] == 3].copy()
print(f"   Non-citizens: {len(df_noncit):,}")

# Filter to those who arrived by 2007 (continuous residence requirement)
# YRIMMIG == 0 means N/A (native-born), which shouldn't apply here
df_noncit = df_noncit[(df_noncit['YRIMMIG'] > 0) & (df_noncit['YRIMMIG'] <= 2007)]
print(f"   Arrived by 2007: {len(df_noncit):,}")

# Calculate age at immigration
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# Arrived before age 16
df_noncit['arrived_before_16'] = (df_noncit['age_at_immig'] < 16).astype(int)

# Under 31 as of June 15, 2012 (born after June 15, 1981)
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means: BIRTHYR > 1981, OR BIRTHYR == 1981 and BIRTHQTR >= 3
df_noncit['under_31_june_2012'] = (
    (df_noncit['BIRTHYR'] > 1981) |
    ((df_noncit['BIRTHYR'] == 1981) & (df_noncit['BIRTHQTR'] >= 3))
).astype(int)

# DACA eligibility: arrived before 16 AND under 31 in June 2012
df_noncit['daca_eligible'] = (
    (df_noncit['arrived_before_16'] == 1) &
    (df_noncit['under_31_june_2012'] == 1)
).astype(int)

print(f"   DACA eligible: {df_noncit['daca_eligible'].sum():,}")
print(f"   Not DACA eligible: {(df_noncit['daca_eligible'] == 0).sum():,}")

# =============================================================================
# STEP 3: RESTRICT SAMPLE FOR ANALYSIS
# =============================================================================
print("\n[3] Restricting sample for analysis...")

# Working-age adults (16-45)
df_analysis = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 45)].copy()
print(f"   Ages 16-45: {len(df_analysis):,}")

# Exclude 2012 (ambiguous treatment year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
print(f"   Excluding 2012: {len(df_analysis):,}")

# =============================================================================
# STEP 4: CREATE OUTCOME AND TREATMENT VARIABLES
# =============================================================================
print("\n[4] Creating analysis variables...")

# Post-DACA indicator (2013-2016)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Full-time employment: UHRSWORK >= 35
# Note: UHRSWORK == 0 could be not in labor force or unemployed
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Employed indicator (any employment)
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# Full-time among employed
df_analysis['fulltime_if_employed'] = np.where(
    df_analysis['EMPSTAT'] == 1,
    (df_analysis['UHRSWORK'] >= 35).astype(int),
    np.nan
)

# Interaction term
df_analysis['daca_post'] = df_analysis['daca_eligible'] * df_analysis['post']

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_coll'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

# =============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# =============================================================================
print("\n[5] Descriptive Statistics")
print("-" * 70)

# Summary by treatment group and period
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Treatment Group and Period:")
print(summary_stats)

# Save summary for report
summary_dict = {}
for daca in [0, 1]:
    for post in [0, 1]:
        try:
            subset = df_analysis[(df_analysis['daca_eligible'] == daca) & (df_analysis['post'] == post)]
            key = f"daca{daca}_post{post}"
            summary_dict[key] = {
                'n': len(subset),
                'n_weighted': subset['PERWT'].sum(),
                'fulltime_mean': subset['fulltime'].mean(),
                'fulltime_weighted': np.average(subset['fulltime'], weights=subset['PERWT']),
                'employed_mean': subset['employed'].mean(),
                'age_mean': subset['AGE'].mean(),
                'female_mean': subset['female'].mean(),
                'married_mean': subset['married'].mean(),
                'educ_hs_mean': subset['educ_hs'].mean()
            }
        except:
            pass

# =============================================================================
# STEP 6: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================
print("\n[6] Difference-in-Differences Estimation")
print("-" * 70)

# Simple DiD (no controls)
print("\nModel 1: Simple DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_post', data=df_analysis).fit()
print(f"   DiD coefficient (daca_post): {model1.params['daca_post']:.4f}")
print(f"   Std. Error: {model1.bse['daca_post']:.4f}")
print(f"   t-statistic: {model1.tvalues['daca_post']:.4f}")
print(f"   p-value: {model1.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['daca_post', 0]:.4f}, {model1.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   N: {int(model1.nobs):,}")

# DiD with individual controls
print("\nModel 2: DiD with individual controls")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + female + married + educ_hs + educ_coll',
                 data=df_analysis).fit()
print(f"   DiD coefficient (daca_post): {model2.params['daca_post']:.4f}")
print(f"   Std. Error: {model2.bse['daca_post']:.4f}")
print(f"   t-statistic: {model2.tvalues['daca_post']:.4f}")
print(f"   p-value: {model2.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['daca_post', 0]:.4f}, {model2.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   N: {int(model2.nobs):,}")

# DiD with state fixed effects
print("\nModel 3: DiD with controls + state fixed effects")
df_analysis['state_fe'] = pd.Categorical(df_analysis['STATEFIP'])
model3 = smf.ols('fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + female + married + educ_hs + educ_coll + C(STATEFIP)',
                 data=df_analysis).fit()
print(f"   DiD coefficient (daca_post): {model3.params['daca_post']:.4f}")
print(f"   Std. Error: {model3.bse['daca_post']:.4f}")
print(f"   t-statistic: {model3.tvalues['daca_post']:.4f}")
print(f"   p-value: {model3.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['daca_post', 0]:.4f}, {model3.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   N: {int(model3.nobs):,}")

# DiD with year fixed effects
print("\nModel 4: DiD with controls + state and year fixed effects")
model4 = smf.ols('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)',
                 data=df_analysis).fit()
print(f"   DiD coefficient (daca_post): {model4.params['daca_post']:.4f}")
print(f"   Std. Error: {model4.bse['daca_post']:.4f}")
print(f"   t-statistic: {model4.tvalues['daca_post']:.4f}")
print(f"   p-value: {model4.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['daca_post', 0]:.4f}, {model4.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   N: {int(model4.nobs):,}")

# =============================================================================
# STEP 7: WEIGHTED REGRESSION (PREFERRED SPECIFICATION)
# =============================================================================
print("\n[7] Weighted DiD Estimation (Population Weights)")
print("-" * 70)

# Preferred specification: Weighted regression with controls and fixed effects
print("\nModel 5 (PREFERRED): Weighted DiD with controls + state and year FE")
from statsmodels.regression.linear_model import WLS

# Prepare data for WLS
X_vars = ['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_coll']
df_reg = df_analysis.dropna(subset=['fulltime'] + X_vars + ['PERWT', 'YEAR', 'STATEFIP']).copy()

# Ensure numeric types
for col in X_vars:
    df_reg[col] = df_reg[col].astype(float)

# Add state and year dummies
state_dummies = pd.get_dummies(df_reg['STATEFIP'], prefix='state', drop_first=True).astype(float)
year_dummies = pd.get_dummies(df_reg['YEAR'], prefix='year', drop_first=True).astype(float)

X = pd.concat([df_reg[X_vars].reset_index(drop=True),
               state_dummies.reset_index(drop=True),
               year_dummies.reset_index(drop=True)], axis=1)
X = sm.add_constant(X)
y = df_reg['fulltime'].astype(float).reset_index(drop=True)
weights = df_reg['PERWT'].astype(float).reset_index(drop=True)

model5 = WLS(y, X, weights=weights).fit()
print(f"   DiD coefficient (daca_post): {model5.params['daca_post']:.4f}")
print(f"   Std. Error: {model5.bse['daca_post']:.4f}")
print(f"   t-statistic: {model5.tvalues['daca_post']:.4f}")
print(f"   p-value: {model5.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['daca_post', 0]:.4f}, {model5.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   Weighted N: {weights.sum():,.0f}")
print(f"   Observations: {len(df_reg):,}")

# =============================================================================
# STEP 8: ROBUST STANDARD ERRORS
# =============================================================================
print("\n[8] Model with Robust (HC1) Standard Errors")
print("-" * 70)

model5_robust = WLS(y, X, weights=weights).fit(cov_type='HC1')
print(f"   DiD coefficient (daca_post): {model5_robust.params['daca_post']:.4f}")
print(f"   Robust Std. Error: {model5_robust.bse['daca_post']:.4f}")
print(f"   t-statistic: {model5_robust.tvalues['daca_post']:.4f}")
print(f"   p-value: {model5_robust.pvalues['daca_post']:.4f}")
print(f"   95% CI: [{model5_robust.conf_int().loc['daca_post', 0]:.4f}, {model5_robust.conf_int().loc['daca_post', 1]:.4f}]")

# =============================================================================
# STEP 9: ROBUSTNESS CHECKS
# =============================================================================
print("\n[9] Robustness Checks")
print("-" * 70)

# Check 1: Include 2012 in pre-period
print("\nRobustness 1: Include 2012 in pre-period")
df_robust1 = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 45)].copy()
df_robust1['post'] = (df_robust1['YEAR'] >= 2013).astype(int)
df_robust1['fulltime'] = (df_robust1['UHRSWORK'] >= 35).astype(int)
df_robust1['daca_post'] = df_robust1['daca_eligible'] * df_robust1['post']
df_robust1['female'] = (df_robust1['SEX'] == 2).astype(int)
df_robust1['married'] = (df_robust1['MARST'].isin([1, 2])).astype(int)
df_robust1['age_sq'] = df_robust1['AGE'] ** 2
df_robust1['educ_hs'] = (df_robust1['EDUC'] >= 6).astype(int)
df_robust1['educ_coll'] = (df_robust1['EDUC'] >= 10).astype(int)

model_r1 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)',
                   data=df_robust1, weights=df_robust1['PERWT']).fit()
print(f"   DiD coefficient: {model_r1.params['daca_post']:.4f} (SE: {model_r1.bse['daca_post']:.4f})")

# Check 2: Employment as outcome (instead of full-time)
print("\nRobustness 2: Any employment as outcome")
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
df_reg2 = df_analysis.dropna(subset=['employed'] + X_vars + ['PERWT']).copy()
for col in X_vars:
    df_reg2[col] = df_reg2[col].astype(float)
y2 = df_reg2['employed'].astype(float).reset_index(drop=True)
X2 = pd.concat([df_reg2[X_vars].reset_index(drop=True).astype(float),
                pd.get_dummies(df_reg2['STATEFIP'], prefix='state', drop_first=True).astype(float).reset_index(drop=True),
                pd.get_dummies(df_reg2['YEAR'], prefix='year', drop_first=True).astype(float).reset_index(drop=True)], axis=1)
X2 = sm.add_constant(X2)
weights2 = df_reg2['PERWT'].astype(float).reset_index(drop=True)
model_r2 = WLS(y2, X2, weights=weights2).fit()
print(f"   DiD coefficient: {model_r2.params['daca_post']:.4f} (SE: {model_r2.bse['daca_post']:.4f})")

# Check 3: Restrict to younger ages (18-30)
print("\nRobustness 3: Restrict to ages 18-30")
df_young = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 30)].copy()
df_young['age_sq'] = df_young['AGE'] ** 2
model_r3 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)',
                   data=df_young, weights=df_young['PERWT']).fit()
print(f"   DiD coefficient: {model_r3.params['daca_post']:.4f} (SE: {model_r3.bse['daca_post']:.4f})")
print(f"   N: {len(df_young):,}")

# Check 4: Males only
print("\nRobustness 4: Males only")
df_male = df_analysis[df_analysis['female'] == 0].copy()
model_r4 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)',
                   data=df_male, weights=df_male['PERWT']).fit()
print(f"   DiD coefficient: {model_r4.params['daca_post']:.4f} (SE: {model_r4.bse['daca_post']:.4f})")
print(f"   N: {len(df_male):,}")

# Check 5: Females only
print("\nRobustness 5: Females only")
df_female = df_analysis[df_analysis['female'] == 1].copy()
model_r5 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)',
                   data=df_female, weights=df_female['PERWT']).fit()
print(f"   DiD coefficient: {model_r5.params['daca_post']:.4f} (SE: {model_r5.bse['daca_post']:.4f})")
print(f"   N: {len(df_female):,}")

# =============================================================================
# STEP 10: EVENT STUDY / PRE-TRENDS
# =============================================================================
print("\n[10] Event Study Analysis (Pre-trends check)")
print("-" * 70)

# Create year-specific interactions
df_event = df_analysis.copy()
for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df_event[f'daca_y{yr}'] = df_event['daca_eligible'] * (df_event['YEAR'] == yr).astype(int)

event_vars = [f'daca_y{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
event_formula = 'fulltime ~ daca_eligible + ' + ' + '.join(event_vars) + ' + AGE + age_sq + female + married + educ_hs + educ_coll + C(STATEFIP) + C(YEAR)'
model_event = smf.wls(event_formula, data=df_event, weights=df_event['PERWT']).fit()

print("Year-specific effects (relative to 2011):")
for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'daca_y{yr}'
    if var in model_event.params:
        print(f"   {yr}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f})")

# =============================================================================
# STEP 11: SAVE RESULTS
# =============================================================================
print("\n[11] Saving Results")
print("-" * 70)

# Create results dictionary
results = {
    'preferred_estimate': {
        'coefficient': float(model5_robust.params['daca_post']),
        'std_error': float(model5_robust.bse['daca_post']),
        'ci_lower': float(model5_robust.conf_int().loc['daca_post', 0]),
        'ci_upper': float(model5_robust.conf_int().loc['daca_post', 1]),
        'pvalue': float(model5_robust.pvalues['daca_post']),
        'n_obs': int(len(df_reg)),
        'n_weighted': float(weights.sum())
    },
    'models': {
        'model1_simple': {
            'coef': float(model1.params['daca_post']),
            'se': float(model1.bse['daca_post']),
            'n': int(model1.nobs)
        },
        'model2_controls': {
            'coef': float(model2.params['daca_post']),
            'se': float(model2.bse['daca_post']),
            'n': int(model2.nobs)
        },
        'model3_state_fe': {
            'coef': float(model3.params['daca_post']),
            'se': float(model3.bse['daca_post']),
            'n': int(model3.nobs)
        },
        'model4_twoway_fe': {
            'coef': float(model4.params['daca_post']),
            'se': float(model4.bse['daca_post']),
            'n': int(model4.nobs)
        },
        'model5_weighted': {
            'coef': float(model5.params['daca_post']),
            'se': float(model5.bse['daca_post']),
            'n': int(len(df_reg))
        }
    },
    'robustness': {
        'include_2012': float(model_r1.params['daca_post']),
        'employment_outcome': float(model_r2.params['daca_post']),
        'ages_18_30': float(model_r3.params['daca_post']),
        'males_only': float(model_r4.params['daca_post']),
        'females_only': float(model_r5.params['daca_post'])
    },
    'event_study': {yr: float(model_event.params[f'daca_y{yr}'])
                    for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
                    if f'daca_y{yr}' in model_event.params},
    'summary_stats': summary_dict
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results.json")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\nPreferred Estimate (Weighted DiD with robust SE):")
print(f"   Effect of DACA eligibility on full-time employment: {model5_robust.params['daca_post']:.4f}")
print(f"   Robust Standard Error: {model5_robust.bse['daca_post']:.4f}")
print(f"   95% Confidence Interval: [{model5_robust.conf_int().loc['daca_post', 0]:.4f}, {model5_robust.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   p-value: {model5_robust.pvalues['daca_post']:.4f}")
print(f"   Sample size: {len(df_reg):,}")
print(f"   Weighted population: {weights.sum():,.0f}")

# Interpretation
effect_pct = model5_robust.params['daca_post'] * 100
print(f"\nInterpretation:")
print(f"   DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if effect_pct > 0:
    print(f"   increase in the probability of full-time employment.")
else:
    print(f"   decrease in the probability of full-time employment.")

print("\nAnalysis complete!")

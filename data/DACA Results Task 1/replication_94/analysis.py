"""
DACA Replication Study - Analysis Script
Participant ID: 94

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations: {len(df):,}")

# ============================================================================
# 2. FILTER TO TARGET POPULATION
# ============================================================================
print("\n2. Filtering to target population...")
print("   - Hispanic-Mexican (HISPAN == 1)")
print("   - Born in Mexico (BPL == 200)")
print("   - Non-citizen (CITIZEN == 3)")

# Filter to Hispanic-Mexican born in Mexico
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"   After Hispanic-Mexican & Mexico-born filter: {len(df_mex):,}")

# Filter to non-citizens (as per instructions, assume non-citizens without papers are undocumented)
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter: {len(df_mex):,}")

# ============================================================================
# 3. CONSTRUCT DACA ELIGIBILITY VARIABLES
# ============================================================================
print("\n3. Constructing DACA eligibility variables...")

# Calculate age at arrival
# YRIMMIG = 0 means N/A, filter those out
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"   After filtering YRIMMIG > 0: {len(df_mex):,}")

df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# DACA eligibility criteria:
# 1. Arrived before age 16
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)

# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# Conservative: BIRTHYR >= 1982, or BIRTHYR = 1981 and BIRTHQTR >= 3 (July+)
df_mex['under_31_june2012'] = ((df_mex['BIRTHYR'] >= 1982) |
                               ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))).astype(int)

# 3. In US since at least June 15, 2007 (YRIMMIG <= 2007)
df_mex['in_us_since_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# Combined DACA eligibility
df_mex['daca_eligible'] = ((df_mex['arrived_before_16'] == 1) &
                            (df_mex['under_31_june2012'] == 1) &
                            (df_mex['in_us_since_2007'] == 1)).astype(int)

print(f"   DACA eligible: {df_mex['daca_eligible'].sum():,} ({df_mex['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# 4. CONSTRUCT TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n4. Constructing treatment and control groups...")

# Treatment: DACA eligible
# Control: Arrived at age 16+ but meet other age criteria (under 31 in 2012, in US since 2007)
# This provides a comparison group of similar immigrants who just missed the arrival age cutoff

df_mex['control_group'] = ((df_mex['arrived_before_16'] == 0) &
                            (df_mex['under_31_june2012'] == 1) &
                            (df_mex['in_us_since_2007'] == 1)).astype(int)

print(f"   Treatment (DACA eligible): {df_mex['daca_eligible'].sum():,}")
print(f"   Control (arrived age 16+, meet other criteria): {df_mex['control_group'].sum():,}")

# Create analysis sample: treatment or control
df_analysis = df_mex[(df_mex['daca_eligible'] == 1) | (df_mex['control_group'] == 1)].copy()
df_analysis['treated'] = df_analysis['daca_eligible']

print(f"   Analysis sample size: {len(df_analysis):,}")

# ============================================================================
# 5. DEFINE TIME PERIODS
# ============================================================================
print("\n5. Defining time periods...")

# Exclude 2012 (DACA implemented mid-year June 15)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_analysis):,}")

# Post period: 2013-2016
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"   Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"   Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# 6. CONSTRUCT OUTCOME VARIABLE
# ============================================================================
print("\n6. Constructing outcome variable...")

# Full-time employment: usually works 35+ hours per week
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"   Full-time employment rate: {df_analysis['fulltime'].mean()*100:.1f}%")

# ============================================================================
# 7. CREATE CONTROL VARIABLES
# ============================================================================
print("\n7. Creating control variables...")

# Age (will use age at survey)
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUCD'],
                                  bins=[0, 1, 50, 60, 100, 200],
                                  labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])

# Married indicator
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# State fixed effects (convert to string for dummy creation)
df_analysis['state'] = df_analysis['STATEFIP'].astype(str)

# Year fixed effects
df_analysis['year_str'] = df_analysis['YEAR'].astype(str)

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n8. Summary Statistics")
print("=" * 70)

# By treatment status
print("\nBy Treatment Status (Pre-period only):")
pre_data = df_analysis[df_analysis['post'] == 0]
for treat in [0, 1]:
    subset = pre_data[pre_data['treated'] == treat]
    label = "Treatment" if treat == 1 else "Control"
    print(f"\n{label} Group:")
    print(f"   N = {len(subset):,}")
    print(f"   Mean age: {subset['age'].mean():.1f}")
    print(f"   Female: {subset['female'].mean()*100:.1f}%")
    print(f"   Married: {subset['married'].mean()*100:.1f}%")
    print(f"   Full-time employed: {subset['fulltime'].mean()*100:.1f}%")
    print(f"   Mean age at arrival: {subset['age_at_arrival'].mean():.1f}")

# ============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("9. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Model 1: Basic DiD without controls
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['SERIAL']})
print(f"   DiD Estimate (treated_post): {model1.params['treated_post']:.4f}")
print(f"   Std Error: {model1.bse['treated_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model1.pvalues['treated_post']:.4f}")
print(f"   N: {model1.nobs:,.0f}")

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ treated + post + treated_post + age + age_sq + female + married + C(educ_cat)',
                  data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['SERIAL']})
print(f"   DiD Estimate (treated_post): {model2.params['treated_post']:.4f}")
print(f"   Std Error: {model2.bse['treated_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model2.pvalues['treated_post']:.4f}")
print(f"   N: {model2.nobs:,.0f}")

# Model 3: DiD with demographic controls + state FE
print("\nModel 3: DiD with demographic controls + state FE")
model3 = smf.ols('fulltime ~ treated + post + treated_post + age + age_sq + female + married + C(educ_cat) + C(state)',
                  data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['SERIAL']})
print(f"   DiD Estimate (treated_post): {model3.params['treated_post']:.4f}")
print(f"   Std Error: {model3.bse['treated_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['treated_post']:.4f}")
print(f"   N: {model3.nobs:,.0f}")

# Model 4: Full model with year FE (PREFERRED SPECIFICATION)
print("\nModel 4: Full model with year FE (PREFERRED SPECIFICATION)")
model4 = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                  data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['SERIAL']})
print(f"   DiD Estimate (treated_post): {model4.params['treated_post']:.4f}")
print(f"   Std Error: {model4.bse['treated_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['treated_post']:.4f}")
print(f"   N: {model4.nobs:,.0f}")
print(f"   R-squared: {model4.rsquared:.4f}")

# ============================================================================
# 10. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("10. ROBUSTNESS CHECKS")
print("=" * 70)

# 10.1 By gender
print("\n10.1 By Gender")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df_analysis[df_analysis['SEX'] == sex]
    model = smf.ols('fulltime ~ treated + treated_post + age + age_sq + married + C(educ_cat) + C(state) + C(year_str)',
                    data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['SERIAL']})
    print(f"   {label}: DiD = {model.params['treated_post']:.4f} (SE = {model.bse['treated_post']:.4f}), N = {model.nobs:,.0f}")

# 10.2 Placebo test with pre-period only
print("\n10.2 Placebo Test (Pre-period: 2006-2008 vs 2009-2011)")
pre_only = df_analysis[df_analysis['post'] == 0].copy()
pre_only['placebo_post'] = (pre_only['YEAR'] >= 2009).astype(int)
pre_only['treated_placebo'] = pre_only['treated'] * pre_only['placebo_post']
model_placebo = smf.ols('fulltime ~ treated + placebo_post + treated_placebo + age + age_sq + female + married + C(educ_cat) + C(state)',
                         data=pre_only).fit(cov_type='cluster', cov_kwds={'groups': pre_only['SERIAL']})
print(f"   Placebo DiD: {model_placebo.params['treated_placebo']:.4f} (SE = {model_placebo.bse['treated_placebo']:.4f})")
print(f"   p-value: {model_placebo.pvalues['treated_placebo']:.4f}")

# 10.3 Alternative control group - slightly older cohort
print("\n10.3 Alternative Control: Slightly older cohort (born 1979-1981)")
df_alt = df_mex[(df_mex['BIRTHYR'].between(1979, 1981)) &
                 (df_mex['arrived_before_16'] == 1) &
                 (df_mex['in_us_since_2007'] == 1) &
                 (df_mex['YEAR'] != 2012)].copy()
df_alt['post'] = (df_alt['YEAR'] >= 2013).astype(int)
df_alt['fulltime'] = (df_alt['UHRSWORK'] >= 35).astype(int)
# These are too old for DACA (over 31 in 2012), serve as control
df_alt['treated'] = 0

# Combine with treatment group
df_treat = df_analysis[df_analysis['treated'] == 1].copy()
df_alt_analysis = pd.concat([df_treat, df_alt], ignore_index=True)
df_alt_analysis['age'] = df_alt_analysis['AGE']
df_alt_analysis['age_sq'] = df_alt_analysis['age'] ** 2
df_alt_analysis['female'] = (df_alt_analysis['SEX'] == 2).astype(int)
df_alt_analysis['married'] = (df_alt_analysis['MARST'].isin([1, 2])).astype(int)
df_alt_analysis['state'] = df_alt_analysis['STATEFIP'].astype(str)
df_alt_analysis['year_str'] = df_alt_analysis['YEAR'].astype(str)
df_alt_analysis['educ_cat'] = pd.cut(df_alt_analysis['EDUCD'],
                                      bins=[0, 1, 50, 60, 100, 200],
                                      labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])
df_alt_analysis['treated_post'] = df_alt_analysis['treated'] * df_alt_analysis['post']

model_alt = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                     data=df_alt_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_alt_analysis['SERIAL']})
print(f"   Alt Control DiD: {model_alt.params['treated_post']:.4f} (SE = {model_alt.bse['treated_post']:.4f})")
print(f"   N: {model_alt.nobs:,.0f}")

# 10.4 Event study
print("\n10.4 Event Study Coefficients")
df_analysis['year_treat'] = df_analysis['treated'].astype(str) + '_' + df_analysis['YEAR'].astype(str)
# Create year dummies interacted with treatment (reference: 2011)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_yr_{yr}'] = (df_analysis['treated'] * (df_analysis['YEAR'] == yr)).astype(int)

event_formula = 'fulltime ~ treated + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str) + ' + \
                ' + '.join([f'treat_yr_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
model_event = smf.ols(event_formula, data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['SERIAL']})

print("   Year    Coefficient    Std Error")
print("   " + "-" * 35)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_yr_{yr}']
    se = model_event.bse[f'treat_yr_{yr}']
    sig = '*' if model_event.pvalues[f'treat_yr_{yr}'] < 0.05 else ''
    print(f"   {yr}      {coef:7.4f}       {se:7.4f}  {sig}")

# ============================================================================
# 11. WEIGHTED ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("11. WEIGHTED ANALYSIS (Using PERWT)")
print("=" * 70)

import statsmodels.api as sm

# Prepare data for weighted regression
X_vars = ['treated', 'treated_post', 'age', 'age_sq', 'female', 'married']
df_weighted = df_analysis.dropna(subset=['fulltime', 'PERWT'] + X_vars)

# Add dummies for state and year
state_dummies = pd.get_dummies(df_weighted['state'], prefix='state', drop_first=True)
year_dummies = pd.get_dummies(df_weighted['year_str'], prefix='year', drop_first=True)
educ_dummies = pd.get_dummies(df_weighted['educ_cat'], prefix='educ', drop_first=True)

X = pd.concat([df_weighted[X_vars], educ_dummies, state_dummies, year_dummies], axis=1)
X = sm.add_constant(X)
y = df_weighted['fulltime']
weights = df_weighted['PERWT']

model_weighted = sm.WLS(y, X, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_weighted['SERIAL']})
print(f"   Weighted DiD Estimate: {model_weighted.params['treated_post']:.4f}")
print(f"   Std Error: {model_weighted.bse['treated_post']:.4f}")
print(f"   95% CI: [{model_weighted.conf_int().loc['treated_post', 0]:.4f}, {model_weighted.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model_weighted.pvalues['treated_post']:.4f}")
print(f"   N: {model_weighted.nobs:,.0f}")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("12. FINAL SUMMARY - PREFERRED ESTIMATE")
print("=" * 70)

print(f"""
PREFERRED SPECIFICATION: Model 4 (Full model with year FE, unweighted)

Effect Size: {model4.params['treated_post']:.4f}
Standard Error: {model4.bse['treated_post']:.4f}
95% Confidence Interval: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
p-value: {model4.pvalues['treated_post']:.4f}
Sample Size: {model4.nobs:,.0f}
R-squared: {model4.rsquared:.4f}

Interpretation:
DACA eligibility is associated with a {abs(model4.params['treated_post'])*100:.2f} percentage point
{'increase' if model4.params['treated_post'] > 0 else 'decrease'} in full-time employment probability.
This effect is {'statistically significant' if model4.pvalues['treated_post'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

# Save key results to file
results_dict = {
    'effect_size': model4.params['treated_post'],
    'std_error': model4.bse['treated_post'],
    'ci_lower': model4.conf_int().loc['treated_post', 0],
    'ci_upper': model4.conf_int().loc['treated_post', 1],
    'p_value': model4.pvalues['treated_post'],
    'n_obs': int(model4.nobs),
    'r_squared': model4.rsquared,
    'weighted_effect': model_weighted.params['treated_post'],
    'weighted_se': model_weighted.bse['treated_post']
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\nResults saved to results_summary.json")
print("\nAnalysis complete.")

"""
DACA Replication Study - Revised Analysis Script
Participant ID: 94

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.

This revised analysis addresses concerns about comparability between treatment
and control groups by:
1. Restricting to working-age population (16-40)
2. Using a narrower bandwidth around the arrival age cutoff
3. Adding additional robustness checks
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - REVISED ANALYSIS")
print("=" * 70)

print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations: {len(df):,}")

# ============================================================================
# 2. FILTER TO TARGET POPULATION
# ============================================================================
print("\n2. Filtering to target population...")

# Filter to Hispanic-Mexican born in Mexico, non-citizens
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)].copy()
print(f"   Hispanic-Mexican, Mexico-born, non-citizen: {len(df_mex):,}")

# Filter to those with valid immigration year
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"   With valid YRIMMIG: {len(df_mex):,}")

# Restrict to working age (16-40) for better comparability
df_mex = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 40)].copy()
print(f"   Working age (16-40): {len(df_mex):,}")

# ============================================================================
# 3. CONSTRUCT ELIGIBILITY VARIABLES
# ============================================================================
print("\n3. Constructing eligibility variables...")

# Age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# DACA eligibility criteria
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)
df_mex['under_31_june2012'] = ((df_mex['BIRTHYR'] >= 1982) |
                               ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))).astype(int)
df_mex['in_us_since_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# Full DACA eligibility
df_mex['daca_eligible'] = ((df_mex['arrived_before_16'] == 1) &
                            (df_mex['under_31_june2012'] == 1) &
                            (df_mex['in_us_since_2007'] == 1)).astype(int)

print(f"   DACA eligible: {df_mex['daca_eligible'].sum():,}")

# ============================================================================
# 4. ANALYSIS APPROACH A: ARRIVAL AGE REGRESSION DISCONTINUITY
# ============================================================================
print("\n" + "=" * 70)
print("4. APPROACH A: ARRIVAL AGE BANDWIDTH ANALYSIS")
print("=" * 70)

# Use narrow bandwidth around age 16 cutoff for arrival
# Treatment: arrived at age 10-15 (just under cutoff)
# Control: arrived at age 16-21 (just over cutoff)
# Both groups must meet other criteria (under 31 in 2012, in US since 2007)

df_narrow = df_mex[(df_mex['under_31_june2012'] == 1) &
                   (df_mex['in_us_since_2007'] == 1) &
                   (df_mex['age_at_arrival'] >= 10) &
                   (df_mex['age_at_arrival'] <= 21)].copy()

df_narrow['treated'] = (df_narrow['age_at_arrival'] < 16).astype(int)
df_narrow = df_narrow[df_narrow['YEAR'] != 2012].copy()
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']

# Controls
df_narrow['age'] = df_narrow['AGE']
df_narrow['age_sq'] = df_narrow['age'] ** 2
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'].isin([1, 2])).astype(int)
df_narrow['state'] = df_narrow['STATEFIP'].astype(str)
df_narrow['year_str'] = df_narrow['YEAR'].astype(str)
df_narrow['educ_cat'] = pd.cut(df_narrow['EDUCD'],
                               bins=[0, 1, 50, 60, 100, 200],
                               labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])

print(f"\nNarrow Bandwidth Sample (arrival age 10-21):")
print(f"   Treatment (arrived 10-15): {(df_narrow['treated']==1).sum():,}")
print(f"   Control (arrived 16-21): {(df_narrow['treated']==0).sum():,}")
print(f"   Pre-period: {(df_narrow['post']==0).sum():,}")
print(f"   Post-period: {(df_narrow['post']==1).sum():,}")

# Summary stats
print("\nPre-period Summary (Narrow Bandwidth):")
pre_narrow = df_narrow[df_narrow['post'] == 0]
for t, label in [(1, 'Treatment (arr 10-15)'), (0, 'Control (arr 16-21)')]:
    s = pre_narrow[pre_narrow['treated'] == t]
    print(f"   {label}:")
    print(f"      N = {len(s):,}, Age = {s['age'].mean():.1f}, Female = {s['female'].mean()*100:.1f}%")
    print(f"      Full-time = {s['fulltime'].mean()*100:.1f}%, Arr age = {s['age_at_arrival'].mean():.1f}")

# Run DiD
model_narrow = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                        data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
print(f"\nDiD Result (Narrow Bandwidth):")
print(f"   Effect: {model_narrow.params['treated_post']:.4f}")
print(f"   SE: {model_narrow.bse['treated_post']:.4f}")
print(f"   95% CI: [{model_narrow.conf_int().loc['treated_post', 0]:.4f}, {model_narrow.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model_narrow.pvalues['treated_post']:.4f}")
print(f"   N: {model_narrow.nobs:,.0f}")

# ============================================================================
# 5. APPROACH B: AGE COHORT COMPARISON
# ============================================================================
print("\n" + "=" * 70)
print("5. APPROACH B: AGE COHORT COMPARISON")
print("=" * 70)

# Compare DACA-eligible young cohort to slightly older cohort
# that arrived before 16 but is too old for DACA (born 1977-1981, over 31 in 2012)
# Both groups arrived as children, so similar in that dimension

df_cohort = df_mex[(df_mex['arrived_before_16'] == 1) &
                   (df_mex['in_us_since_2007'] == 1)].copy()

# Treatment: born 1982-1995 (under 31 in 2012, DACA eligible if arrived <16)
# Control: born 1977-1981 (31-35 in 2012, too old for DACA)
df_cohort = df_cohort[df_cohort['BIRTHYR'].between(1977, 1995)].copy()
df_cohort['treated'] = (df_cohort['BIRTHYR'] >= 1982).astype(int)

df_cohort = df_cohort[df_cohort['YEAR'] != 2012].copy()
df_cohort['post'] = (df_cohort['YEAR'] >= 2013).astype(int)
df_cohort['fulltime'] = (df_cohort['UHRSWORK'] >= 35).astype(int)
df_cohort['treated_post'] = df_cohort['treated'] * df_cohort['post']

# Controls
df_cohort['age'] = df_cohort['AGE']
df_cohort['age_sq'] = df_cohort['age'] ** 2
df_cohort['female'] = (df_cohort['SEX'] == 2).astype(int)
df_cohort['married'] = (df_cohort['MARST'].isin([1, 2])).astype(int)
df_cohort['state'] = df_cohort['STATEFIP'].astype(str)
df_cohort['year_str'] = df_cohort['YEAR'].astype(str)
df_cohort['educ_cat'] = pd.cut(df_cohort['EDUCD'],
                               bins=[0, 1, 50, 60, 100, 200],
                               labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])

print(f"\nCohort Comparison Sample:")
print(f"   Treatment (born 1982-1995, DACA eligible): {(df_cohort['treated']==1).sum():,}")
print(f"   Control (born 1977-1981, too old): {(df_cohort['treated']==0).sum():,}")

# Summary stats
print("\nPre-period Summary (Cohort Comparison):")
pre_cohort = df_cohort[df_cohort['post'] == 0]
for t, label in [(1, 'Treatment (born 82-95)'), (0, 'Control (born 77-81)')]:
    s = pre_cohort[pre_cohort['treated'] == t]
    print(f"   {label}:")
    print(f"      N = {len(s):,}, Age = {s['age'].mean():.1f}, Female = {s['female'].mean()*100:.1f}%")
    print(f"      Full-time = {s['fulltime'].mean()*100:.1f}%, Birth year = {s['BIRTHYR'].mean():.1f}")

# Run DiD
model_cohort = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                        data=df_cohort).fit(cov_type='cluster', cov_kwds={'groups': df_cohort['SERIAL']})
print(f"\nDiD Result (Cohort Comparison):")
print(f"   Effect: {model_cohort.params['treated_post']:.4f}")
print(f"   SE: {model_cohort.bse['treated_post']:.4f}")
print(f"   95% CI: [{model_cohort.conf_int().loc['treated_post', 0]:.4f}, {model_cohort.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {model_cohort.pvalues['treated_post']:.4f}")
print(f"   N: {model_cohort.nobs:,.0f}")

# ============================================================================
# 6. EVENT STUDY FOR NARROW BANDWIDTH
# ============================================================================
print("\n" + "=" * 70)
print("6. EVENT STUDY (NARROW BANDWIDTH)")
print("=" * 70)

for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_narrow[f'treat_yr_{yr}'] = (df_narrow['treated'] * (df_narrow['YEAR'] == yr)).astype(int)

event_formula = 'fulltime ~ treated + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str) + ' + \
                ' + '.join([f'treat_yr_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
model_event = smf.ols(event_formula, data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})

print("   Year    Coefficient    Std Error    p-value")
print("   " + "-" * 45)
event_results = []
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_yr_{yr}']
    se = model_event.bse[f'treat_yr_{yr}']
    pval = model_event.pvalues[f'treat_yr_{yr}']
    sig = '*' if pval < 0.05 else ''
    print(f"   {yr}      {coef:8.4f}      {se:8.4f}     {pval:.4f}  {sig}")
    event_results.append({'year': yr, 'coef': coef, 'se': se, 'pval': pval})

# ============================================================================
# 7. TRIPLE DIFFERENCES (DDD)
# ============================================================================
print("\n" + "=" * 70)
print("7. TRIPLE DIFFERENCES (Citizenship Status)")
print("=" * 70)

# Use naturalized citizens as additional control (they don't need DACA)
df_citizen = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) &
                (df['CITIZEN'].isin([2, 3])) &  # Naturalized (2) or non-citizen (3)
                (df['YRIMMIG'] > 0) &
                (df['AGE'] >= 16) & (df['AGE'] <= 40)].copy()

df_citizen['age_at_arrival'] = df_citizen['YRIMMIG'] - df_citizen['BIRTHYR']
df_citizen = df_citizen[(df_citizen['age_at_arrival'] >= 0) & (df_citizen['age_at_arrival'] < 16)].copy()
df_citizen = df_citizen[df_citizen['BIRTHYR'] >= 1982].copy()  # Under 31 in 2012
df_citizen = df_citizen[df_citizen['YRIMMIG'] <= 2007].copy()  # In US since 2007
df_citizen = df_citizen[df_citizen['YEAR'] != 2012].copy()

df_citizen['noncitizen'] = (df_citizen['CITIZEN'] == 3).astype(int)
df_citizen['post'] = (df_citizen['YEAR'] >= 2013).astype(int)
df_citizen['fulltime'] = (df_citizen['UHRSWORK'] >= 35).astype(int)
df_citizen['noncit_post'] = df_citizen['noncitizen'] * df_citizen['post']

# Controls
df_citizen['age'] = df_citizen['AGE']
df_citizen['age_sq'] = df_citizen['age'] ** 2
df_citizen['female'] = (df_citizen['SEX'] == 2).astype(int)
df_citizen['married'] = (df_citizen['MARST'].isin([1, 2])).astype(int)
df_citizen['state'] = df_citizen['STATEFIP'].astype(str)
df_citizen['year_str'] = df_citizen['YEAR'].astype(str)
df_citizen['educ_cat'] = pd.cut(df_citizen['EDUCD'],
                                bins=[0, 1, 50, 60, 100, 200],
                                labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])

print(f"Sample: DACA-eligible age arriving <16, comparing non-citizens to naturalized")
print(f"   Non-citizens: {(df_citizen['noncitizen']==1).sum():,}")
print(f"   Naturalized: {(df_citizen['noncitizen']==0).sum():,}")

model_ddd = smf.ols('fulltime ~ noncitizen + post + noncit_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                     data=df_citizen).fit(cov_type='cluster', cov_kwds={'groups': df_citizen['SERIAL']})
print(f"\nDDD Result (Non-citizen vs Naturalized):")
print(f"   Effect: {model_ddd.params['noncit_post']:.4f}")
print(f"   SE: {model_ddd.bse['noncit_post']:.4f}")
print(f"   95% CI: [{model_ddd.conf_int().loc['noncit_post', 0]:.4f}, {model_ddd.conf_int().loc['noncit_post', 1]:.4f}]")
print(f"   p-value: {model_ddd.pvalues['noncit_post']:.4f}")
print(f"   N: {model_ddd.nobs:,.0f}")

# ============================================================================
# 8. ROBUSTNESS: DIFFERENT BANDWIDTHS
# ============================================================================
print("\n" + "=" * 70)
print("8. ROBUSTNESS: DIFFERENT ARRIVAL AGE BANDWIDTHS")
print("=" * 70)

for bw in [(12, 19), (10, 21), (8, 23), (5, 26)]:
    low, high = bw
    df_bw = df_mex[(df_mex['under_31_june2012'] == 1) &
                   (df_mex['in_us_since_2007'] == 1) &
                   (df_mex['age_at_arrival'] >= low) &
                   (df_mex['age_at_arrival'] <= high)].copy()
    df_bw['treated'] = (df_bw['age_at_arrival'] < 16).astype(int)
    df_bw = df_bw[df_bw['YEAR'] != 2012].copy()
    df_bw['post'] = (df_bw['YEAR'] >= 2013).astype(int)
    df_bw['fulltime'] = (df_bw['UHRSWORK'] >= 35).astype(int)
    df_bw['treated_post'] = df_bw['treated'] * df_bw['post']
    df_bw['age'] = df_bw['AGE']
    df_bw['age_sq'] = df_bw['age'] ** 2
    df_bw['female'] = (df_bw['SEX'] == 2).astype(int)
    df_bw['married'] = (df_bw['MARST'].isin([1, 2])).astype(int)
    df_bw['state'] = df_bw['STATEFIP'].astype(str)
    df_bw['year_str'] = df_bw['YEAR'].astype(str)
    df_bw['educ_cat'] = pd.cut(df_bw['EDUCD'],
                               bins=[0, 1, 50, 60, 100, 200],
                               labels=['none', 'less_hs', 'hs', 'some_college', 'college_plus'])

    model_bw = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(educ_cat) + C(state) + C(year_str)',
                        data=df_bw).fit(cov_type='cluster', cov_kwds={'groups': df_bw['SERIAL']})
    print(f"   Bandwidth [{low}, {high}]: Effect = {model_bw.params['treated_post']:.4f} (SE = {model_bw.bse['treated_post']:.4f}), N = {model_bw.nobs:,.0f}")

# ============================================================================
# 9. SUBGROUP ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("9. SUBGROUP ANALYSIS (NARROW BANDWIDTH)")
print("=" * 70)

# By gender
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df_narrow[df_narrow['SEX'] == sex]
    model = smf.ols('fulltime ~ treated + treated_post + age + age_sq + married + C(educ_cat) + C(state) + C(year_str)',
                    data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['SERIAL']})
    print(f"   {label}: Effect = {model.params['treated_post']:.4f} (SE = {model.bse['treated_post']:.4f}), N = {model.nobs:,.0f}")

# By education
print("\nBy Education Level:")
for educ_val, label in [(1, 'Less than HS'), (2, 'HS or more')]:
    if educ_val == 1:
        subset = df_narrow[df_narrow['EDUCD'] < 60]
    else:
        subset = df_narrow[df_narrow['EDUCD'] >= 60]
    model = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + C(state) + C(year_str)',
                    data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['SERIAL']})
    print(f"   {label}: Effect = {model.params['treated_post']:.4f} (SE = {model.bse['treated_post']:.4f}), N = {model.nobs:,.0f}")

# ============================================================================
# 10. FINAL PREFERRED ESTIMATE
# ============================================================================
print("\n" + "=" * 70)
print("10. FINAL PREFERRED ESTIMATE")
print("=" * 70)

# Use narrow bandwidth as preferred specification
preferred_effect = model_narrow.params['treated_post']
preferred_se = model_narrow.bse['treated_post']
preferred_ci = model_narrow.conf_int().loc['treated_post']
preferred_pval = model_narrow.pvalues['treated_post']
preferred_n = int(model_narrow.nobs)

print(f"""
PREFERRED SPECIFICATION: Narrow Bandwidth DiD (arrival age 10-21)

Effect Size: {preferred_effect:.4f}
Standard Error: {preferred_se:.4f}
95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]
p-value: {preferred_pval:.4f}
Sample Size: {preferred_n:,}

Interpretation:
DACA eligibility is associated with a {preferred_effect*100:.2f} percentage point
{'increase' if preferred_effect > 0 else 'decrease'} in the probability of full-time employment
(working 35+ hours per week). This effect is
{'statistically significant' if preferred_pval < 0.05 else 'not statistically significant'} at the 5% level.

The estimate compares Mexican-born non-citizens who arrived at ages 10-15
(DACA eligible) to those who arrived at ages 16-21 (just above the age-at-arrival
cutoff), before and after DACA implementation. Both groups meet other DACA
criteria (under 31 in 2012, in US since 2007).
""")

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================
results = {
    'preferred_estimate': {
        'specification': 'Narrow Bandwidth DiD (arrival age 10-21)',
        'effect_size': float(preferred_effect),
        'std_error': float(preferred_se),
        'ci_lower': float(preferred_ci[0]),
        'ci_upper': float(preferred_ci[1]),
        'p_value': float(preferred_pval),
        'n_obs': preferred_n
    },
    'cohort_comparison': {
        'effect_size': float(model_cohort.params['treated_post']),
        'std_error': float(model_cohort.bse['treated_post']),
        'n_obs': int(model_cohort.nobs)
    },
    'triple_diff': {
        'effect_size': float(model_ddd.params['noncit_post']),
        'std_error': float(model_ddd.bse['noncit_post']),
        'n_obs': int(model_ddd.nobs)
    },
    'event_study': event_results
}

with open('results_revised.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results_revised.json")
print("\nAnalysis complete.")

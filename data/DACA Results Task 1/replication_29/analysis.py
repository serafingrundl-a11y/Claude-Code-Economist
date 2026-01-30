"""
DACA Replication Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US (2006-2016)

Author: Replication Study
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
os.chdir(r'C:\Users\seraf\DACA Results Task 1\replication_29')

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)
print()

# =============================================================================
# PART 1: DATA LOADING AND SAMPLE SELECTION
# =============================================================================

print("PART 1: Loading and filtering data...")
print("-"*60)

# DACA eligibility criteria (from instructions):
# 1. Arrived unlawfully in the US before their 16th birthday
# 2. Had not yet had their 31st birthday as of June 15, 2012
# 3. Lived continuously in the US since June 15, 2007
# 4. Were present in the US on June 15, 2012 and did not have lawful status

# Key variable codes from data dictionary:
# HISPAN = 1 for Mexican ethnicity
# BPL = 200 for Mexico (birthplace)
# CITIZEN = 3 for "Not a citizen" (proxy for undocumented)
# UHRSWORK >= 35 for full-time employment
# YEAR: 2006-2016

# Read data in chunks due to large file size
chunk_size = 500000
data_path = 'data/data.csv'

# First pass: filter to relevant population
print("Loading and filtering data (Hispanic-Mexican, born in Mexico)...")

chunks = []
for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
    # Filter to Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)

print(f"Initial sample (Hispanic-Mexican, born in Mexico): {len(df):,} observations")
print(f"Years covered: {df['YEAR'].min()} to {df['YEAR'].max()}")

# =============================================================================
# PART 2: DEFINE DACA ELIGIBILITY
# =============================================================================

print()
print("PART 2: Defining DACA eligibility criteria...")
print("-"*60)

# Calculate age at DACA (June 15, 2012)
# Using midpoint of birth year for approximation
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Calculate age at immigration
# YRIMMIG contains year of immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Criteria for DACA eligibility:
# 1. Arrived before 16th birthday: age_at_immig < 16
# 2. Under 31 on June 15, 2012: age_at_daca < 31 (born after June 15, 1981)
#    To be conservative, BIRTHYR >= 1982
# 3. In US since June 15, 2007: YRIMMIG <= 2007 (been in US at least 5 years by 2012)
# 4. Not a citizen (proxy for undocumented): CITIZEN == 3

# Handle missing/invalid values
df['YRIMMIG_valid'] = df['YRIMMIG'].replace(0, np.nan)
df['age_at_immig'] = df['YRIMMIG_valid'] - df['BIRTHYR']

# Create DACA eligibility indicator based on observable characteristics
# Note: We can only observe proxies for true eligibility

# Criterion 1: Arrived before age 16
df['arrived_before_16'] = (df['age_at_immig'] < 16) & (df['age_at_immig'] >= 0)

# Criterion 2: Under 31 on June 15, 2012 (born >= 1982 to be safe)
df['under_31_at_daca'] = df['BIRTHYR'] >= 1982

# Criterion 3: In US since 2007 (arrived 2007 or earlier)
df['in_us_since_2007'] = df['YRIMMIG_valid'] <= 2007

# Criterion 4: Not a citizen (proxy for undocumented)
df['non_citizen'] = df['CITIZEN'] == 3

# Combined DACA eligibility (treatment group)
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_at_daca'] &
    df['in_us_since_2007'] &
    df['non_citizen']
)

print(f"Observations meeting each criterion:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Under 31 at DACA (2012): {df['under_31_at_daca'].sum():,}")
print(f"  In US since 2007: {df['in_us_since_2007'].sum():,}")
print(f"  Non-citizen: {df['non_citizen'].sum():,}")
print(f"  DACA eligible (all criteria): {df['daca_eligible'].sum():,}")

# =============================================================================
# PART 3: CREATE ANALYSIS SAMPLE AND VARIABLES
# =============================================================================

print()
print("PART 3: Creating analysis sample and outcome variables...")
print("-"*60)

# Define full-time employment (usually working 35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create post-DACA indicator (DACA implemented June 15, 2012)
# Since ACS doesn't have month, 2012 is ambiguous
# Conservative approach: treat 2013+ as post-DACA
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Also create version where 2012 is excluded for robustness
df['year_2012'] = (df['YEAR'] == 2012).astype(int)

# Create treatment interaction
df['daca_x_post'] = df['daca_eligible'].astype(int) * df['post']

# Restrict to working-age population (16-64)
# Also applying age restrictions relevant to DACA-eligible comparison
df_analysis = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()

print(f"Working-age sample (16-64): {len(df_analysis):,} observations")
print(f"  DACA eligible: {df_analysis['daca_eligible'].sum():,}")
print(f"  Not DACA eligible: {(~df_analysis['daca_eligible']).sum():,}")

# Further restrict to non-citizens for cleaner comparison
# Compare DACA-eligible non-citizens to non-eligible non-citizens
df_noncit = df_analysis[df_analysis['non_citizen']].copy()

print(f"\nNon-citizen sample: {len(df_noncit):,} observations")
print(f"  DACA eligible: {df_noncit['daca_eligible'].sum():,}")
print(f"  Not DACA eligible (comparison): {(~df_noncit['daca_eligible']).sum():,}")

# =============================================================================
# PART 4: DESCRIPTIVE STATISTICS
# =============================================================================

print()
print("PART 4: Descriptive Statistics...")
print("-"*60)

# Summary statistics by DACA eligibility and pre/post period
def summarize_group(data, name):
    """Calculate summary statistics for a group."""
    n = len(data)
    if n == 0:
        return None
    return {
        'Group': name,
        'N': n,
        'Full-time Emp Rate': data['fulltime'].mean() * 100,
        'Age': data['AGE'].mean(),
        'Female': (data['SEX'] == 2).mean() * 100,
        'Married': (data['MARST'].isin([1, 2])).mean() * 100,
        'Years in US': data['YRSUSA1'].mean(),
        'HS or More': (data['EDUCD'] >= 62).mean() * 100,
    }

# Pre-period (2006-2011)
pre_eligible = df_noncit[(df_noncit['daca_eligible']) & (df_noncit['YEAR'] < 2012)]
pre_ineligible = df_noncit[(~df_noncit['daca_eligible']) & (df_noncit['YEAR'] < 2012)]

# Post-period (2013-2016)
post_eligible = df_noncit[(df_noncit['daca_eligible']) & (df_noncit['YEAR'] >= 2013)]
post_ineligible = df_noncit[(~df_noncit['daca_eligible']) & (df_noncit['YEAR'] >= 2013)]

desc_stats = pd.DataFrame([
    summarize_group(pre_eligible, 'Pre-DACA, Eligible'),
    summarize_group(pre_ineligible, 'Pre-DACA, Ineligible'),
    summarize_group(post_eligible, 'Post-DACA, Eligible'),
    summarize_group(post_ineligible, 'Post-DACA, Ineligible'),
])

print("\nDescriptive Statistics by Group:")
print(desc_stats.to_string(index=False))

# Calculate simple DiD
pre_diff = pre_eligible['fulltime'].mean() - pre_ineligible['fulltime'].mean()
post_diff = post_eligible['fulltime'].mean() - post_ineligible['fulltime'].mean()
simple_did = post_diff - pre_diff

print(f"\nSimple Difference-in-Differences (Full-time Employment Rate):")
print(f"  Pre-period difference (Eligible - Ineligible): {pre_diff*100:.2f} pp")
print(f"  Post-period difference (Eligible - Ineligible): {post_diff*100:.2f} pp")
print(f"  DiD estimate: {simple_did*100:.2f} percentage points")

# =============================================================================
# PART 5: DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================

print()
print("PART 5: Difference-in-Differences Regression Analysis...")
print("-"*60)

# Prepare variables for regression
df_reg = df_noncit.copy()
df_reg['daca'] = df_reg['daca_eligible'].astype(int)
df_reg['female'] = (df_reg['SEX'] == 2).astype(int)
df_reg['married'] = df_reg['MARST'].isin([1, 2]).astype(int)
df_reg['hs_or_more'] = (df_reg['EDUCD'] >= 62).astype(int)
df_reg['age_sq'] = df_reg['AGE'] ** 2

# Exclude 2012 (ambiguous treatment year)
df_reg_no2012 = df_reg[df_reg['YEAR'] != 2012].copy()

print(f"Regression sample (excluding 2012): {len(df_reg_no2012):,}")

# Model 1: Basic DiD
print("\nModel 1: Basic Difference-in-Differences")
model1 = smf.ols('fulltime ~ daca + post + daca:post', data=df_reg_no2012).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg_no2012['STATEFIP']}
)
print(f"  DiD coefficient (daca:post): {model1.params['daca:post']:.4f}")
print(f"  Standard error: {model1.bse['daca:post']:.4f}")
print(f"  95% CI: [{model1.conf_int().loc['daca:post', 0]:.4f}, {model1.conf_int().loc['daca:post', 1]:.4f}]")
print(f"  p-value: {model1.pvalues['daca:post']:.4f}")

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with Demographic Controls")
model2 = smf.ols('fulltime ~ daca + post + daca:post + AGE + age_sq + female + married + hs_or_more',
                 data=df_reg_no2012).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg_no2012['STATEFIP']}
)
print(f"  DiD coefficient (daca:post): {model2.params['daca:post']:.4f}")
print(f"  Standard error: {model2.bse['daca:post']:.4f}")
print(f"  95% CI: [{model2.conf_int().loc['daca:post', 0]:.4f}, {model2.conf_int().loc['daca:post', 1]:.4f}]")
print(f"  p-value: {model2.pvalues['daca:post']:.4f}")

# Model 3: DiD with year fixed effects
print("\nModel 3: DiD with Year Fixed Effects")
df_reg_no2012['year_factor'] = pd.Categorical(df_reg_no2012['YEAR'])
model3 = smf.ols('fulltime ~ daca + C(year_factor) + daca:post + AGE + age_sq + female + married + hs_or_more',
                 data=df_reg_no2012).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg_no2012['STATEFIP']}
)
print(f"  DiD coefficient (daca:post): {model3.params['daca:post']:.4f}")
print(f"  Standard error: {model3.bse['daca:post']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['daca:post', 0]:.4f}, {model3.conf_int().loc['daca:post', 1]:.4f}]")
print(f"  p-value: {model3.pvalues['daca:post']:.4f}")

# Model 4: DiD with state fixed effects
print("\nModel 4: DiD with State and Year Fixed Effects (Preferred Specification)")
df_reg_no2012['state_factor'] = pd.Categorical(df_reg_no2012['STATEFIP'])
model4 = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca:post + AGE + age_sq + female + married + hs_or_more',
                 data=df_reg_no2012).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg_no2012['STATEFIP']}
)
print(f"  DiD coefficient (daca:post): {model4.params['daca:post']:.4f}")
print(f"  Standard error: {model4.bse['daca:post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['daca:post', 0]:.4f}, {model4.conf_int().loc['daca:post', 1]:.4f}]")
print(f"  p-value: {model4.pvalues['daca:post']:.4f}")
print(f"  R-squared: {model4.rsquared:.4f}")
print(f"  N: {int(model4.nobs):,}")

# =============================================================================
# PART 6: ROBUSTNESS CHECKS
# =============================================================================

print()
print("PART 6: Robustness Checks...")
print("-"*60)

# Robustness 1: Include 2012
print("\nRobustness 1: Including 2012 (with post=1 for 2012)")
df_reg_with2012 = df_reg.copy()
df_reg_with2012['post_inc2012'] = (df_reg_with2012['YEAR'] >= 2012).astype(int)
df_reg_with2012['daca_x_post_inc2012'] = df_reg_with2012['daca'] * df_reg_with2012['post_inc2012']
df_reg_with2012['year_factor'] = pd.Categorical(df_reg_with2012['YEAR'])
df_reg_with2012['state_factor'] = pd.Categorical(df_reg_with2012['STATEFIP'])

model_r1 = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca_x_post_inc2012 + AGE + age_sq + female + married + hs_or_more',
                   data=df_reg_with2012).fit(
    cov_type='cluster', cov_kwds={'groups': df_reg_with2012['STATEFIP']}
)
print(f"  DiD coefficient: {model_r1.params['daca_x_post_inc2012']:.4f} (SE: {model_r1.bse['daca_x_post_inc2012']:.4f})")

# Robustness 2: Restrict to ages 18-30 (core DACA-eligible ages)
print("\nRobustness 2: Core DACA age range (18-30)")
df_core_age = df_reg_no2012[(df_reg_no2012['AGE'] >= 18) & (df_reg_no2012['AGE'] <= 30)].copy()
df_core_age['year_factor'] = pd.Categorical(df_core_age['YEAR'])
df_core_age['state_factor'] = pd.Categorical(df_core_age['STATEFIP'])

model_r2 = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca:post + AGE + age_sq + female + married + hs_or_more',
                   data=df_core_age).fit(
    cov_type='cluster', cov_kwds={'groups': df_core_age['STATEFIP']}
)
print(f"  DiD coefficient: {model_r2.params['daca:post']:.4f} (SE: {model_r2.bse['daca:post']:.4f})")
print(f"  N: {int(model_r2.nobs):,}")

# Robustness 3: Alternative control group - all non-citizen immigrants (not just Mexican)
# This would require loading more data, so we'll note this as a limitation

# Robustness 4: Placebo test - check for pre-trends
print("\nRobustness 3: Pre-trends test")
df_pre = df_reg[df_reg['YEAR'] < 2012].copy()
df_pre['year_trend'] = df_pre['YEAR'] - 2006
df_pre['daca_x_trend'] = df_pre['daca'] * df_pre['year_trend']
df_pre['year_factor'] = pd.Categorical(df_pre['YEAR'])
df_pre['state_factor'] = pd.Categorical(df_pre['STATEFIP'])

model_r3 = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca_x_trend + AGE + age_sq + female + married + hs_or_more',
                   data=df_pre).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']}
)
print(f"  Pre-trend coefficient (daca x year): {model_r3.params['daca_x_trend']:.4f} (SE: {model_r3.bse['daca_x_trend']:.4f})")
print(f"  p-value: {model_r3.pvalues['daca_x_trend']:.4f}")

# =============================================================================
# PART 7: EVENT STUDY
# =============================================================================

print()
print("PART 7: Event Study Analysis...")
print("-"*60)

# Create year-specific treatment effects (event study)
df_event = df_reg.copy()
df_event['year_factor'] = pd.Categorical(df_event['YEAR'])
df_event['state_factor'] = pd.Categorical(df_event['STATEFIP'])

# Create year dummies interacted with DACA eligibility
years = sorted(df_event['YEAR'].unique())
for year in years:
    if year != 2011:  # 2011 is reference year (last pre-treatment year)
        df_event[f'daca_x_{year}'] = (df_event['daca'] * (df_event['YEAR'] == year)).astype(int)

# Build formula
year_interactions = ' + '.join([f'daca_x_{year}' for year in years if year != 2011])
formula = f'fulltime ~ daca + C(year_factor) + C(state_factor) + {year_interactions} + AGE + age_sq + female + married + hs_or_more'

model_event = smf.ols(formula, data=df_event).fit(
    cov_type='cluster', cov_kwds={'groups': df_event['STATEFIP']}
)

print("Event study coefficients (relative to 2011):")
event_results = []
for year in years:
    if year != 2011:
        coef = model_event.params[f'daca_x_{year}']
        se = model_event.bse[f'daca_x_{year}']
        ci_low = model_event.conf_int().loc[f'daca_x_{year}', 0]
        ci_high = model_event.conf_int().loc[f'daca_x_{year}', 1]
        pval = model_event.pvalues[f'daca_x_{year}']
        print(f"  {year}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}], p={pval:.4f}")
        event_results.append({'year': year, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})
    else:
        event_results.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})

# =============================================================================
# PART 8: HETEROGENEOUS EFFECTS
# =============================================================================

print()
print("PART 8: Heterogeneity Analysis...")
print("-"*60)

# By gender
print("\nBy Gender:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_reg_no2012[df_reg_no2012['SEX'] == sex].copy()
    df_sub['year_factor'] = pd.Categorical(df_sub['YEAR'])
    df_sub['state_factor'] = pd.Categorical(df_sub['STATEFIP'])
    model_sub = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca:post + AGE + age_sq + married + hs_or_more',
                        data=df_sub).fit(
        cov_type='cluster', cov_kwds={'groups': df_sub['STATEFIP']}
    )
    print(f"  {label}: DiD = {model_sub.params['daca:post']:.4f} (SE: {model_sub.bse['daca:post']:.4f}), N = {int(model_sub.nobs):,}")

# By education
print("\nBy Education:")
for edu, label in [(0, 'Less than HS'), (1, 'HS or More')]:
    df_sub = df_reg_no2012[df_reg_no2012['hs_or_more'] == edu].copy()
    df_sub['year_factor'] = pd.Categorical(df_sub['YEAR'])
    df_sub['state_factor'] = pd.Categorical(df_sub['STATEFIP'])
    model_sub = smf.ols('fulltime ~ daca + C(year_factor) + C(state_factor) + daca:post + AGE + age_sq + female + married',
                        data=df_sub).fit(
        cov_type='cluster', cov_kwds={'groups': df_sub['STATEFIP']}
    )
    print(f"  {label}: DiD = {model_sub.params['daca:post']:.4f} (SE: {model_sub.bse['daca:post']:.4f}), N = {int(model_sub.nobs):,}")

# =============================================================================
# PART 9: SAVE RESULTS
# =============================================================================

print()
print("PART 9: Saving Results...")
print("-"*60)

# Create results dictionary
results = {
    'sample_size': int(model4.nobs),
    'preferred_estimate': {
        'coefficient': model4.params['daca:post'],
        'standard_error': model4.bse['daca:post'],
        'ci_lower': model4.conf_int().loc['daca:post', 0],
        'ci_upper': model4.conf_int().loc['daca:post', 1],
        'p_value': model4.pvalues['daca:post'],
        'r_squared': model4.rsquared,
    },
    'descriptive_stats': desc_stats.to_dict('records'),
    'event_study': event_results,
    'model_comparisons': {
        'model1_basic': {'coef': model1.params['daca:post'], 'se': model1.bse['daca:post']},
        'model2_controls': {'coef': model2.params['daca:post'], 'se': model2.bse['daca:post']},
        'model3_year_fe': {'coef': model3.params['daca:post'], 'se': model3.bse['daca:post']},
        'model4_full_fe': {'coef': model4.params['daca:post'], 'se': model4.bse['daca:post']},
    },
    'robustness': {
        'include_2012': {'coef': model_r1.params['daca_x_post_inc2012'], 'se': model_r1.bse['daca_x_post_inc2012']},
        'core_age_18_30': {'coef': model_r2.params['daca:post'], 'se': model_r2.bse['daca:post']},
        'pre_trend': {'coef': model_r3.params['daca_x_trend'], 'se': model_r3.bse['daca_x_trend'], 'pval': model_r3.pvalues['daca_x_trend']},
    },
    'pre_post_means': {
        'pre_eligible': pre_eligible['fulltime'].mean(),
        'pre_ineligible': pre_ineligible['fulltime'].mean(),
        'post_eligible': post_eligible['fulltime'].mean(),
        'post_ineligible': post_ineligible['fulltime'].mean(),
    }
}

# Save to JSON
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print("Results saved to analysis_results.json")

# Save descriptive stats table
desc_stats.to_csv('descriptive_statistics.csv', index=False)
print("Descriptive statistics saved to descriptive_statistics.csv")

# Save regression tables
with open('regression_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("REGRESSION RESULTS: EFFECT OF DACA ON FULL-TIME EMPLOYMENT\n")
    f.write("="*80 + "\n\n")

    f.write("Model 1: Basic DiD\n")
    f.write("-"*40 + "\n")
    f.write(model1.summary().as_text())
    f.write("\n\n")

    f.write("Model 4: Preferred Specification (State & Year FE)\n")
    f.write("-"*40 + "\n")
    # Write key results since full summary is very long
    f.write(f"DiD Coefficient (daca:post): {model4.params['daca:post']:.6f}\n")
    f.write(f"Standard Error: {model4.bse['daca:post']:.6f}\n")
    f.write(f"t-statistic: {model4.tvalues['daca:post']:.4f}\n")
    f.write(f"p-value: {model4.pvalues['daca:post']:.6f}\n")
    f.write(f"95% CI: [{model4.conf_int().loc['daca:post', 0]:.6f}, {model4.conf_int().loc['daca:post', 1]:.6f}]\n")
    f.write(f"R-squared: {model4.rsquared:.4f}\n")
    f.write(f"Observations: {int(model4.nobs):,}\n")

print("Regression results saved to regression_results.txt")

# Save event study results
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print()
print(f"PREFERRED ESTIMATE (Model 4 with State and Year Fixed Effects):")
print(f"  Effect of DACA eligibility on full-time employment: {model4.params['daca:post']*100:.2f} percentage points")
print(f"  Standard Error: {model4.bse['daca:post']*100:.2f} pp")
print(f"  95% Confidence Interval: [{model4.conf_int().loc['daca:post', 0]*100:.2f}, {model4.conf_int().loc['daca:post', 1]*100:.2f}] pp")
print(f"  Sample Size: {int(model4.nobs):,}")
print()

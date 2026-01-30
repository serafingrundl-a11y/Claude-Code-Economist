"""
DACA Replication Study - Analysis Script
=========================================
This script performs a difference-in-differences analysis to estimate
the causal effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals.

Author: Anonymous (Replication ID 71)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# -----------------------------------------------------------------------------
# STEP 1: Load and Filter Data
# -----------------------------------------------------------------------------
print("\n[Step 1] Loading data...")

# Define columns to load (to reduce memory usage)
cols_to_use = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'MARST', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK']

# Load data in chunks to handle large file
chunk_size = 1000000
chunks = []

print("Loading data in chunks...")
for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_to_use,
                                       chunksize=chunk_size, low_memory=False)):
    # Apply initial filters to reduce memory
    # Hispanic-Mexican ethnicity
    chunk = chunk[chunk['HISPAN'] == 1]
    # Born in Mexico
    chunk = chunk[chunk['BPL'] == 200]
    # Not a citizen (proxy for undocumented)
    chunk = chunk[chunk['CITIZEN'] == 3]
    # Years 2006-2016 (should already be filtered, but ensure)
    chunk = chunk[(chunk['YEAR'] >= 2006) & (chunk['YEAR'] <= 2016)]

    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"After initial filtering: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 2: Apply DACA Eligibility Criteria
# -----------------------------------------------------------------------------
print("\n[Step 2] Applying DACA eligibility criteria...")

# Remove invalid immigration year data
df = df[df['YRIMMIG'] > 0]
df = df[df['YRIMMIG'] != 996]  # Not reported

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA requirement: Arrived before 16th birthday
df = df[df['age_at_immig'] < 16]

# DACA requirement: Present in US since June 15, 2007
# Use YRIMMIG <= 2007 as proxy for continuous presence
df = df[df['YRIMMIG'] <= 2007]

print(f"After DACA eligibility filters: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 3: Define Treatment and Control Groups
# -----------------------------------------------------------------------------
print("\n[Step 3] Defining treatment and control groups...")

# Calculate age as of June 15, 2012 (DACA implementation)
# Using birth year and quarter for more precision
# Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec
# June 15 is in Q2, so:
# - If born Q1-Q2: age on June 15 = 2012 - BIRTHYR
# - If born Q3-Q4: age on June 15 = 2012 - BIRTHYR - 1 (not yet had birthday)

def calc_age_june2012(row):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - row['BIRTHYR']
    # If born after June (Q3 or Q4), subtract 1
    if row['BIRTHQTR'] >= 3:
        return base_age - 1
    return base_age

df['age_june2012'] = df.apply(calc_age_june2012, axis=1)

# Treatment group: Ages 26-30 as of June 15, 2012 (DACA eligible by age)
# Control group: Ages 31-35 as of June 15, 2012 (too old for DACA)
df['treatment'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control'] == 1)]

print(f"Treatment group (ages 26-30): {df['treatment'].sum():,} observations")
print(f"Control group (ages 31-35): {df['control'].sum():,} observations")
print(f"Total in analysis sample: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 4: Define Time Periods and Outcome Variable
# -----------------------------------------------------------------------------
print("\n[Step 4] Defining time periods and outcome variable...")

# Exclude 2012 (DACA implemented mid-year)
df = df[df['YEAR'] != 2012]

# Pre-period: 2006-2011
# Post-period: 2013-2016
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Outcome: Full-time employment (usually works 35+ hours per week)
# UHRSWORK: Usual hours worked per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create interaction term for DiD
df['treat_post'] = df['treatment'] * df['post']

print(f"Pre-period observations: {(df['post']==0).sum():,}")
print(f"Post-period observations: {(df['post']==1).sum():,}")
print(f"Final analysis sample: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 5: Summary Statistics
# -----------------------------------------------------------------------------
print("\n[Step 5] Generating summary statistics...")

# Create covariates
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)  # Married (spouse present or absent)

# Education categories
df['educ_lesshs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['educ_coll'] = (df['EDUC'] >= 10).astype(int)

# Summary stats by group
print("\nSummary Statistics (Pre-period, unweighted means):")
pre_df = df[df['post'] == 0]

stats_vars = ['fulltime', 'female', 'married', 'AGE', 'educ_lesshs', 'educ_hs',
              'educ_somecoll', 'educ_coll']

summary_stats = pd.DataFrame()
for var in stats_vars:
    treat_mean = pre_df[pre_df['treatment'] == 1][var].mean()
    control_mean = pre_df[pre_df['treatment'] == 0][var].mean()
    diff = treat_mean - control_mean
    summary_stats = pd.concat([summary_stats, pd.DataFrame({
        'Variable': [var],
        'Treatment': [treat_mean],
        'Control': [control_mean],
        'Difference': [diff]
    })])

summary_stats = summary_stats.reset_index(drop=True)
print(summary_stats.to_string(index=False))

# Sample sizes by year and group
print("\nSample sizes by year and treatment status:")
sample_by_year = df.groupby(['YEAR', 'treatment']).size().unstack(fill_value=0)
sample_by_year.columns = ['Control', 'Treatment']
print(sample_by_year)

# Full-time employment rates by year and group
print("\nFull-time employment rates by year and treatment status:")
ft_by_year = df.groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack(fill_value=0)
ft_by_year.columns = ['Control', 'Treatment']
print(ft_by_year.round(4))

# -----------------------------------------------------------------------------
# STEP 6: Difference-in-Differences Analysis
# -----------------------------------------------------------------------------
print("\n[Step 6] Running Difference-in-Differences analysis...")

# Model 1: Basic DiD (no covariates)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treatment + post + treat_post',
                  data=df, weights=df['PERWT'])
result1 = model1.fit(cov_type='HC1')  # Robust standard errors
print(result1.summary().tables[1])

# Model 2: DiD with demographic covariates
print("\n--- Model 2: DiD with demographic covariates ---")
model2 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + AGE',
                  data=df, weights=df['PERWT'])
result2 = model2.fit(cov_type='HC1')
print(result2.summary().tables[1])

# Model 3: DiD with demographic + education covariates
print("\n--- Model 3: DiD with demographic and education covariates ---")
model3 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + AGE + educ_hs + educ_somecoll + educ_coll',
                  data=df, weights=df['PERWT'])
result3 = model3.fit(cov_type='HC1')
print(result3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects ---")
df['year_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.wls('fulltime ~ treatment + C(year_factor) + treat_post + female + married + AGE + educ_hs + educ_somecoll + educ_coll',
                  data=df, weights=df['PERWT'])
result4 = model4.fit(cov_type='HC1')

# Extract key coefficients
print("\nKey coefficient (treat_post):")
print(f"  Coefficient: {result4.params['treat_post']:.6f}")
print(f"  Std Error:   {result4.bse['treat_post']:.6f}")
print(f"  t-statistic: {result4.tvalues['treat_post']:.4f}")
print(f"  p-value:     {result4.pvalues['treat_post']:.6f}")
ci = result4.conf_int().loc['treat_post']
print(f"  95% CI:      [{ci[0]:.6f}, {ci[1]:.6f}]")

# Model 5: DiD with state fixed effects (preferred specification)
print("\n--- Model 5: DiD with year and state fixed effects (PREFERRED) ---")
df['state_factor'] = pd.Categorical(df['STATEFIP'])
model5 = smf.wls('fulltime ~ treatment + C(year_factor) + C(state_factor) + treat_post + female + married + AGE + educ_hs + educ_somecoll + educ_coll',
                  data=df, weights=df['PERWT'])
result5 = model5.fit(cov_type='HC1')

print("\nPreferred specification key coefficient (treat_post):")
print(f"  Coefficient: {result5.params['treat_post']:.6f}")
print(f"  Std Error:   {result5.bse['treat_post']:.6f}")
print(f"  t-statistic: {result5.tvalues['treat_post']:.4f}")
print(f"  p-value:     {result5.pvalues['treat_post']:.6f}")
ci5 = result5.conf_int().loc['treat_post']
print(f"  95% CI:      [{ci5[0]:.6f}, {ci5[1]:.6f}]")

# -----------------------------------------------------------------------------
# STEP 7: Manual DiD Calculation (verification)
# -----------------------------------------------------------------------------
print("\n[Step 7] Manual DiD calculation (verification)...")

# Calculate weighted means
def weighted_mean(data, var, weight):
    return np.average(data[var], weights=data[weight])

# Pre-period means
treat_pre = weighted_mean(df[(df['treatment']==1) & (df['post']==0)], 'fulltime', 'PERWT')
control_pre = weighted_mean(df[(df['treatment']==0) & (df['post']==0)], 'fulltime', 'PERWT')

# Post-period means
treat_post = weighted_mean(df[(df['treatment']==1) & (df['post']==1)], 'fulltime', 'PERWT')
control_post = weighted_mean(df[(df['treatment']==0) & (df['post']==1)], 'fulltime', 'PERWT')

print("\n2x2 DiD Table (weighted):")
print(f"                  Pre-DACA    Post-DACA    Diff")
print(f"Treatment (26-30): {treat_pre:.4f}      {treat_post:.4f}       {treat_post-treat_pre:+.4f}")
print(f"Control (31-35):   {control_pre:.4f}      {control_post:.4f}       {control_post-control_pre:+.4f}")
print(f"DiD Estimate:                              {(treat_post-treat_pre)-(control_post-control_pre):+.4f}")

# -----------------------------------------------------------------------------
# STEP 8: Robustness Checks
# -----------------------------------------------------------------------------
print("\n[Step 8] Running robustness checks...")

# 8a. Analysis by sex
print("\n--- 8a. Separate estimates by sex ---")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treatment + C(year_factor) + treat_post',
                         data=df_sex, weights=df_sex['PERWT'])
    result_sex = model_sex.fit(cov_type='HC1')
    print(f"{sex_label}: DiD estimate = {result_sex.params['treat_post']:.4f} "
          f"(SE = {result_sex.bse['treat_post']:.4f}), N = {len(df_sex):,}")

# 8b. Placebo test: Compare 31-35 vs 36-40 (neither eligible)
print("\n--- 8b. Placebo test: Age 31-35 vs 36-40 ---")
# Reload data for placebo
chunks_placebo = []
for chunk in pd.read_csv('data/data.csv', usecols=cols_to_use,
                          chunksize=chunk_size, low_memory=False):
    chunk = chunk[chunk['HISPAN'] == 1]
    chunk = chunk[chunk['BPL'] == 200]
    chunk = chunk[chunk['CITIZEN'] == 3]
    chunk = chunk[(chunk['YEAR'] >= 2006) & (chunk['YEAR'] <= 2016)]
    chunk = chunk[chunk['YRIMMIG'] > 0]
    chunk = chunk[chunk['YRIMMIG'] != 996]
    chunk['age_at_immig'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['age_at_immig'] < 16]
    chunk = chunk[chunk['YRIMMIG'] <= 2007]
    chunks_placebo.append(chunk)

df_placebo = pd.concat(chunks_placebo, ignore_index=True)
df_placebo['age_june2012'] = df_placebo.apply(calc_age_june2012, axis=1)
df_placebo['placebo_treat'] = ((df_placebo['age_june2012'] >= 31) & (df_placebo['age_june2012'] <= 35)).astype(int)
df_placebo['placebo_control'] = ((df_placebo['age_june2012'] >= 36) & (df_placebo['age_june2012'] <= 40)).astype(int)
df_placebo = df_placebo[(df_placebo['placebo_treat'] == 1) | (df_placebo['placebo_control'] == 1)]
df_placebo = df_placebo[df_placebo['YEAR'] != 2012]
df_placebo['post'] = (df_placebo['YEAR'] >= 2013).astype(int)
df_placebo['fulltime'] = (df_placebo['UHRSWORK'] >= 35).astype(int)
df_placebo['treat_post'] = df_placebo['placebo_treat'] * df_placebo['post']
df_placebo['year_factor'] = pd.Categorical(df_placebo['YEAR'])

model_placebo = smf.wls('fulltime ~ placebo_treat + C(year_factor) + treat_post',
                         data=df_placebo, weights=df_placebo['PERWT'])
result_placebo = model_placebo.fit(cov_type='HC1')
print(f"Placebo DiD estimate: {result_placebo.params['treat_post']:.4f} "
      f"(SE = {result_placebo.bse['treat_post']:.4f}), "
      f"p-value = {result_placebo.pvalues['treat_post']:.4f}")

# 8c. Event study (year-by-year effects)
print("\n--- 8c. Event study (year-by-year treatment effects) ---")
df['year_treat'] = df['treatment'] * df['YEAR']
event_study_results = []
for year in sorted(df['YEAR'].unique()):
    df[f'treat_year_{year}'] = (df['treatment'] * (df['YEAR'] == year)).astype(int)

# Drop 2011 as reference year
treat_year_vars = [f'treat_year_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]
formula_event = 'fulltime ~ treatment + C(year_factor) + ' + ' + '.join(treat_year_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT'])
result_event = model_event.fit(cov_type='HC1')

print("\nYear-specific treatment effects (reference: 2011):")
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var_name = f'treat_year_{year}'
        coef = result_event.params[var_name]
        se = result_event.bse[var_name]
        pval = result_event.pvalues[var_name]
        ci = result_event.conf_int().loc[var_name]
        marker = "*" if pval < 0.05 else ""
        print(f"  {year}: {coef:+.4f} (SE={se:.4f}, p={pval:.3f}) {marker}")

# -----------------------------------------------------------------------------
# STEP 9: Export Results
# -----------------------------------------------------------------------------
print("\n[Step 9] Exporting results...")

# Save key results
results_summary = {
    'Model': ['Basic DiD', 'DiD + Demographics', 'DiD + Demographics + Education',
              'DiD + Year FE', 'DiD + Year + State FE (Preferred)'],
    'Coefficient': [result1.params['treat_post'], result2.params['treat_post'],
                   result3.params['treat_post'], result4.params['treat_post'],
                   result5.params['treat_post']],
    'Std_Error': [result1.bse['treat_post'], result2.bse['treat_post'],
                  result3.bse['treat_post'], result4.bse['treat_post'],
                  result5.bse['treat_post']],
    'p_value': [result1.pvalues['treat_post'], result2.pvalues['treat_post'],
                result3.pvalues['treat_post'], result4.pvalues['treat_post'],
                result5.pvalues['treat_post']],
    'CI_lower': [result1.conf_int().loc['treat_post'][0],
                 result2.conf_int().loc['treat_post'][0],
                 result3.conf_int().loc['treat_post'][0],
                 result4.conf_int().loc['treat_post'][0],
                 result5.conf_int().loc['treat_post'][0]],
    'CI_upper': [result1.conf_int().loc['treat_post'][1],
                 result2.conf_int().loc['treat_post'][1],
                 result3.conf_int().loc['treat_post'][1],
                 result4.conf_int().loc['treat_post'][1],
                 result5.conf_int().loc['treat_post'][1]]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save full-time rates by year for plotting
ft_rates = df.groupby(['YEAR', 'treatment']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_rates.columns = ['Control', 'Treatment']
ft_rates.to_csv('fulltime_rates_by_year.csv')
print("Full-time rates by year saved to fulltime_rates_by_year.csv")

# Save sample sizes
sample_sizes = df.groupby(['YEAR', 'treatment', 'post']).agg({
    'fulltime': ['count', 'sum'],
    'PERWT': 'sum'
}).reset_index()
sample_sizes.columns = ['Year', 'Treatment', 'Post', 'N', 'N_Fulltime', 'Sum_Weights']
sample_sizes.to_csv('sample_sizes.csv', index=False)
print("Sample sizes saved to sample_sizes.csv")

# Save summary statistics
summary_stats.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

# -----------------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE - SUMMARY OF KEY FINDINGS")
print("=" * 70)

print(f"\nPreferred Estimate (Model 5: Year + State FE with covariates):")
print(f"  Effect of DACA eligibility on full-time employment: {result5.params['treat_post']:.4f}")
print(f"  Standard Error: {result5.bse['treat_post']:.4f}")
print(f"  95% Confidence Interval: [{ci5[0]:.4f}, {ci5[1]:.4f}]")
print(f"  p-value: {result5.pvalues['treat_post']:.4f}")
print(f"\n  Sample Size: {len(df):,} person-year observations")
print(f"  Treatment Group (Ages 26-30): {df['treatment'].sum():,} observations")
print(f"  Control Group (Ages 31-35): {(df['treatment']==0).sum():,} observations")

print("\n" + "=" * 70)

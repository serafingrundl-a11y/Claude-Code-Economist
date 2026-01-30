"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals

This script performs a difference-in-differences analysis comparing:
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_06")

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)
print("\nLoading data...")

# Load the data in chunks due to large file size
# We'll filter as we load to reduce memory usage
chunksize = 500000
chunks = []

for i, chunk in enumerate(pd.read_csv('data/data.csv', chunksize=chunksize)):
    # Filter to relevant years (2006-2016)
    chunk = chunk[(chunk['YEAR'] >= 2006) & (chunk['YEAR'] <= 2016)]

    # Filter to Hispanic-Mexican (HISPAN == 1)
    chunk = chunk[chunk['HISPAN'] == 1]

    # Filter to born in Mexico (BPL == 200)
    chunk = chunk[chunk['BPL'] == 200]

    # Filter to non-citizens (CITIZEN == 3)
    # This proxies for undocumented status as we cannot distinguish documented vs undocumented
    chunk = chunk[chunk['CITIZEN'] == 3]

    chunks.append(chunk)
    print(f"  Processed chunk {i+1}...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nData loaded. Total observations after initial filtering: {len(df):,}")

# Data cleaning and variable creation
print("\n" + "="*80)
print("DATA CLEANING AND VARIABLE CREATION")
print("="*80)

# Check for missing values in key variables
print("\nMissing values in key variables:")
key_vars = ['YEAR', 'BIRTHYR', 'UHRSWORK', 'PERWT', 'SEX', 'EDUC', 'MARST', 'AGE']
for var in key_vars:
    missing = df[var].isna().sum()
    print(f"  {var}: {missing:,} ({100*missing/len(df):.2f}%)")

# Create full-time employment indicator (35+ hours per week)
# UHRSWORK: usual hours worked per week
# Value 0 means N/A (not applicable, likely not working)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Create employed indicator (for robustness)
# EMPSTAT: 1 = Employed
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Define age at DACA implementation (June 15, 2012)
# We use the person's age as of 2012 based on their birth year
# Since we don't have exact birth date, we approximate
df['age_at_daca'] = 2012 - df['BIRTHYR']

# DACA eligibility criteria:
# 1. Age < 31 as of June 15, 2012 (born after June 15, 1981)
# 2. Arrived before 16th birthday
# 3. Continuous US residence since June 15, 2007
# 4. Present in US on June 15, 2012

# For this analysis, following the instructions:
# Treatment group: Ages 26-30 at DACA implementation
# Control group: Ages 31-35 at DACA implementation

# Create treatment group indicator
# Treatment = 1 if age at DACA was 26-30
# Treatment = 0 if age at DACA was 31-35
df['treat'] = np.where((df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 30), 1,
                np.where((df['age_at_daca'] >= 31) & (df['age_at_daca'] <= 35), 0, np.nan))

# Filter to only treatment and control groups
df = df.dropna(subset=['treat'])
df['treat'] = df['treat'].astype(int)

print(f"\nObservations after restricting to treatment/control age groups: {len(df):,}")

# Create post-DACA indicator
# DACA implemented June 15, 2012
# Since we can't distinguish before/after within 2012, we exclude 2012
# Post = 1 for years 2013-2016
# Post = 0 for years 2006-2011
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Exclude 2012 due to ambiguity (policy implemented mid-year)
df = df[df['YEAR'] != 2012]

print(f"Observations after excluding 2012: {len(df):,}")

# Create interaction term
df['treat_post'] = df['treat'] * df['post']

# Additional DACA eligibility proxies
# YRIMMIG: Year of immigration
# Need to have arrived before age 16 to be DACA eligible
df['arrived_before_16'] = (df['YRIMMIG'] > 0) & ((df['YRIMMIG'] - df['BIRTHYR']) < 16)

# Need to have been in US since 2007 for DACA eligibility
df['in_us_since_2007'] = (df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)

# Create a stricter DACA eligibility proxy
# For treated group: must have arrived before 16 and been in US since 2007
# For control group: would have been eligible if not for age
df['daca_eligible_proxy'] = (df['arrived_before_16']) & (df['in_us_since_2007'])

print(f"\nObservations meeting DACA arrival criteria: {df['daca_eligible_proxy'].sum():,} ({100*df['daca_eligible_proxy'].mean():.1f}%)")

# Create covariates
# Sex (female indicator)
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status (married indicator)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['edu_less_hs'] = (df['EDUC'] <= 5).astype(int)
df['edu_hs'] = (df['EDUC'] == 6).astype(int)
df['edu_some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['edu_college_plus'] = (df['EDUC'] >= 10).astype(int)

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']
df.loc[df['years_in_us'] < 0, 'years_in_us'] = np.nan

# State fixed effects
df['state'] = df['STATEFIP']

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Summary statistics by treatment status and period
print("\n--- Sample sizes by group ---")
for post_val in [0, 1]:
    for treat_val in [0, 1]:
        n = len(df[(df['post'] == post_val) & (df['treat'] == treat_val)])
        period = "Pre-DACA (2006-2011)" if post_val == 0 else "Post-DACA (2013-2016)"
        group = "Treatment (26-30)" if treat_val == 1 else "Control (31-35)"
        print(f"  {period}, {group}: {n:,}")

# Full-time employment rates by group
print("\n--- Full-time employment rates by group ---")
ft_rates = df.groupby(['post', 'treat'])['fulltime'].mean()
print("Pre-DACA:")
print(f"  Control (31-35): {ft_rates[(0, 0)]:.4f}")
print(f"  Treatment (26-30): {ft_rates[(0, 1)]:.4f}")
print("Post-DACA:")
print(f"  Control (31-35): {ft_rates[(1, 0)]:.4f}")
print(f"  Treatment (26-30): {ft_rates[(1, 1)]:.4f}")

# Simple DD calculation
dd_simple = (ft_rates[(1, 1)] - ft_rates[(0, 1)]) - (ft_rates[(1, 0)] - ft_rates[(0, 0)])
print(f"\nSimple Difference-in-Differences: {dd_simple:.4f}")

# Descriptive statistics table
print("\n--- Descriptive Statistics (Pre-period, by treatment status) ---")
pre_df = df[df['post'] == 0]
desc_vars = ['fulltime', 'employed', 'female', 'married', 'AGE',
             'edu_less_hs', 'edu_hs', 'edu_some_college', 'edu_college_plus']

print(f"{'Variable':<20} {'Control Mean':>12} {'Control SD':>12} {'Treat Mean':>12} {'Treat SD':>12}")
print("-" * 68)
for var in desc_vars:
    ctrl_mean = pre_df[pre_df['treat'] == 0][var].mean()
    ctrl_sd = pre_df[pre_df['treat'] == 0][var].std()
    treat_mean = pre_df[pre_df['treat'] == 1][var].mean()
    treat_sd = pre_df[pre_df['treat'] == 1][var].std()
    print(f"{var:<20} {ctrl_mean:>12.4f} {ctrl_sd:>12.4f} {treat_mean:>12.4f} {treat_sd:>12.4f}")

# Save descriptive statistics for report
desc_stats = []
for var in desc_vars:
    ctrl_mean = pre_df[pre_df['treat'] == 0][var].mean()
    ctrl_sd = pre_df[pre_df['treat'] == 0][var].std()
    treat_mean = pre_df[pre_df['treat'] == 1][var].mean()
    treat_sd = pre_df[pre_df['treat'] == 1][var].std()
    desc_stats.append({
        'Variable': var,
        'Control_Mean': ctrl_mean,
        'Control_SD': ctrl_sd,
        'Treatment_Mean': treat_mean,
        'Treatment_SD': treat_sd
    })
desc_df = pd.DataFrame(desc_stats)
desc_df.to_csv('descriptive_stats.csv', index=False)

print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Model 1: Basic DD without controls
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df['state']})
print(model1.summary())

# Model 2: DD with demographic controls
print("\n--- Model 2: DD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df['state']})
print(model2.summary())

# Model 3: DD with demographic controls and state fixed effects
print("\n--- Model 3: DD with Demographic Controls and State FE ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(EDUC) + C(state)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df['state']})
print("\nCoefficients for main variables:")
print(f"treat_post (DD estimate): {model3.params['treat_post']:.5f}")
print(f"  Std. Error: {model3.bse['treat_post']:.5f}")
print(f"  t-stat: {model3.tvalues['treat_post']:.3f}")
print(f"  p-value: {model3.pvalues['treat_post']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['treat_post', 0]:.5f}, {model3.conf_int().loc['treat_post', 1]:.5f}]")

# Model 4: DD with year fixed effects instead of simple post
print("\n--- Model 4: DD with Year Fixed Effects ---")
model4 = smf.wls('fulltime ~ treat + C(YEAR) + treat:C(YEAR) + female + married + C(EDUC) + C(state)',
                  data=df, weights=df['PERWT']).fit(cov_type='cluster',
                  cov_kwds={'groups': df['state']})
print("\nYear-specific treatment effects:")
for var in model4.params.index:
    if 'treat:' in var and 'YEAR' in var:
        year = var.split('[T.')[1].split(']')[0]
        print(f"  {year}: {model4.params[var]:.5f} (SE: {model4.bse[var]:.5f}, p: {model4.pvalues[var]:.4f})")

# Model 5: Restricted to those meeting DACA arrival criteria
print("\n--- Model 5: Restricted Sample (DACA-eligible proxy) ---")
df_eligible = df[df['daca_eligible_proxy'] == True]
print(f"Sample size: {len(df_eligible):,}")
if len(df_eligible) > 100:
    model5 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(EDUC)',
                      data=df_eligible, weights=df_eligible['PERWT']).fit(cov_type='HC1')
    print(f"treat_post (DD estimate): {model5.params['treat_post']:.5f}")
    print(f"  Std. Error: {model5.bse['treat_post']:.5f}")
    print(f"  p-value: {model5.pvalues['treat_post']:.4f}")
else:
    print("  Insufficient observations for restricted sample analysis")
    model5 = None

# Save regression results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results_summary = {
    'Model': ['Basic DD', 'DD + Demographics', 'DD + Demographics + State FE', 'Event Study'],
    'Estimate': [model1.params['treat_post'], model2.params['treat_post'],
                 model3.params['treat_post'], 'See year effects'],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], '-'],
    'P_Value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], '-'],
    'N': [model1.nobs, model2.nobs, model3.nobs, model4.nobs]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("Results saved to regression_results.csv")

# Create summary statistics for the report
print("\n" + "="*80)
print("SUMMARY FOR REPORT")
print("="*80)

print(f"\n=== PREFERRED ESTIMATE ===")
print(f"Model: DD with demographic controls and state fixed effects (Model 3)")
print(f"Effect Size: {model3.params['treat_post']:.5f}")
print(f"Standard Error: {model3.bse['treat_post']:.5f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.5f}, {model3.conf_int().loc['treat_post', 1]:.5f}]")
print(f"Sample Size: {int(model3.nobs):,}")
print(f"P-value: {model3.pvalues['treat_post']:.4f}")

# Additional robustness checks
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Employment (any hours) instead of full-time
print("\n--- Robustness: Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ treat + post + treat_post + female + married + C(EDUC) + C(state)',
                     data=df, weights=df['PERWT']).fit(cov_type='cluster',
                     cov_kwds={'groups': df['state']})
print(f"DD estimate for employment: {model_emp.params['treat_post']:.5f}")
print(f"  Std. Error: {model_emp.bse['treat_post']:.5f}")
print(f"  p-value: {model_emp.pvalues['treat_post']:.4f}")

# Robustness 2: By sex
print("\n--- Robustness: Separate by Sex ---")
for sex in [1, 2]:
    sex_label = "Male" if sex == 1 else "Female"
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post + married + C(EDUC) + C(state)',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='cluster',
                         cov_kwds={'groups': df_sex['state']})
    print(f"{sex_label}: DD estimate = {model_sex.params['treat_post']:.5f} (SE: {model_sex.bse['treat_post']:.5f}, p: {model_sex.pvalues['treat_post']:.4f})")

# Create data for parallel trends visualization
print("\n--- Creating data for parallel trends plot ---")
trends_data = df.groupby(['YEAR', 'treat']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).reset_index()
trends_data.columns = ['Year', 'Treatment', 'FullTime_Rate', 'Weight']
trends_data.to_csv('parallel_trends_data.csv', index=False)
print("Parallel trends data saved to parallel_trends_data.csv")

# Create weighted means by year and treatment
print("\n--- Full-time employment rates by year and group ---")
for year in sorted(df['YEAR'].unique()):
    for treat in [0, 1]:
        subset = df[(df['YEAR'] == year) & (df['treat'] == treat)]
        rate = np.average(subset['fulltime'], weights=subset['PERWT'])
        group = "Treatment" if treat == 1 else "Control"
        print(f"Year {year}, {group}: {rate:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Final summary statistics for the report
final_stats = {
    'Total_Observations': len(df),
    'N_Treatment': len(df[df['treat'] == 1]),
    'N_Control': len(df[df['treat'] == 0]),
    'N_Pre': len(df[df['post'] == 0]),
    'N_Post': len(df[df['post'] == 1]),
    'DD_Estimate': model3.params['treat_post'],
    'DD_SE': model3.bse['treat_post'],
    'DD_CI_Lower': model3.conf_int().loc['treat_post', 0],
    'DD_CI_Upper': model3.conf_int().loc['treat_post', 1],
    'DD_Pvalue': model3.pvalues['treat_post']
}

final_df = pd.DataFrame([final_stats])
final_df.to_csv('final_summary.csv', index=False)
print("\nFinal summary saved to final_summary.csv")

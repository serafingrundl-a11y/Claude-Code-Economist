"""
DACA Replication Analysis Script
================================
This script performs the difference-in-differences analysis for estimating
the effect of DACA eligibility on full-time employment among Mexican-born
Hispanic individuals in the United States.

Research Design:
- Treatment: Ages 26-30 as of June 15, 2012
- Control: Ages 31-35 as of June 15, 2012
- Pre-period: 2006-2011
- Post-period: 2013-2016
- Outcome: Full-time employment (usually working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

def calculate_age_june_2012(birthyr, birthqtr):
    """Calculate age as of June 15, 2012"""
    base_age = 2012 - birthyr
    # If born in Q3 or Q4, they haven't had their birthday by June 15
    if birthqtr in [3, 4]:
        return base_age - 1
    else:
        return base_age

# =============================================================================
# STEP 1: Load Data with Filtering
# =============================================================================
print("="*70)
print("STEP 1: Loading and Filtering ACS Data")
print("="*70)

# Load only necessary columns and filter immediately to reduce memory
cols_needed = ['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR', 'BIRTHQTR',
               'AGE', 'SEX', 'MARST', 'EDUCD', 'UHRSWORK', 'EMPSTAT', 'PERWT', 'STATEFIP']

# Read in chunks to filter
chunks = []
chunksize = 1000000
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize):
    # Apply immediate filtering
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    chunk = chunk[chunk['YEAR'] != 2012]
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Observations after initial filters: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Calculate Age as of June 15, 2012
# =============================================================================
print("\n" + "="*70)
print("STEP 2: Calculating Age as of June 15, 2012")
print("="*70)

df['age_june_2012'] = df.apply(lambda row: calculate_age_june_2012(row['BIRTHYR'], row['BIRTHQTR']), axis=1)
print(f"Age range in sample: {df['age_june_2012'].min()} to {df['age_june_2012'].max()}")

# =============================================================================
# STEP 3: DACA Eligibility Criteria
# =============================================================================
print("\n" + "="*70)
print("STEP 3: Applying DACA Eligibility Criteria")
print("="*70)

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter: Arrived before 16th birthday
df = df[df['age_at_arrival'] < 16]
print(f"After arrived before age 16: {len(df):,}")

# Filter: Arrived by 2007 (continuous residence since June 15, 2007)
df = df[df['YRIMMIG'] <= 2007]
print(f"After YRIMMIG <= 2007 (continuous residence): {len(df):,}")

# =============================================================================
# STEP 4: Define Treatment and Control Groups
# =============================================================================
print("\n" + "="*70)
print("STEP 4: Defining Treatment and Control Groups")
print("="*70)

# Treatment group: Age 26-30 as of June 15, 2012
# Control group: Age 31-35 as of June 15, 2012
df['treated'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"After restricting to ages 26-35 (as of June 2012): {len(df):,}")
print(f"Treatment group (26-30): {df['treated'].sum():,}")
print(f"Control group (31-35): {df['control'].sum():,}")

# =============================================================================
# STEP 5: Define Post-Treatment Period and Outcome
# =============================================================================
print("\n" + "="*70)
print("STEP 5: Defining Time Periods and Outcome")
print("="*70)

# Post = 1 for years 2013-2016, 0 for 2006-2011
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"Pre-period observations (2006-2011): {(df['post']==0).sum():,}")
print(f"Post-period observations (2013-2016): {(df['post']==1).sum():,}")

# Full-time employment: Usually working 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employment rate: {df['fulltime'].mean()*100:.2f}%")

# Create control variables
df['male'] = (df['SEX'] == 1).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)
df['educ_hs'] = (df['EDUCD'] >= 62).astype(int)

# Create interaction
df['treated_post'] = df['treated'] * df['post']

# =============================================================================
# STEP 6: Descriptive Statistics
# =============================================================================
print("\n" + "="*70)
print("STEP 6: Descriptive Statistics")
print("="*70)

# By treatment status and time period
desc_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'UHRSWORK': 'mean',
    'AGE': 'mean',
    'male': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nDescriptive Statistics by Group and Period:")
print(desc_stats)

# Calculate raw difference-in-differences
pre_treat = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
pre_control = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()
post_control = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_control - pre_control)

print(f"\n--- Raw Difference-in-Differences ---")
print(f"Treatment Pre:    {pre_treat:.4f}")
print(f"Treatment Post:   {post_treat:.4f}")
print(f"Treatment Change: {post_treat - pre_treat:.4f}")
print(f"Control Pre:      {pre_control:.4f}")
print(f"Control Post:     {post_control:.4f}")
print(f"Control Change:   {post_control - pre_control:.4f}")
print(f"Raw DiD:          {raw_did:.4f}")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n" + "="*70)
print("STEP 7: Difference-in-Differences Regression Analysis")
print("="*70)

# Store results
results_list = []

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(model1.summary().tables[1])
results_list.append({
    'Model': 'Model 1: Basic DiD (Unweighted)',
    'Coefficient': model1.params['treated_post'],
    'SE': model1.bse['treated_post'],
    'CI_Lower': model1.conf_int().loc['treated_post', 0],
    'CI_Upper': model1.conf_int().loc['treated_post', 1],
    'P_Value': model1.pvalues['treated_post'],
    'N': len(df)
})

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (Weighted) ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
results_list.append({
    'Model': 'Model 2: Basic DiD (Weighted)',
    'Coefficient': model2.params['treated_post'],
    'SE': model2.bse['treated_post'],
    'CI_Lower': model2.conf_int().loc['treated_post', 0],
    'CI_Upper': model2.conf_int().loc['treated_post', 1],
    'P_Value': model2.pvalues['treated_post'],
    'N': len(df)
})

# Model 3: DiD with controls (weighted)
print("\n--- Model 3: DiD with Demographics (Weighted) ---")
model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + educ_hs',
                  data=df, weights=df['PERWT']).fit()
print(model3.summary().tables[1])
results_list.append({
    'Model': 'Model 3: With Demographics (Weighted)',
    'Coefficient': model3.params['treated_post'],
    'SE': model3.bse['treated_post'],
    'CI_Lower': model3.conf_int().loc['treated_post', 0],
    'CI_Upper': model3.conf_int().loc['treated_post', 1],
    'P_Value': model3.pvalues['treated_post'],
    'N': len(df)
})

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with Year Fixed Effects (Weighted) ---")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + male + married + educ_hs',
                  data=df, weights=df['PERWT']).fit()

print(f"DiD Coefficient (treated_post): {model4.params['treated_post']:.4f}")
print(f"Standard Error: {model4.bse['treated_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"P-value: {model4.pvalues['treated_post']:.4f}")
results_list.append({
    'Model': 'Model 4: With Year FE (Weighted)',
    'Coefficient': model4.params['treated_post'],
    'SE': model4.bse['treated_post'],
    'CI_Lower': model4.conf_int().loc['treated_post', 0],
    'CI_Upper': model4.conf_int().loc['treated_post', 1],
    'P_Value': model4.pvalues['treated_post'],
    'N': len(df)
})

# Model 5: DiD with state fixed effects (weighted)
print("\n--- Model 5: DiD with State and Year Fixed Effects (Weighted) ---")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + male + married + educ_hs',
                  data=df, weights=df['PERWT']).fit()

print(f"DiD Coefficient (treated_post): {model5.params['treated_post']:.4f}")
print(f"Standard Error: {model5.bse['treated_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"P-value: {model5.pvalues['treated_post']:.4f}")
results_list.append({
    'Model': 'Model 5: With Year + State FE (Weighted)',
    'Coefficient': model5.params['treated_post'],
    'SE': model5.bse['treated_post'],
    'CI_Lower': model5.conf_int().loc['treated_post', 0],
    'CI_Upper': model5.conf_int().loc['treated_post', 1],
    'P_Value': model5.pvalues['treated_post'],
    'N': len(df)
})

# =============================================================================
# STEP 8: Preferred Specification with Robust Standard Errors
# =============================================================================
print("\n" + "="*70)
print("STEP 8: Preferred Specification with Robust Standard Errors")
print("="*70)

model_preferred = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + male + married + educ_hs',
                           data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\n=== PREFERRED ESTIMATE ===")
print(f"Effect Size: {model_preferred.params['treated_post']:.4f}")
print(f"Robust SE:   {model_preferred.bse['treated_post']:.4f}")
print(f"95% CI:      [{model_preferred.conf_int().loc['treated_post', 0]:.4f}, {model_preferred.conf_int().loc['treated_post', 1]:.4f}]")
print(f"T-statistic: {model_preferred.tvalues['treated_post']:.4f}")
print(f"P-value:     {model_preferred.pvalues['treated_post']:.4f}")
print(f"Sample Size: {len(df):,}")
results_list.append({
    'Model': 'Model 6: Preferred (Year + State FE, Robust SE)',
    'Coefficient': model_preferred.params['treated_post'],
    'SE': model_preferred.bse['treated_post'],
    'CI_Lower': model_preferred.conf_int().loc['treated_post', 0],
    'CI_Upper': model_preferred.conf_int().loc['treated_post', 1],
    'P_Value': model_preferred.pvalues['treated_post'],
    'N': len(df)
})

# =============================================================================
# STEP 9: Heterogeneity Analysis
# =============================================================================
print("\n" + "="*70)
print("STEP 9: Heterogeneity Analysis")
print("="*70)

# By gender
print("\n--- Heterogeneity by Gender ---")
df_male = df[df['male'] == 1]
df_female = df[df['male'] == 0]

model_male = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + married + educ_hs',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + married + educ_hs',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"Males:   DiD = {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f}), N = {len(df_male):,}")
print(f"Females: DiD = {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f}), N = {len(df_female):,}")

# Store heterogeneity results
het_results = [
    {'Subgroup': 'Males', 'Coefficient': model_male.params['treated_post'],
     'SE': model_male.bse['treated_post'], 'N': len(df_male)},
    {'Subgroup': 'Females', 'Coefficient': model_female.params['treated_post'],
     'SE': model_female.bse['treated_post'], 'N': len(df_female)}
]

# =============================================================================
# STEP 10: Event Study / Pre-Trends
# =============================================================================
print("\n" + "="*70)
print("STEP 10: Event Study / Pre-Trends Check")
print("="*70)

# Create year dummies interacted with treatment (omit 2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'treat_{yr}'] = df['treated'] * (df['YEAR'] == yr).astype(int)

model_event = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + male + married + educ_hs',
                       data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Event Study Coefficients (Reference: 2011):")
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
event_results = []
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    ci_lo = model_event.conf_int().loc[var, 0]
    ci_hi = model_event.conf_int().loc[var, 1]
    print(f"{var}: {coef:8.4f} (SE: {se:.4f}, p={pval:.3f})")
    event_results.append({
        'Year_Interaction': var,
        'Coefficient': coef,
        'SE': se,
        'CI_Lower': ci_lo,
        'CI_Upper': ci_hi,
        'P_Value': pval
    })

# =============================================================================
# STEP 11: Summary Statistics for Report
# =============================================================================
print("\n" + "="*70)
print("STEP 11: Summary Statistics for Report")
print("="*70)

print(f"\n--- Sample Characteristics ---")
print(f"Total observations: {len(df):,}")
print(f"Treatment group (26-30): {df['treated'].sum():,}")
print(f"Control group (31-35): {(1-df['treated']).sum():,}")
print(f"Pre-period (2006-2011): {(df['post']==0).sum():,}")
print(f"Post-period (2013-2016): {(df['post']==1).sum():,}")

print("\n--- Variable Means by Treatment Status ---")
summary_vars = ['fulltime', 'UHRSWORK', 'AGE', 'male', 'married', 'educ_hs']
for var in summary_vars:
    treat_mean = df[df['treated']==1][var].mean()
    ctrl_mean = df[df['treated']==0][var].mean()
    print(f"{var:15} Treatment: {treat_mean:.4f}  Control: {ctrl_mean:.4f}")

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n" + "="*70)
print("STEP 12: Saving Results")
print("="*70)

# Save regression results
results_df = pd.DataFrame(results_list)
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to: regression_results.csv")

# Save event study results
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to: event_study_results.csv")

# Save heterogeneity results
het_df = pd.DataFrame(het_results)
het_df.to_csv('heterogeneity_results.csv', index=False)
print("Heterogeneity results saved to: heterogeneity_results.csv")

# Save summary statistics
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'UHRSWORK': ['mean', 'std'],
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to: summary_statistics.csv")

# Save key results for reporting
key_results = {
    'preferred_estimate': model_preferred.params['treated_post'],
    'robust_se': model_preferred.bse['treated_post'],
    'ci_lower': model_preferred.conf_int().loc['treated_post', 0],
    'ci_upper': model_preferred.conf_int().loc['treated_post', 1],
    'pvalue': model_preferred.pvalues['treated_post'],
    'tstat': model_preferred.tvalues['treated_post'],
    'sample_size': len(df),
    'treatment_n': int(df['treated'].sum()),
    'control_n': int((1-df['treated']).sum()),
    'pre_period_n': int((df['post']==0).sum()),
    'post_period_n': int((df['post']==1).sum()),
    'raw_did': raw_did,
    'pre_treat_mean': pre_treat,
    'post_treat_mean': post_treat,
    'pre_control_mean': pre_control,
    'post_control_mean': post_control
}

key_df = pd.DataFrame([key_results])
key_df.to_csv('key_results.csv', index=False)
print("Key results saved to: key_results.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nPreferred Estimate: {key_results['preferred_estimate']:.4f}")
print(f"Robust Std. Error:  {key_results['robust_se']:.4f}")
print(f"95% CI:             [{key_results['ci_lower']:.4f}, {key_results['ci_upper']:.4f}]")
print(f"T-statistic:        {key_results['tstat']:.4f}")
print(f"P-value:            {key_results['pvalue']:.6f}")
print(f"Sample Size:        {key_results['sample_size']:,}")
print(f"\nInterpretation: DACA eligibility is associated with a {key_results['preferred_estimate']*100:.2f} percentage point")
print(f"increase in the probability of full-time employment among the treatment group")
print(f"(ages 26-30 as of June 2012) relative to the control group (ages 31-35).")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

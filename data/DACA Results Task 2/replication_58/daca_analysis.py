"""
DACA Replication Study: Effect of DACA on Full-Time Employment
Among Hispanic-Mexican, Mexico-born individuals

Research Question: What was the causal impact of DACA eligibility on full-time employment
(defined as usually working 35+ hours per week)?

Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
Control group: Ages 31-35 at DACA implementation (otherwise would be eligible)
Pre-period: 2006-2011
Post-period: 2013-2016 (2012 excluded due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("="*80)

# Load data in chunks, filtering as we go to reduce memory
print("\n[1] Loading data with filtering...")
data_path = "data/data.csv"

# Columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter
chunks = []
chunk_size = 1000000
total_rows = 0

for chunk in pd.read_csv(data_path, usecols=cols_needed, chunksize=chunk_size):
    total_rows += len(chunk)
    # Filter for Hispanic-Mexican (HISPAN == 1), Mexico-born (BPL == 200), non-citizen (CITIZEN == 3)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations loaded: {total_rows:,}")
print(f"Filtered sample: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: DEFINE DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n[2] Defining DACA eligibility criteria...")

# DACA was implemented on June 15, 2012
# Eligibility requirements:
# 1. Arrived before 16th birthday
# 2. Under age 31 as of June 15, 2012 (i.e., born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (5 years before)
# 4. Present in US on June 15, 2012 and no lawful status

# Calculate age as of June 15, 2012
# BIRTHYR gives birth year, BIRTHQTR gives quarter (1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec)
# June 15 falls in Q2

def calc_age_at_daca(row):
    """Calculate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # Age in years as of June 15, 2012
    age = 2012 - birth_year

    # Adjust for birth quarter (if born after June 15, subtract 1)
    # Q1 (Jan-Mar): person had birthday by June 15
    # Q2 (Apr-Jun): roughly 50% had birthday by June 15
    # Q3 (Jul-Sep): had not had birthday yet
    # Q4 (Oct-Dec): had not had birthday yet

    if birth_qtr in [3, 4]:
        age -= 1
    # For Q2, we'll treat as having reached the birthday (conservative approach)

    return age

df['age_at_daca'] = df.apply(calc_age_at_daca, axis=1)
print(f"Age at DACA range: {df['age_at_daca'].min()} to {df['age_at_daca'].max()}")

# Check if arrived before 16th birthday
# YRIMMIG gives year of immigration
# Person arrived before 16 if: YRIMMIG <= BIRTHYR + 15

def arrived_before_16(row):
    """Check if person arrived in US before their 16th birthday"""
    if pd.isna(row['YRIMMIG']) or row['YRIMMIG'] == 0:
        return False
    return row['YRIMMIG'] <= row['BIRTHYR'] + 15

df['arrived_before_16'] = df.apply(arrived_before_16, axis=1)
print(f"Arrived before age 16: {df['arrived_before_16'].sum():,} ({df['arrived_before_16'].mean()*100:.1f}%)")

# Check continuous residence since June 2007 (5 years before DACA)
# Use YRIMMIG - must have arrived by 2007
def continuous_residence(row):
    """Check if person was in US since June 2007"""
    if pd.isna(row['YRIMMIG']) or row['YRIMMIG'] == 0:
        return False
    return row['YRIMMIG'] <= 2007

df['continuous_residence'] = df.apply(continuous_residence, axis=1)
print(f"Continuous residence since 2007: {df['continuous_residence'].sum():,} ({df['continuous_residence'].mean()*100:.1f}%)")

# ============================================================================
# STEP 3: DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================
print("\n[3] Defining treatment and control groups...")

# Treatment: Ages 26-30 at DACA (June 15, 2012)
# These are the oldest people who were eligible (under 31)
# Control: Ages 31-35 at DACA
# These would have been eligible except for age (too old by 1-5 years)

# Apply additional eligibility criteria (arrived before 16, continuous residence)
# to both groups for comparability

df['eligible_except_age'] = (
    df['arrived_before_16'] &
    df['continuous_residence']
)
print(f"Eligible except for age: {df['eligible_except_age'].sum():,} ({df['eligible_except_age'].mean()*100:.1f}%)")

# Define treatment group (ages 26-30)
df['treatment'] = (
    (df['age_at_daca'] >= 26) &
    (df['age_at_daca'] <= 30) &
    df['eligible_except_age']
)

# Define control group (ages 31-35)
df['control'] = (
    (df['age_at_daca'] >= 31) &
    (df['age_at_daca'] <= 35) &
    df['eligible_except_age']
)

print(f"Treatment group (ages 26-30, eligible): {df['treatment'].sum():,}")
print(f"Control group (ages 31-35, eligible except age): {df['control'].sum():,}")

# Filter to only treatment and control
df_analysis = df[df['treatment'] | df['control']].copy()
print(f"Total analysis sample: {len(df_analysis):,}")

# ============================================================================
# STEP 4: DEFINE PRE AND POST PERIODS
# ============================================================================
print("\n[4] Defining pre and post periods...")

# DACA announced June 15, 2012
# Exclude 2012 because we can't distinguish pre/post in that year
# Pre: 2006-2011
# Post: 2013-2016

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['treat'] = df_analysis['treatment'].astype(int)

# Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

print(f"\nObservations by year:")
print(df_analysis['YEAR'].value_counts().sort_index())

print(f"\nObservations by period:")
print(f"Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# STEP 5: DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n[5] Defining outcome variable...")

# Full-time employment: UHRSWORK >= 35 hours per week
# UHRSWORK is "usual hours worked per week"

print(f"UHRSWORK range: {df_analysis['UHRSWORK'].min()} to {df_analysis['UHRSWORK'].max()}")

# Create full-time indicator
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employment rate: {df_analysis['fulltime'].mean()*100:.1f}%")

# Also create employed indicator for reference
# EMPSTAT: 1 = Employed, 2 = Unemployed, 3 = Not in labor force
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
print(f"Employment rate: {df_analysis['employed'].mean()*100:.1f}%")

# ============================================================================
# STEP 6: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Create summary by group and period
summary = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'SEX': lambda x: (x == 2).mean(),  # Female proportion
    'EDUC': ['mean'],
    'PERWT': ['sum']
}).round(4)

print("\nSummary by Treatment Group and Period:")
print(summary)

# ============================================================================
# STEP 7: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================
print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("="*80)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# Basic DiD regression (unweighted)
print("\n[7.1] Basic DiD (Unweighted):")
model_basic = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print(model_basic.summary().tables[1])

# DiD with survey weights
print("\n[7.2] DiD with Survey Weights (PERWT):")
model_weighted = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_analysis,
                         weights=df_analysis['PERWT']).fit()
print(model_weighted.summary().tables[1])

# DiD with covariates
print("\n[7.3] DiD with Covariates:")

# Create covariate dummies
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

# Age (centered at 30)
df_analysis['age_centered'] = df_analysis['AGE'] - 30

model_covariates = smf.wls(
    'fulltime ~ treat + post + treat_post + female + married + educ_hs + age_centered',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit()
print(model_covariates.summary().tables[1])

# DiD with year fixed effects
print("\n[7.4] DiD with Year Fixed Effects:")
# Create year dummies
for year in df_analysis['YEAR'].unique():
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)

year_vars = ' + '.join([f'year_{y}' for y in sorted(df_analysis['YEAR'].unique())[1:]])  # exclude one for reference
model_yearfe = smf.wls(
    f'fulltime ~ treat + treat_post + {year_vars}',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit()
print(model_yearfe.summary().tables[1])

# DiD with state fixed effects
print("\n[7.5] DiD with State Fixed Effects:")
model_statefe = smf.wls(
    'fulltime ~ treat + post + treat_post + C(STATEFIP)',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
# Only print key coefficients
print(f"treat_post coefficient: {model_statefe.params['treat_post']:.4f}")
print(f"treat_post std err: {model_statefe.bse['treat_post']:.4f}")
print(f"treat_post p-value: {model_statefe.pvalues['treat_post']:.4f}")

# Full model with covariates and fixed effects
print("\n[7.6] Full Model (Covariates + Year FE):")
model_full = smf.wls(
    f'fulltime ~ treat + treat_post + female + married + educ_hs + age_centered + {year_vars}',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_full.params['treat_post']:.4f}")
print(f"treat_post std err: {model_full.bse['treat_post']:.4f}")
print(f"treat_post 95% CI: [{model_full.conf_int().loc['treat_post', 0]:.4f}, {model_full.conf_int().loc['treat_post', 1]:.4f}]")
print(f"treat_post p-value: {model_full.pvalues['treat_post']:.4f}")

# ============================================================================
# STEP 8: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# 8.1 Placebo test - use 2009 as fake treatment year (pre-period only)
print("\n[8.1] Placebo Test (Fake Treatment in 2009):")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['treat'] * df_placebo['post_placebo']

if len(df_placebo) > 100:
    model_placebo = smf.wls(
        'fulltime ~ treat + post_placebo + treat_post_placebo',
        data=df_placebo,
        weights=df_placebo['PERWT']
    ).fit()
    print(f"Placebo DiD coefficient: {model_placebo.params['treat_post_placebo']:.4f}")
    print(f"Placebo p-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# 8.2 Alternative age bands
print("\n[8.2] Alternative Age Bands (24-28 vs 33-37):")
df_alt = df[
    ((df['age_at_daca'] >= 24) & (df['age_at_daca'] <= 28) & df['eligible_except_age']) |
    ((df['age_at_daca'] >= 33) & (df['age_at_daca'] <= 37) & df['eligible_except_age'])
].copy()
df_alt = df_alt[df_alt['YEAR'] != 2012].copy()
df_alt['treat'] = ((df_alt['age_at_daca'] >= 24) & (df_alt['age_at_daca'] <= 28)).astype(int)
df_alt['post'] = (df_alt['YEAR'] >= 2013).astype(int)
df_alt['treat_post'] = df_alt['treat'] * df_alt['post']
df_alt['fulltime'] = (df_alt['UHRSWORK'] >= 35).astype(int)

if len(df_alt) > 100:
    model_alt = smf.wls(
        'fulltime ~ treat + post + treat_post',
        data=df_alt,
        weights=df_alt['PERWT']
    ).fit()
    print(f"Alternative DiD coefficient: {model_alt.params['treat_post']:.4f}")
    print(f"Alternative p-value: {model_alt.pvalues['treat_post']:.4f}")

# 8.3 Gender subgroups
print("\n[8.3] Subgroup Analysis by Gender:")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender].copy()
    if len(df_gender) > 100:
        model_gender = smf.wls(
            'fulltime ~ treat + post + treat_post',
            data=df_gender,
            weights=df_gender['PERWT']
        ).fit()
        print(f"{label}: DiD = {model_gender.params['treat_post']:.4f} (p={model_gender.pvalues['treat_post']:.4f})")

# ============================================================================
# STEP 9: PREFERRED SPECIFICATION
# ============================================================================
print("\n" + "="*80)
print("PREFERRED SPECIFICATION")
print("="*80)

# Preferred: Weighted DiD with year FE and covariates
# This is the model_full specification

preferred_coef = model_weighted.params['treat_post']
preferred_se = model_weighted.bse['treat_post']
preferred_ci = model_weighted.conf_int().loc['treat_post']
preferred_pval = model_weighted.pvalues['treat_post']
n_obs = len(df_analysis)

print(f"\nPreferred Model: Weighted DiD (Basic)")
print(f"Effect size (DiD coefficient): {preferred_coef:.4f}")
print(f"Standard error: {preferred_se:.4f}")
print(f"95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"P-value: {preferred_pval:.4f}")
print(f"Sample size: {n_obs:,}")

# Calculate weighted sample sizes
weighted_n_treat_pre = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)]['PERWT'].sum()
weighted_n_treat_post = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)]['PERWT'].sum()
weighted_n_ctrl_pre = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)]['PERWT'].sum()
weighted_n_ctrl_post = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)]['PERWT'].sum()

print(f"\nWeighted sample sizes:")
print(f"Treatment, Pre:  {weighted_n_treat_pre:,.0f}")
print(f"Treatment, Post: {weighted_n_treat_post:,.0f}")
print(f"Control, Pre:    {weighted_n_ctrl_pre:,.0f}")
print(f"Control, Post:   {weighted_n_ctrl_post:,.0f}")

# ============================================================================
# STEP 10: EXPORT RESULTS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Save key results to file
results = {
    'preferred_effect': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_lower': preferred_ci[0],
    'preferred_ci_upper': preferred_ci[1],
    'preferred_pval': preferred_pval,
    'sample_size': n_obs,
    'weighted_n_treat_pre': weighted_n_treat_pre,
    'weighted_n_treat_post': weighted_n_treat_post,
    'weighted_n_ctrl_pre': weighted_n_ctrl_pre,
    'weighted_n_ctrl_post': weighted_n_ctrl_post
}

# Calculate simple means for the 2x2 table
means = df_analysis.groupby(['treat', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
means.columns = ['treat', 'post', 'mean_fulltime']
print("\nWeighted means for 2x2 table:")
print(means)

# Save detailed results
results['mean_treat_pre'] = means[(means['treat']==1) & (means['post']==0)]['mean_fulltime'].values[0]
results['mean_treat_post'] = means[(means['treat']==1) & (means['post']==1)]['mean_fulltime'].values[0]
results['mean_ctrl_pre'] = means[(means['treat']==0) & (means['post']==0)]['mean_fulltime'].values[0]
results['mean_ctrl_post'] = means[(means['treat']==0) & (means['post']==1)]['mean_fulltime'].values[0]

# Manual DiD calculation for verification
did_manual = (results['mean_treat_post'] - results['mean_treat_pre']) - (results['mean_ctrl_post'] - results['mean_ctrl_pre'])
print(f"\nManual DiD calculation: {did_manual:.4f}")
print(f"Regression DiD: {preferred_coef:.4f}")

# Save results to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# ============================================================================
# STEP 11: ADDITIONAL STATISTICS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("ADDITIONAL STATISTICS FOR REPORT")
print("="*80)

# Covariate balance
print("\nCovariate Balance (Pre-period only):")
df_pre = df_analysis[df_analysis['post'] == 0]
for var in ['AGE', 'female', 'married', 'educ_hs']:
    if var == 'AGE':
        mean_treat = np.average(df_pre[df_pre['treat']==1][var], weights=df_pre[df_pre['treat']==1]['PERWT'])
        mean_ctrl = np.average(df_pre[df_pre['treat']==0][var], weights=df_pre[df_pre['treat']==0]['PERWT'])
    else:
        df_analysis[var] = df_analysis[var].astype(float)
        mean_treat = np.average(df_pre[df_pre['treat']==1][var], weights=df_pre[df_pre['treat']==1]['PERWT'])
        mean_ctrl = np.average(df_pre[df_pre['treat']==0][var], weights=df_pre[df_pre['treat']==0]['PERWT'])
    print(f"{var}: Treatment={mean_treat:.3f}, Control={mean_ctrl:.3f}, Diff={mean_treat-mean_ctrl:.3f}")

# Year-by-year employment rates
print("\nYear-by-year full-time employment rates:")
yearly = df_analysis.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
yearly.columns = ['YEAR', 'treat', 'fulltime_rate']
yearly_pivot = yearly.pivot(index='YEAR', columns='treat', values='fulltime_rate')
yearly_pivot.columns = ['Control', 'Treatment']
yearly_pivot['Difference'] = yearly_pivot['Treatment'] - yearly_pivot['Control']
print(yearly_pivot.round(4))

# Save for plotting
yearly_pivot.to_csv('yearly_fulltime_rates.csv')
print("\nYearly rates saved to yearly_fulltime_rates.csv")

# ============================================================================
# STEP 12: SAVE ALL REGRESSION RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING ALL REGRESSION RESULTS")
print("="*80)

# Create a comprehensive results table
reg_results = []

# Model 1: Basic unweighted
reg_results.append({
    'model': '(1) Basic OLS',
    'coef': model_basic.params['treat_post'],
    'se': model_basic.bse['treat_post'],
    'pval': model_basic.pvalues['treat_post'],
    'ci_lower': model_basic.conf_int().loc['treat_post', 0],
    'ci_upper': model_basic.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'No',
    'covariates': 'No',
    'year_fe': 'No'
})

# Model 2: Weighted
reg_results.append({
    'model': '(2) Weighted',
    'coef': model_weighted.params['treat_post'],
    'se': model_weighted.bse['treat_post'],
    'pval': model_weighted.pvalues['treat_post'],
    'ci_lower': model_weighted.conf_int().loc['treat_post', 0],
    'ci_upper': model_weighted.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'Yes',
    'covariates': 'No',
    'year_fe': 'No'
})

# Model 3: Weighted with covariates
reg_results.append({
    'model': '(3) With Covariates',
    'coef': model_covariates.params['treat_post'],
    'se': model_covariates.bse['treat_post'],
    'pval': model_covariates.pvalues['treat_post'],
    'ci_lower': model_covariates.conf_int().loc['treat_post', 0],
    'ci_upper': model_covariates.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'Yes',
    'covariates': 'Yes',
    'year_fe': 'No'
})

# Model 4: Year FE
reg_results.append({
    'model': '(4) Year FE',
    'coef': model_yearfe.params['treat_post'],
    'se': model_yearfe.bse['treat_post'],
    'pval': model_yearfe.pvalues['treat_post'],
    'ci_lower': model_yearfe.conf_int().loc['treat_post', 0],
    'ci_upper': model_yearfe.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'Yes',
    'covariates': 'No',
    'year_fe': 'Yes'
})

# Model 5: State FE
reg_results.append({
    'model': '(5) State FE',
    'coef': model_statefe.params['treat_post'],
    'se': model_statefe.bse['treat_post'],
    'pval': model_statefe.pvalues['treat_post'],
    'ci_lower': model_statefe.conf_int().loc['treat_post', 0],
    'ci_upper': model_statefe.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'Yes',
    'covariates': 'No',
    'year_fe': 'No (State FE)'
})

# Model 6: Full model
reg_results.append({
    'model': '(6) Full Model',
    'coef': model_full.params['treat_post'],
    'se': model_full.bse['treat_post'],
    'pval': model_full.pvalues['treat_post'],
    'ci_lower': model_full.conf_int().loc['treat_post', 0],
    'ci_upper': model_full.conf_int().loc['treat_post', 1],
    'n': len(df_analysis),
    'weights': 'Yes',
    'covariates': 'Yes',
    'year_fe': 'Yes'
})

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to regression_results.csv")
print(reg_df.to_string(index=False))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

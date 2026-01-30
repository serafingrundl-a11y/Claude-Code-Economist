"""
DACA Replication Study: Effect on Full-Time Employment
=======================================================
Research Question: What was the causal impact of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born individuals?

Author: Replication 74
Date: 2026-01-25
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
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_74")

print("="*80)
print("DACA IMPACT ON FULL-TIME EMPLOYMENT: REPLICATION STUDY")
print("="*80)

# =============================================================================
# STEP 1: Load and Initial Exploration
# =============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

# Load data in chunks due to large file size
# First, let's identify the exact columns we need
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'MARST'
]

print("Loading data (this may take a few minutes)...")

# Read data with specified columns and dtypes
dtype_dict = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'STATEFIP': 'int8',
    'AGE': 'int16',
    'SEX': 'int8',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'LABFORCE': 'int8',
    'UHRSWORK': 'int8',
    'MARST': 'int8'
}

df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtype_dict)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Sample Restriction - Hispanic-Mexican, Mexican-born
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Sample Restriction")
print("="*80)

# Restrict to Hispanic-Mexican (HISPAN == 1)
print(f"\nHISPAN distribution (before restriction):")
print(df['HISPAN'].value_counts().head(10))

# HISPAN = 1 means Mexican
df_hisp = df[df['HISPAN'] == 1].copy()
print(f"\nAfter restricting to Hispanic-Mexican (HISPAN=1): {len(df_hisp):,}")

# Restrict to Mexican-born (BPL == 200)
print(f"\nBPL distribution among Hispanic-Mexican:")
print(df_hisp['BPL'].value_counts().head(10))

df_mex = df_hisp[df_hisp['BPL'] == 200].copy()
print(f"\nAfter restricting to Mexican-born (BPL=200): {len(df_mex):,}")

# Free memory
del df, df_hisp
import gc
gc.collect()

# =============================================================================
# STEP 3: Define Analysis Sample - Non-citizens
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Focus on Non-Citizens")
print("="*80)

print(f"\nCITIZEN distribution among Mexican-born Hispanic-Mexicans:")
print(df_mex['CITIZEN'].value_counts())

# For DACA analysis, focus on non-citizens (CITIZEN == 3)
# Note: We cannot distinguish documented vs undocumented
df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"\nAfter restricting to non-citizens (CITIZEN=3): {len(df_noncit):,}")

# =============================================================================
# STEP 4: Define DACA Eligibility
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Define DACA Eligibility Criteria")
print("="*80)

# DACA Eligibility Criteria:
# 1. Arrived in US before 16th birthday: (YRIMMIG - BIRTHYR) < 16
# 2. Born after June 15, 1981: BIRTHYR >= 1982 (conservative)
#    Or BIRTHYR == 1981 and BIRTHQTR >= 2 (April-June or later)
# 3. In US since June 15, 2007: YRIMMIG <= 2007
# 4. Non-citizen (already restricted above)

# Calculate age at immigration
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# Filter out observations with missing/invalid immigration year
df_noncit = df_noncit[df_noncit['YRIMMIG'] > 0].copy()
print(f"After removing missing YRIMMIG: {len(df_noncit):,}")

# Define eligibility components
# 1. Arrived before age 16
df_noncit['arrived_young'] = (df_noncit['age_at_immig'] < 16) & (df_noncit['age_at_immig'] >= 0)

# 2. Born after June 15, 1981 (not yet 31 on June 15, 2012)
df_noncit['young_enough'] = (df_noncit['BIRTHYR'] >= 1982) | \
                            ((df_noncit['BIRTHYR'] == 1981) & (df_noncit['BIRTHQTR'] >= 2))

# 3. In US since June 15, 2007 (continuous residence)
df_noncit['in_us_since_2007'] = df_noncit['YRIMMIG'] <= 2007

# Combined DACA eligibility (for post-2012 periods)
df_noncit['daca_eligible'] = (df_noncit['arrived_young'] &
                               df_noncit['young_enough'] &
                               df_noncit['in_us_since_2007'])

print(f"\nDACA eligibility components:")
print(f"  Arrived before age 16: {df_noncit['arrived_young'].sum():,} ({df_noncit['arrived_young'].mean()*100:.1f}%)")
print(f"  Born after mid-1981: {df_noncit['young_enough'].sum():,} ({df_noncit['young_enough'].mean()*100:.1f}%)")
print(f"  In US since 2007: {df_noncit['in_us_since_2007'].sum():,} ({df_noncit['in_us_since_2007'].mean()*100:.1f}%)")
print(f"  DACA eligible (all criteria): {df_noncit['daca_eligible'].sum():,} ({df_noncit['daca_eligible'].mean()*100:.1f}%)")

# =============================================================================
# STEP 5: Define Time Periods and Outcome
# =============================================================================
print("\n" + "="*80)
print("STEP 5: Define Time Periods and Outcome Variable")
print("="*80)

# Time periods: Pre (2006-2011), Post (2013-2016)
# Exclude 2012 due to mid-year implementation
df_analysis = df_noncit[df_noncit['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Outcome: Full-time employment (usually working 35+ hours per week)
# Only defined for employed individuals, but we'll define for all and condition on labor force
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# In labor force
df_analysis['in_labor_force'] = (df_analysis['LABFORCE'] == 2).astype(int)

print(f"\nYear distribution:")
print(df_analysis.groupby('YEAR').size())

print(f"\nPre/Post distribution:")
print(df_analysis['post'].value_counts())

# =============================================================================
# STEP 6: Restrict to Working-Age Population
# =============================================================================
print("\n" + "="*80)
print("STEP 6: Working-Age Sample (18-40)")
print("="*80)

# Restrict to ages 18-40 (relevant for DACA age range)
df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 40)].copy()
print(f"After restricting to ages 18-40: {len(df_analysis):,}")

# Summary by eligibility and period
print(f"\nSample by DACA eligibility and period:")
print(df_analysis.groupby(['daca_eligible', 'post']).size().unstack())

# =============================================================================
# STEP 7: Create Control Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Create Control Variables")
print("="*80)

# Education categories (6 bins = 5 labels)
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUCD'],
                                  bins=[0, 25, 62, 71, 101, 116, 999],
                                  labels=['no_school', 'less_than_hs', 'hs_grad', 'some_college', 'college', 'grad'])

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Married indicator
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Years in US at time of survey
df_analysis['years_in_us'] = df_analysis['YEAR'] - df_analysis['YRIMMIG']

print("Control variables created:")
print(f"  - Education categories")
print(f"  - Female indicator")
print(f"  - Married indicator")
print(f"  - Age and age squared")
print(f"  - Years in US")

# =============================================================================
# STEP 8: Descriptive Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 8: Descriptive Statistics")
print("="*80)

# Descriptive stats by eligibility group
print("\n--- Pre-Period (2006-2011) Characteristics ---")
pre_stats = df_analysis[df_analysis['post'] == 0].groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'years_in_us': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'in_labor_force': 'mean',
    'PERWT': 'sum'
}).round(3)
pre_stats.columns = ['Mean Age', 'Female', 'Married', 'Years in US',
                     'Employed', 'Full-time', 'In LF', 'Pop Weight']
print(pre_stats)

print("\n--- Post-Period (2013-2016) Characteristics ---")
post_stats = df_analysis[df_analysis['post'] == 1].groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'years_in_us': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'in_labor_force': 'mean',
    'PERWT': 'sum'
}).round(3)
post_stats.columns = ['Mean Age', 'Female', 'Married', 'Years in US',
                      'Employed', 'Full-time', 'In LF', 'Pop Weight']
print(post_stats)

# =============================================================================
# STEP 9: Main Difference-in-Differences Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 9: Difference-in-Differences Analysis")
print("="*80)

# Create interaction term
df_analysis['eligible_x_post'] = df_analysis['daca_eligible'].astype(int) * df_analysis['post']

# Convert boolean to int for regression
df_analysis['eligible'] = df_analysis['daca_eligible'].astype(int)

# Model 1: Basic DiD
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.wls('fulltime ~ eligible + post + eligible_x_post',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ eligible + post + eligible_x_post + AGE + age_sq + female + married',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with full controls including state and year FE
print("\n--- Model 3: DiD with State and Year Fixed Effects ---")
# Create year dummies (excluding base year)
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
df_analysis['state_factor'] = pd.Categorical(df_analysis['STATEFIP'])

model3 = smf.wls('fulltime ~ eligible + eligible_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Print key coefficients
print(f"\nKey coefficients from Model 3:")
print(f"DACA Eligible: {model3.params['eligible']:.4f} (SE: {model3.bse['eligible']:.4f})")
print(f"Eligible x Post (DiD): {model3.params['eligible_x_post']:.4f} (SE: {model3.bse['eligible_x_post']:.4f})")
print(f"  t-stat: {model3.tvalues['eligible_x_post']:.2f}, p-value: {model3.pvalues['eligible_x_post']:.4f}")

# =============================================================================
# STEP 10: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 10: Robustness Checks")
print("="*80)

# Robustness 1: Employment (any hours) instead of full-time
print("\n--- Robustness 1: Effect on Employment (Any Work) ---")
model_emp = smf.wls('employed ~ eligible + eligible_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                    data=df_analysis,
                    weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD Effect on Employment: {model_emp.params['eligible_x_post']:.4f} (SE: {model_emp.bse['eligible_x_post']:.4f})")

# Robustness 2: Among those in labor force only
print("\n--- Robustness 2: Full-time Among Labor Force Participants ---")
df_lf = df_analysis[df_analysis['in_labor_force'] == 1].copy()
model_lf = smf.wls('fulltime ~ eligible + eligible_x_post + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                   data=df_lf,
                   weights=df_lf['PERWT']).fit(cov_type='HC1')
print(f"DiD Effect (LF only): {model_lf.params['eligible_x_post']:.4f} (SE: {model_lf.bse['eligible_x_post']:.4f})")

# Robustness 3: By gender
print("\n--- Robustness 3: Effect by Gender ---")
df_male = df_analysis[df_analysis['female'] == 0].copy()
df_female = df_analysis[df_analysis['female'] == 1].copy()

model_male = smf.wls('fulltime ~ eligible + eligible_x_post + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
                     data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ eligible + eligible_x_post + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
                       data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"DiD Effect - Males: {model_male.params['eligible_x_post']:.4f} (SE: {model_male.bse['eligible_x_post']:.4f})")
print(f"DiD Effect - Females: {model_female.params['eligible_x_post']:.4f} (SE: {model_female.bse['eligible_x_post']:.4f})")

# =============================================================================
# STEP 11: Event Study / Pre-Trend Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 11: Event Study Analysis (Pre-Trend Check)")
print("="*80)

# Create year-specific interactions
for year in df_analysis['YEAR'].unique():
    df_analysis[f'eligible_x_{year}'] = (df_analysis['eligible'] * (df_analysis['YEAR'] == year)).astype(int)

# Base year: 2011 (last pre-treatment year)
year_interactions = [f'eligible_x_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2011]

formula_event = 'fulltime ~ eligible + ' + ' + '.join(year_interactions) + ' + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)'
model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Base Year: 2011):")
print("-" * 50)
for year in sorted(df_analysis['YEAR'].unique()):
    if year != 2011:
        coef_name = f'eligible_x_{year}'
        print(f"{year}: {model_event.params[coef_name]:7.4f} (SE: {model_event.bse[coef_name]:.4f})")

# =============================================================================
# STEP 12: Simple 2x2 DiD Table
# =============================================================================
print("\n" + "="*80)
print("STEP 12: Simple 2x2 Difference-in-Differences Table")
print("="*80)

# Calculate weighted means
def weighted_mean(data, values, weights):
    return np.average(data[values], weights=data[weights])

# 2x2 table
did_table = pd.DataFrame(index=['Eligible', 'Not Eligible', 'Difference'],
                         columns=['Pre (2006-2011)', 'Post (2013-2016)', 'Difference'])

# Eligible Pre
elig_pre = df_analysis[(df_analysis['eligible'] == 1) & (df_analysis['post'] == 0)]
did_table.loc['Eligible', 'Pre (2006-2011)'] = weighted_mean(elig_pre, 'fulltime', 'PERWT')

# Eligible Post
elig_post = df_analysis[(df_analysis['eligible'] == 1) & (df_analysis['post'] == 1)]
did_table.loc['Eligible', 'Post (2013-2016)'] = weighted_mean(elig_post, 'fulltime', 'PERWT')

# Not Eligible Pre
notelig_pre = df_analysis[(df_analysis['eligible'] == 0) & (df_analysis['post'] == 0)]
did_table.loc['Not Eligible', 'Pre (2006-2011)'] = weighted_mean(notelig_pre, 'fulltime', 'PERWT')

# Not Eligible Post
notelig_post = df_analysis[(df_analysis['eligible'] == 0) & (df_analysis['post'] == 1)]
did_table.loc['Not Eligible', 'Post (2013-2016)'] = weighted_mean(notelig_post, 'fulltime', 'PERWT')

# Calculate differences
did_table['Difference'] = did_table['Post (2013-2016)'].astype(float) - did_table['Pre (2006-2011)'].astype(float)
did_table.loc['Difference', 'Pre (2006-2011)'] = did_table.loc['Eligible', 'Pre (2006-2011)'] - did_table.loc['Not Eligible', 'Pre (2006-2011)']
did_table.loc['Difference', 'Post (2013-2016)'] = did_table.loc['Eligible', 'Post (2013-2016)'] - did_table.loc['Not Eligible', 'Post (2013-2016)']
did_table.loc['Difference', 'Difference'] = did_table.loc['Eligible', 'Difference'] - did_table.loc['Not Eligible', 'Difference']

print("\nFull-Time Employment Rates (Weighted):")
print(did_table.round(4))
print(f"\nDiD Estimate: {did_table.loc['Difference', 'Difference']:.4f}")

# =============================================================================
# STEP 13: Save Results for Report
# =============================================================================
print("\n" + "="*80)
print("STEP 13: Summary of Key Results")
print("="*80)

# Save key results
results_dict = {
    'n_total': len(df_analysis),
    'n_eligible': df_analysis['eligible'].sum(),
    'n_not_eligible': (df_analysis['eligible'] == 0).sum(),
    'did_estimate_basic': model1.params['eligible_x_post'],
    'did_se_basic': model1.bse['eligible_x_post'],
    'did_estimate_controls': model2.params['eligible_x_post'],
    'did_se_controls': model2.bse['eligible_x_post'],
    'did_estimate_full': model3.params['eligible_x_post'],
    'did_se_full': model3.bse['eligible_x_post'],
    'did_pvalue': model3.pvalues['eligible_x_post']
}

print("\n=== PREFERRED ESTIMATE ===")
print(f"DiD Effect (Full Model with State and Year FE):")
print(f"  Coefficient: {results_dict['did_estimate_full']:.4f}")
print(f"  Std. Error:  {results_dict['did_se_full']:.4f}")
print(f"  95% CI:      [{results_dict['did_estimate_full'] - 1.96*results_dict['did_se_full']:.4f}, {results_dict['did_estimate_full'] + 1.96*results_dict['did_se_full']:.4f}]")
print(f"  p-value:     {results_dict['did_pvalue']:.4f}")
print(f"  Sample Size: {results_dict['n_total']:,}")

# =============================================================================
# STEP 14: Export Tables for LaTeX
# =============================================================================
print("\n" + "="*80)
print("STEP 14: Export Results")
print("="*80)

# Save summary statistics
summary_pre = df_analysis[df_analysis['post'] == 0].groupby('eligible').agg({
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'years_in_us': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'in_labor_force': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
summary_pre.to_csv('summary_stats_pre.csv')

summary_post = df_analysis[df_analysis['post'] == 1].groupby('eligible').agg({
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'years_in_us': 'mean',
    'employed': 'mean',
    'fulltime': 'mean',
    'in_labor_force': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
summary_post.to_csv('summary_stats_post.csv')

# Save DiD table
did_table.to_csv('did_table.csv')

# Save event study coefficients
event_coefs = []
for year in sorted(df_analysis['YEAR'].unique()):
    if year != 2011:
        coef_name = f'eligible_x_{year}'
        event_coefs.append({
            'Year': year,
            'Coefficient': model_event.params[coef_name],
            'SE': model_event.bse[coef_name],
            'CI_lower': model_event.params[coef_name] - 1.96 * model_event.bse[coef_name],
            'CI_upper': model_event.params[coef_name] + 1.96 * model_event.bse[coef_name]
        })
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_coefs.csv', index=False)

# Save regression results
reg_results = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Full (State/Year FE)'],
    'DiD_Coef': [model1.params['eligible_x_post'], model2.params['eligible_x_post'], model3.params['eligible_x_post']],
    'SE': [model1.bse['eligible_x_post'], model2.bse['eligible_x_post'], model3.bse['eligible_x_post']],
    'p_value': [model1.pvalues['eligible_x_post'], model2.pvalues['eligible_x_post'], model3.pvalues['eligible_x_post']],
    'N': [model1.nobs, model2.nobs, model3.nobs],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared]
})
reg_results.to_csv('regression_results.csv', index=False)

# Robustness results
robust_results = pd.DataFrame({
    'Specification': ['Employment (any)', 'Full-time (LF only)', 'Males only', 'Females only'],
    'DiD_Coef': [model_emp.params['eligible_x_post'], model_lf.params['eligible_x_post'],
                 model_male.params['eligible_x_post'], model_female.params['eligible_x_post']],
    'SE': [model_emp.bse['eligible_x_post'], model_lf.bse['eligible_x_post'],
           model_male.bse['eligible_x_post'], model_female.bse['eligible_x_post']],
    'N': [model_emp.nobs, model_lf.nobs, model_male.nobs, model_female.nobs]
})
robust_results.to_csv('robustness_results.csv', index=False)

# Year-by-group means for plotting
yearly_means = df_analysis.groupby(['YEAR', 'eligible']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()
yearly_means.to_csv('yearly_means.csv', index=False)

print("Results exported to CSV files.")
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

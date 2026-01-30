"""
DACA Effect on Full-Time Employment Replication Analysis
Research Question: Effect of DACA eligibility on probability of working 35+ hours/week
among Hispanic-Mexican, Mexican-born individuals in the United States.

This script performs a difference-in-differences analysis using ACS data 2006-2016.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA EFFECT ON FULL-TIME EMPLOYMENT - REPLICATION ANALYSIS")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA IN CHUNKS (file is very large)
# ============================================================================
print("\n[1] Loading data in chunks (selecting Hispanic-Mexican, Mexican-born)...")
data_path = "data/data.csv"

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter to Hispanic-Mexican, Mexico-born
chunks = []
chunk_size = 500000
total_rows = 0

for chunk in pd.read_csv(data_path, usecols=cols_needed, chunksize=chunk_size, low_memory=False):
    total_rows += len(chunk)
    # Filter: HISPAN == 1 (Mexican) and BPL == 200 (Mexico)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"    Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

df = pd.concat(chunks, ignore_index=True)
print(f"    Total rows in original data: {total_rows:,}")
print(f"    Rows after filtering (Hispanic-Mexican, Mexican-born): {len(df):,}")

# ============================================================================
# STEP 2: EXCLUDE 2012 (ambiguous pre/post period)
# ============================================================================
print("\n[2] Excluding 2012 (DACA implemented mid-year)...")
df = df[df['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df):,}")

# ============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY CRITERIA
# ============================================================================
print("\n[3] Defining DACA eligibility criteria...")

# Key eligibility criteria from instructions:
# 1. Arrived before 16th birthday
# 2. Not yet 31 as of June 15, 2012 (birth year >= 1982 to be under 31 on June 15, 2012)
# 3. Continuous presence since June 15, 2007 (at least 5 years before 2012)
# 4. Not a citizen (CITIZEN == 3 means "Not a citizen")

# Calculate age at arrival in the US
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Create eligibility indicators
df['arrived_before_16'] = df['age_at_immig'] < 16
df['under_31_in_2012'] = df['BIRTHYR'] > 1981  # Born after June 15, 1981
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007
df['not_citizen'] = df['CITIZEN'] == 3

# DACA eligible if all criteria met
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_in_2012'] &
    df['in_us_since_2007'] &
    df['not_citizen']
).astype(int)

print(f"    DACA eligible: {df['daca_eligible'].sum():,}")
print(f"    Not eligible: {(df['daca_eligible']==0).sum():,}")

# ============================================================================
# STEP 4: DEFINE OUTCOME VARIABLE - FULL-TIME EMPLOYMENT
# ============================================================================
print("\n[4] Defining outcome variable: Full-time employment (35+ hours/week)...")

# Full-time employment: UHRSWORK >= 35
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

print(f"    Full-time employed (35+ hrs): {df['fulltime'].sum():,}")
print(f"    Not full-time: {(df['fulltime']==0).sum():,}")

# ============================================================================
# STEP 5: DEFINE POST-TREATMENT INDICATOR
# ============================================================================
print("\n[5] Defining post-treatment indicator...")

# Post period: 2013-2016 (after DACA implementation)
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"    Pre-period observations (2006-2011): {(df['post']==0).sum():,}")
print(f"    Post-period observations (2013-2016): {(df['post']==1).sum():,}")

# ============================================================================
# STEP 6: CREATE WORKING AGE SAMPLE (16-65)
# ============================================================================
print("\n[6] Restricting to working-age population (16-65)...")

df = df[(df['AGE'] >= 16) & (df['AGE'] <= 65)].copy()
print(f"    Working-age sample: {len(df):,}")

# ============================================================================
# STEP 7: FURTHER SAMPLE RESTRICTIONS
# ============================================================================
print("\n[7] Additional sample restrictions...")

# Keep non-citizens for cleaner comparison
df = df[df['CITIZEN'] == 3].copy()
print(f"    Non-citizens only: {len(df):,}")

# Filter out observations with missing key variables
df = df.dropna(subset=['UHRSWORK', 'YRIMMIG', 'BIRTHYR', 'AGE', 'SEX', 'EDUC'])
print(f"    After dropping missing values: {len(df):,}")

# Filter out invalid year of immigration (0 = N/A)
df = df[df['YRIMMIG'] > 0].copy()
print(f"    After filtering invalid immigration year: {len(df):,}")

# ============================================================================
# STEP 8: CREATE CONTROL VARIABLES
# ============================================================================
print("\n[8] Creating control variables...")

# Age squared
df['age_sq'] = df['AGE'] ** 2

# Sex (binary: male = 1)
df['male'] = (df['SEX'] == 1).astype(int)

# Marital status (married = 1)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']
df.loc[df['years_in_us'] < 0, 'years_in_us'] = 0

print(f"    Final sample size for analysis: {len(df):,}")

# ============================================================================
# STEP 9: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[9] Descriptive Statistics...")
print("-" * 70)

# Summary by eligibility and period
summary = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'EDUC': 'mean',
    'years_in_us': 'mean'
}).round(3)

print("\nSummary Statistics by DACA Eligibility and Time Period:")
print(summary)

# Calculate simple DiD estimate
mean_pre_treat = df[(df['daca_eligible']==1) & (df['post']==0)]['fulltime'].mean()
mean_post_treat = df[(df['daca_eligible']==1) & (df['post']==1)]['fulltime'].mean()
mean_pre_ctrl = df[(df['daca_eligible']==0) & (df['post']==0)]['fulltime'].mean()
mean_post_ctrl = df[(df['daca_eligible']==0) & (df['post']==1)]['fulltime'].mean()

simple_did = (mean_post_treat - mean_pre_treat) - (mean_post_ctrl - mean_pre_ctrl)

print(f"\n{'='*70}")
print("SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print(f"{'='*70}")
print(f"Treatment group (DACA eligible):")
print(f"    Pre-period mean:  {mean_pre_treat:.4f}")
print(f"    Post-period mean: {mean_post_treat:.4f}")
print(f"    Change:           {mean_post_treat - mean_pre_treat:.4f}")
print(f"\nControl group (Not DACA eligible):")
print(f"    Pre-period mean:  {mean_pre_ctrl:.4f}")
print(f"    Post-period mean: {mean_post_ctrl:.4f}")
print(f"    Change:           {mean_post_ctrl - mean_pre_ctrl:.4f}")
print(f"\nSimple DiD estimate: {simple_did:.4f}")

# ============================================================================
# STEP 10: REGRESSION ANALYSIS
# ============================================================================
print(f"\n{'='*70}")
print("REGRESSION ANALYSIS")
print(f"{'='*70}")

# Create interaction term
df['did'] = df['daca_eligible'] * df['post']

# Create year dummies for year fixed effects
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True).astype(float)
for col in year_dummies.columns:
    df[col] = year_dummies[col].values

# Create state dummies for state fixed effects
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True).astype(float)
for col in state_dummies.columns:
    df[col] = state_dummies[col].values

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
X1 = df[['daca_eligible', 'post', 'did']]
X1 = sm.add_constant(X1)
y = df['fulltime']
weights = df['PERWT']

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient: {model1.params['did']:.5f}")
print(f"Standard error:  {model1.bse['did']:.5f}")
print(f"95% CI: [{model1.conf_int().loc['did', 0]:.5f}, {model1.conf_int().loc['did', 1]:.5f}]")
print(f"p-value: {model1.pvalues['did']:.5f}")
print(f"N = {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
X2_vars = ['daca_eligible', 'post', 'did', 'AGE', 'age_sq', 'male', 'married', 'years_in_us']
X2 = df[X2_vars]
X2 = sm.add_constant(X2)

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient: {model2.params['did']:.5f}")
print(f"Standard error:  {model2.bse['did']:.5f}")
print(f"95% CI: [{model2.conf_int().loc['did', 0]:.5f}, {model2.conf_int().loc['did', 1]:.5f}]")
print(f"p-value: {model2.pvalues['did']:.5f}")
print(f"N = {int(model2.nobs):,}")

# Model 3: DiD with controls and year fixed effects
print("\n--- Model 3: DiD with controls and year fixed effects ---")
year_cols = [c for c in df.columns if c.startswith('year_')]
X3_vars = ['daca_eligible', 'did', 'AGE', 'age_sq', 'male', 'married', 'years_in_us'] + year_cols
X3 = df[X3_vars]
X3 = sm.add_constant(X3)

model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['did']:.5f}")
print(f"Standard error:  {model3.bse['did']:.5f}")
print(f"95% CI: [{model3.conf_int().loc['did', 0]:.5f}, {model3.conf_int().loc['did', 1]:.5f}]")
print(f"p-value: {model3.pvalues['did']:.5f}")
print(f"N = {int(model3.nobs):,}")

# Model 4: Full model with state and year fixed effects
print("\n--- Model 4: Full model with state and year fixed effects ---")
state_cols = [c for c in df.columns if c.startswith('state_')]
X4_vars = ['daca_eligible', 'did', 'AGE', 'age_sq', 'male', 'married', 'years_in_us'] + year_cols + state_cols
X4 = df[X4_vars]
X4 = sm.add_constant(X4)

model4 = sm.WLS(y, X4, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['did']:.5f}")
print(f"Standard error:  {model4.bse['did']:.5f}")
print(f"95% CI: [{model4.conf_int().loc['did', 0]:.5f}, {model4.conf_int().loc['did', 1]:.5f}]")
print(f"p-value: {model4.pvalues['did']:.5f}")
print(f"N = {int(model4.nobs):,}")

# ============================================================================
# STEP 11: ROBUSTNESS CHECKS
# ============================================================================
print(f"\n{'='*70}")
print("ROBUSTNESS CHECKS")
print(f"{'='*70}")

# Robustness 1: Clustered standard errors by state
print("\n--- Robustness 1: Clustered standard errors by state ---")
model_clustered = sm.WLS(y, X4, weights=weights).fit(cov_type='cluster',
                                                      cov_kwds={'groups': df['STATEFIP']})
print(f"DiD coefficient: {model_clustered.params['did']:.5f}")
print(f"Clustered SE:    {model_clustered.bse['did']:.5f}")
print(f"95% CI: [{model_clustered.conf_int().loc['did', 0]:.5f}, {model_clustered.conf_int().loc['did', 1]:.5f}]")

# Robustness 2: Pre-trend analysis
print("\n--- Robustness 2: Pre-trend analysis (2006-2011) ---")
df_pre = df[df['post'] == 0].copy()
for yr in [2007, 2008, 2009, 2010, 2011]:
    df_pre[f'year_{yr}'] = (df_pre['YEAR'] == yr).astype(int)
    df_pre[f'elig_{yr}'] = df_pre['daca_eligible'] * df_pre[f'year_{yr}']

X_pretrend_vars = ['daca_eligible', 'year_2007', 'year_2008', 'year_2009', 'year_2010', 'year_2011',
                   'elig_2007', 'elig_2008', 'elig_2009', 'elig_2010', 'elig_2011',
                   'AGE', 'age_sq', 'male', 'married', 'years_in_us']
X_pretrend = df_pre[X_pretrend_vars]
X_pretrend = sm.add_constant(X_pretrend)
y_pre = df_pre['fulltime']
weights_pre = df_pre['PERWT']

model_pretrend = sm.WLS(y_pre, X_pretrend, weights=weights_pre).fit(cov_type='HC1')
print("Pre-treatment year interactions (relative to 2006):")
for year in ['2007', '2008', '2009', '2010', '2011']:
    var = f'elig_{year}'
    print(f"  Eligible x {year}: {model_pretrend.params[var]:.5f} (SE: {model_pretrend.bse[var]:.5f}, p: {model_pretrend.pvalues[var]:.3f})")

# Robustness 3: Alternative control group - recently arrived
print("\n--- Robustness 3: Comparison with recently arrived immigrants ---")
df['recent_arrival'] = (df['YRIMMIG'] > 2007).astype(int)
df_alt_control = df[((df['daca_eligible']==1) | (df['recent_arrival']==1))].copy()
df_alt_control['did_alt'] = df_alt_control['daca_eligible'] * df_alt_control['post']

X_alt = df_alt_control[['daca_eligible', 'post', 'did_alt', 'AGE', 'age_sq', 'male', 'married', 'years_in_us']]
X_alt = sm.add_constant(X_alt)
y_alt = df_alt_control['fulltime']
weights_alt = df_alt_control['PERWT']

model_alt = sm.WLS(y_alt, X_alt, weights=weights_alt).fit(cov_type='HC1')
print(f"DiD coefficient (vs recent arrivals): {model_alt.params['did_alt']:.5f}")
print(f"Standard error: {model_alt.bse['did_alt']:.5f}")
print(f"95% CI: [{model_alt.conf_int().loc['did_alt', 0]:.5f}, {model_alt.conf_int().loc['did_alt', 1]:.5f}]")

# Robustness 4: By gender subgroups
print("\n--- Robustness 4: Subgroup analysis by gender ---")
for gender, label in [(1, 'Male'), (0, 'Female')]:
    df_gender = df[df['male'] == gender]
    X_g = df_gender[['daca_eligible', 'post', 'did', 'AGE', 'age_sq', 'married', 'years_in_us']]
    X_g = sm.add_constant(X_g)
    y_g = df_gender['fulltime']
    w_g = df_gender['PERWT']
    model_g = sm.WLS(y_g, X_g, weights=w_g).fit(cov_type='HC1')
    print(f"  {label}: DiD = {model_g.params['did']:.5f} (SE: {model_g.bse['did']:.5f}, p: {model_g.pvalues['did']:.3f})")

# Robustness 5: Dynamic effects (event study)
print("\n--- Robustness 5: Event study (dynamic effects) ---")
# Create year-specific treatment effects
for yr in df['YEAR'].unique():
    df[f'year_{yr}'] = (df['YEAR'] == yr).astype(int)
    df[f'elig_x_{yr}'] = df['daca_eligible'] * df[f'year_{yr}']

event_vars = ['daca_eligible'] + [f'year_{yr}' for yr in sorted(df['YEAR'].unique()) if yr != 2006]
event_vars += [f'elig_x_{yr}' for yr in sorted(df['YEAR'].unique()) if yr != 2011]  # 2011 is reference
event_vars += ['AGE', 'age_sq', 'male', 'married', 'years_in_us']

X_event = df[[v for v in event_vars if v in df.columns]]
X_event = sm.add_constant(X_event)
model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='HC1')

print("Event study coefficients (relative to 2011):")
for yr in sorted(df['YEAR'].unique()):
    if yr != 2011:
        var = f'elig_x_{yr}'
        if var in model_event.params.index:
            print(f"  {yr}: {model_event.params[var]:.5f} (SE: {model_event.bse[var]:.5f})")

# ============================================================================
# STEP 12: SAVE RESULTS FOR REPORT
# ============================================================================
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*70}")

print(f"\nPreferred Specification: Model 4 (Full DiD with state and year FE)")
print(f"{'='*70}")
print(f"Effect estimate:     {model4.params['did']:.5f}")
print(f"Standard error:      {model4.bse['did']:.5f}")
print(f"95% CI:              [{model4.conf_int().loc['did', 0]:.5f}, {model4.conf_int().loc['did', 1]:.5f}]")
print(f"p-value:             {model4.pvalues['did']:.5f}")
print(f"Sample size:         {len(df):,}")

# Count by groups
n_treat_pre = len(df[(df['daca_eligible']==1) & (df['post']==0)])
n_treat_post = len(df[(df['daca_eligible']==1) & (df['post']==1)])
n_ctrl_pre = len(df[(df['daca_eligible']==0) & (df['post']==0)])
n_ctrl_post = len(df[(df['daca_eligible']==0) & (df['post']==1)])

print(f"\nSample breakdown:")
print(f"  Treatment group, pre-DACA:  {n_treat_pre:,}")
print(f"  Treatment group, post-DACA: {n_treat_post:,}")
print(f"  Control group, pre-DACA:    {n_ctrl_pre:,}")
print(f"  Control group, post-DACA:   {n_ctrl_post:,}")

# Save key results to CSV for LaTeX tables
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Year FE', 'Year + State FE', 'Clustered SE'],
    'Coefficient': [model1.params['did'], model2.params['did'], model3.params['did'],
                    model4.params['did'], model_clustered.params['did']],
    'SE': [model1.bse['did'], model2.bse['did'], model3.bse['did'],
           model4.bse['did'], model_clustered.bse['did']],
    'CI_lower': [model1.conf_int().loc['did', 0], model2.conf_int().loc['did', 0],
                 model3.conf_int().loc['did', 0], model4.conf_int().loc['did', 0],
                 model_clustered.conf_int().loc['did', 0]],
    'CI_upper': [model1.conf_int().loc['did', 1], model2.conf_int().loc['did', 1],
                 model3.conf_int().loc['did', 1], model4.conf_int().loc['did', 1],
                 model_clustered.conf_int().loc['did', 1]],
    'p_value': [model1.pvalues['did'], model2.pvalues['did'], model3.pvalues['did'],
                model4.pvalues['did'], model_clustered.pvalues['did']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model_clustered.nobs)]
})
results_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to regression_results.csv")

# Save descriptive statistics
desc_stats = df.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'male': 'mean',
    'married': 'mean',
    'years_in_us': ['mean', 'std']
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

# Save event study results
event_results = []
for yr in sorted(df['YEAR'].unique()):
    if yr != 2011:
        var = f'elig_x_{yr}'
        if var in model_event.params.index:
            event_results.append({
                'year': yr,
                'coefficient': model_event.params[var],
                'se': model_event.bse[var],
                'ci_lower': model_event.conf_int().loc[var, 0],
                'ci_upper': model_event.conf_int().loc[var, 1]
            })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save pretrend results
pretrend_results = []
for year in ['2007', '2008', '2009', '2010', '2011']:
    var = f'elig_{year}'
    pretrend_results.append({
        'year': year,
        'coefficient': model_pretrend.params[var],
        'se': model_pretrend.bse[var],
        'p_value': model_pretrend.pvalues[var]
    })
pretrend_df = pd.DataFrame(pretrend_results)
pretrend_df.to_csv('pretrend_results.csv', index=False)
print("Pre-trend results saved to pretrend_results.csv")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")

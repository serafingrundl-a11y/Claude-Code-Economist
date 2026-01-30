"""
DACA Replication Study - Main Analysis Script
Research Question: Impact of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US (2013-2016)

Author: Replication Study
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/data.csv'
CHUNK_SIZE = 500000  # Process in chunks due to large file size

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("\n[STEP 1] Loading and filtering data...")
print("Target population: Hispanic-Mexican, Mexico-born, non-citizens")

# Columns we need for analysis
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'MARST',
           'EMPSTAT', 'UHRSWORK', 'STATEFIP']

# Read data in chunks and filter
filtered_chunks = []
total_rows = 0
filtered_rows = 0

for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=CHUNK_SIZE):
    total_rows += len(chunk)

    # Filter criteria:
    # 1. Hispanic-Mexican (HISPAN == 1)
    # 2. Born in Mexico (BPL == 200)
    # 3. Non-citizen (CITIZEN == 3) - proxy for undocumented

    mask = (chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)
    filtered = chunk[mask].copy()
    filtered_rows += len(filtered)

    if len(filtered) > 0:
        filtered_chunks.append(filtered)

    if total_rows % 5000000 == 0:
        print(f"  Processed {total_rows:,} rows, kept {filtered_rows:,}")

df = pd.concat(filtered_chunks, ignore_index=True)
print(f"\nTotal rows processed: {total_rows:,}")
print(f"Rows matching sample criteria: {len(df):,}")

# ============================================================================
# STEP 2: Create Analysis Variables
# ============================================================================
print("\n[STEP 2] Creating analysis variables...")

# Full-time employment outcome (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator (EMPSTAT == 1 is employed)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# DACA implementation date: June 15, 2012
# Key eligibility criteria:
# 1. Arrived before 16th birthday
# 2. Under 31 as of June 15, 2012
# 3. In US continuously since June 15, 2007
# 4. Present in US on June 15, 2012

# For age at arrival, we need: arrived before turning 16
# age_at_arrival = YRIMMIG - BIRTHYR approximately
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Age as of June 15, 2012
# Use BIRTHYR and approximate based on survey year
df['age_june2012'] = 2012 - df['BIRTHYR']

# For more precision with birth quarter:
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born Q1-Q2 (before June), they would have had their birthday by June 15
# If born Q3-Q4 (after June), they wouldn't have had their birthday yet
df['age_june2012_adj'] = df['age_june2012'].copy()
# Those born in Q3 or Q4 would be one year younger as of June 15
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012_adj'] = df['age_june2012'] - 1

# In US since June 2007 (at least 5 years continuous residence)
# YRIMMIG <= 2007 ensures they arrived by 2007
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)

# Under 31 as of June 15, 2012
df['under_31_in_2012'] = (df['age_june2012_adj'] < 31).astype(int)

# ============================================================================
# STEP 3: Define DACA Eligibility
# ============================================================================
print("\n[STEP 3] Defining DACA eligibility...")

# DACA eligible if all criteria met:
# - Arrived before age 16
# - Under 31 as of June 15, 2012
# - In US continuously since at least June 2007 (approximated by YRIMMIG <= 2007)
# Note: We cannot observe "present on June 15, 2012" directly

df['daca_eligible'] = (
    (df['arrived_before_16'] == 1) &
    (df['under_31_in_2012'] == 1) &
    (df['in_us_since_2007'] == 1)
).astype(int)

print(f"DACA eligible: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")
print(f"Not DACA eligible: {(1-df['daca_eligible']).sum():,} ({(1-df['daca_eligible']).mean()*100:.1f}%)")

# ============================================================================
# STEP 4: Define Treatment Period
# ============================================================================
print("\n[STEP 4] Defining treatment period...")

# Exclude 2012 due to mid-year implementation
df = df[df['YEAR'] != 2012].copy()

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"\nYear distribution after excluding 2012:")
print(df.groupby('YEAR').size())

print(f"\nPre-period (2006-2011): {(df['post']==0).sum():,} observations")
print(f"Post-period (2013-2016): {(df['post']==1).sum():,} observations")

# ============================================================================
# STEP 5: Create Additional Control Variables
# ============================================================================
print("\n[STEP 5] Creating control variables...")

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator (MARST: 1=married spouse present, 2=married spouse absent)
df['married'] = df['MARST'].isin([1, 2]).astype(int)

# Education categories
# EDUC: 0-3 less than HS, 6=HS, 7-9=some college, 10+=college
df['educ_less_hs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_some_college'] = df['EDUC'].isin([7, 8, 9]).astype(int)
df['educ_college'] = (df['EDUC'] >= 10).astype(int)

# Age squared for polynomial age control
df['age_sq'] = df['AGE'] ** 2

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']
df['years_in_us'] = df['years_in_us'].clip(lower=0)

print("Control variables created: female, married, education dummies, age, age_sq, years_in_us")

# ============================================================================
# STEP 6: Restrict to Working-Age Population
# ============================================================================
print("\n[STEP 6] Restricting to working-age population (18-64)...")

df_working = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"Working-age sample size: {len(df_working):,}")

# ============================================================================
# STEP 7: Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("[STEP 7] SUMMARY STATISTICS")
print("="*70)

# By eligibility status
print("\n--- By DACA Eligibility Status ---")
for elig in [0, 1]:
    sub = df_working[df_working['daca_eligible'] == elig]
    n = len(sub)
    n_weighted = sub['PERWT'].sum()
    print(f"\n{'DACA Eligible' if elig else 'Not DACA Eligible'}: N={n:,} (Weighted: {n_weighted:,.0f})")
    print(f"  Full-time employment rate: {sub['fulltime'].mean()*100:.2f}%")
    print(f"  Employment rate: {sub['employed'].mean()*100:.2f}%")
    print(f"  Mean age: {sub['AGE'].mean():.1f}")
    print(f"  Female: {sub['female'].mean()*100:.1f}%")
    print(f"  Married: {sub['married'].mean()*100:.1f}%")
    print(f"  Less than HS: {sub['educ_less_hs'].mean()*100:.1f}%")

# By period
print("\n--- By Time Period ---")
for period, label in [(0, 'Pre-DACA (2006-2011)'), (1, 'Post-DACA (2013-2016)')]:
    sub = df_working[df_working['post'] == period]
    print(f"\n{label}: N={len(sub):,}")
    print(f"  Full-time rate (eligible): {sub[sub['daca_eligible']==1]['fulltime'].mean()*100:.2f}%")
    print(f"  Full-time rate (not eligible): {sub[sub['daca_eligible']==0]['fulltime'].mean()*100:.2f}%")

# ============================================================================
# STEP 8: Difference-in-Differences Analysis
# ============================================================================
print("\n" + "="*70)
print("[STEP 8] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*70)

# Create interaction term
df_working['eligible_x_post'] = df_working['daca_eligible'] * df_working['post']

# 8a. Simple DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + eligible_x_post',
                  data=df_working, weights=df_working['PERWT'])
results1 = model1.fit(cov_type='HC1')  # Robust standard errors
print(f"DiD Coefficient (eligible_x_post): {results1.params['eligible_x_post']:.4f}")
print(f"Standard Error: {results1.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{results1.conf_int().loc['eligible_x_post', 0]:.4f}, {results1.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results1.pvalues['eligible_x_post']:.4f}")
print(f"N: {int(results1.nobs):,}")

# 8b. DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + eligible_x_post + ' +
                  'female + married + AGE + age_sq + educ_hs + educ_some_college + educ_college + years_in_us',
                  data=df_working, weights=df_working['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(f"DiD Coefficient (eligible_x_post): {results2.params['eligible_x_post']:.4f}")
print(f"Standard Error: {results2.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{results2.conf_int().loc['eligible_x_post', 0]:.4f}, {results2.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results2.pvalues['eligible_x_post']:.4f}")
print(f"N: {int(results2.nobs):,}")

# 8c. DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
df_working['year_factor'] = pd.Categorical(df_working['YEAR'])
model3 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + ' +
                  'female + married + AGE + age_sq + educ_hs + educ_some_college + educ_college + years_in_us',
                  data=df_working, weights=df_working['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(f"DiD Coefficient (eligible_x_post): {results3.params['eligible_x_post']:.4f}")
print(f"Standard Error: {results3.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{results3.conf_int().loc['eligible_x_post', 0]:.4f}, {results3.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results3.pvalues['eligible_x_post']:.4f}")
print(f"N: {int(results3.nobs):,}")

# 8d. DiD with state fixed effects
print("\n--- Model 4: DiD with year and state fixed effects ---")
model4 = smf.wls('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + ' +
                  'female + married + AGE + age_sq + educ_hs + educ_some_college + educ_college + years_in_us',
                  data=df_working, weights=df_working['PERWT'])
results4 = model4.fit(cov_type='HC1')
print(f"DiD Coefficient (eligible_x_post): {results4.params['eligible_x_post']:.4f}")
print(f"Standard Error: {results4.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['eligible_x_post', 0]:.4f}, {results4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results4.pvalues['eligible_x_post']:.4f}")
print(f"N: {int(results4.nobs):,}")

# ============================================================================
# STEP 9: Robustness - Employment (any) as outcome
# ============================================================================
print("\n" + "="*70)
print("[STEP 9] ROBUSTNESS: Employment (any) as Outcome")
print("="*70)

model_emp = smf.wls('employed ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + ' +
                     'female + married + AGE + age_sq + educ_hs + educ_some_college + educ_college + years_in_us',
                     data=df_working, weights=df_working['PERWT'])
results_emp = model_emp.fit(cov_type='HC1')
print(f"DiD Coefficient (eligible_x_post): {results_emp.params['eligible_x_post']:.4f}")
print(f"Standard Error: {results_emp.bse['eligible_x_post']:.4f}")
print(f"95% CI: [{results_emp.conf_int().loc['eligible_x_post', 0]:.4f}, {results_emp.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results_emp.pvalues['eligible_x_post']:.4f}")

# ============================================================================
# STEP 10: Subgroup Analysis by Gender
# ============================================================================
print("\n" + "="*70)
print("[STEP 10] SUBGROUP ANALYSIS BY GENDER")
print("="*70)

for sex, label in [(0, 'Male'), (1, 'Female')]:
    sub = df_working[df_working['female'] == sex]
    model_sub = smf.wls('fulltime ~ daca_eligible + eligible_x_post + C(YEAR) + C(STATEFIP) + ' +
                         'married + AGE + age_sq + educ_hs + educ_some_college + educ_college + years_in_us',
                         data=sub, weights=sub['PERWT'])
    results_sub = model_sub.fit(cov_type='HC1')
    print(f"\n{label}:")
    print(f"  DiD Coefficient: {results_sub.params['eligible_x_post']:.4f}")
    print(f"  Standard Error: {results_sub.bse['eligible_x_post']:.4f}")
    print(f"  95% CI: [{results_sub.conf_int().loc['eligible_x_post', 0]:.4f}, {results_sub.conf_int().loc['eligible_x_post', 1]:.4f}]")
    print(f"  N: {int(results_sub.nobs):,}")

# ============================================================================
# STEP 11: Event Study / Pre-Trends Check
# ============================================================================
print("\n" + "="*70)
print("[STEP 11] EVENT STUDY - DYNAMIC EFFECTS")
print("="*70)

# Create year interactions with eligibility (reference year: 2011)
for yr in df_working['YEAR'].unique():
    df_working[f'elig_x_{yr}'] = (df_working['daca_eligible'] * (df_working['YEAR'] == yr)).astype(int)

# Drop 2011 as reference year
year_vars = [f'elig_x_{yr}' for yr in sorted(df_working['YEAR'].unique()) if yr != 2011]
formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + \
                ' + C(YEAR) + C(STATEFIP) + female + married + AGE + age_sq + ' + \
                'educ_hs + educ_some_college + educ_college + years_in_us'

model_event = smf.wls(formula_event, data=df_working, weights=df_working['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nYear-by-Year Effects (Reference: 2011):")
print("-" * 60)
for yr in sorted(df_working['YEAR'].unique()):
    if yr == 2011:
        print(f"  {yr}: 0.0000 (reference)")
    else:
        var = f'elig_x_{yr}'
        coef = results_event.params[var]
        se = results_event.bse[var]
        pval = results_event.pvalues[var]
        print(f"  {yr}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")

# ============================================================================
# STEP 12: Save Results for Report
# ============================================================================
print("\n" + "="*70)
print("[STEP 12] SAVING RESULTS")
print("="*70)

# Create results summary
results_summary = {
    'Model 1 (Basic)': {
        'coefficient': results1.params['eligible_x_post'],
        'se': results1.bse['eligible_x_post'],
        'ci_low': results1.conf_int().loc['eligible_x_post', 0],
        'ci_high': results1.conf_int().loc['eligible_x_post', 1],
        'pvalue': results1.pvalues['eligible_x_post'],
        'n': int(results1.nobs)
    },
    'Model 2 (Demographics)': {
        'coefficient': results2.params['eligible_x_post'],
        'se': results2.bse['eligible_x_post'],
        'ci_low': results2.conf_int().loc['eligible_x_post', 0],
        'ci_high': results2.conf_int().loc['eligible_x_post', 1],
        'pvalue': results2.pvalues['eligible_x_post'],
        'n': int(results2.nobs)
    },
    'Model 3 (Year FE)': {
        'coefficient': results3.params['eligible_x_post'],
        'se': results3.bse['eligible_x_post'],
        'ci_low': results3.conf_int().loc['eligible_x_post', 0],
        'ci_high': results3.conf_int().loc['eligible_x_post', 1],
        'pvalue': results3.pvalues['eligible_x_post'],
        'n': int(results3.nobs)
    },
    'Model 4 (Year + State FE)': {
        'coefficient': results4.params['eligible_x_post'],
        'se': results4.bse['eligible_x_post'],
        'ci_low': results4.conf_int().loc['eligible_x_post', 0],
        'ci_high': results4.conf_int().loc['eligible_x_post', 1],
        'pvalue': results4.pvalues['eligible_x_post'],
        'n': int(results4.nobs)
    }
}

# Save summary stats
summary_stats = df_working.groupby(['daca_eligible', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_less_hs': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Group:")
print(summary_stats)

# Event study coefficients
event_study_results = {}
for yr in sorted(df_working['YEAR'].unique()):
    if yr == 2011:
        event_study_results[yr] = {'coef': 0, 'se': 0}
    else:
        var = f'elig_x_{yr}'
        event_study_results[yr] = {
            'coef': results_event.params[var],
            'se': results_event.bse[var]
        }

print("\nEvent Study Results:")
for yr, vals in event_study_results.items():
    print(f"  {yr}: {vals['coef']:.4f} (SE: {vals['se']:.4f})")

# Save to CSV for plotting
event_df = pd.DataFrame([
    {'year': yr, 'coefficient': vals['coef'], 'se': vals['se']}
    for yr, vals in event_study_results.items()
])
event_df.to_csv('event_study_results.csv', index=False)

# Save main results
results_df = pd.DataFrame(results_summary).T
results_df.to_csv('main_results.csv')

# Save detailed model output
with open('model_output.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PREFERRED SPECIFICATION: Model 4 (Year + State Fixed Effects)\n")
    f.write("="*70 + "\n\n")
    f.write(results4.summary().as_text())
    f.write("\n\n")
    f.write("="*70 + "\n")
    f.write("EVENT STUDY RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(results_event.summary().as_text())

print("\nResults saved to:")
print("  - main_results.csv")
print("  - event_study_results.csv")
print("  - model_output.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("="*70)
print(f"\nPreferred specification: Model 4 (Year + State Fixed Effects)")
print(f"Outcome: Full-time employment (35+ hours/week)")
print(f"Treatment: DACA eligibility")
print(f"\nEffect size: {results4.params['eligible_x_post']:.4f}")
print(f"Standard error: {results4.bse['eligible_x_post']:.4f}")
print(f"95% Confidence interval: [{results4.conf_int().loc['eligible_x_post', 0]:.4f}, {results4.conf_int().loc['eligible_x_post', 1]:.4f}]")
print(f"P-value: {results4.pvalues['eligible_x_post']:.4f}")
print(f"Sample size: {int(results4.nobs):,}")
print(f"\nInterpretation: DACA eligibility is associated with a {results4.params['eligible_x_post']*100:.2f} percentage point")
print(f"{'increase' if results4.params['eligible_x_post'] > 0 else 'decrease'} in the probability of full-time employment.")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

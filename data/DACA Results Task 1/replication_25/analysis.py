"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican, Mexican-born people in the US,
what was the causal impact of DACA eligibility on full-time employment (>=35 hrs/week)?

DACA implemented June 15, 2012. Examine effects in 2013-2016.

Eligibility criteria:
1. Arrived unlawfully before 16th birthday
2. Had not yet had 31st birthday as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012 without lawful status
5. Non-citizen, no immigration papers -> assume undocumented

Strategy: Difference-in-differences comparing DACA-eligible to non-eligible
Mexican-born Hispanic non-citizens before and after 2012.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION ANALYSIS")
print("="*70)

# Read the data in chunks due to large file size
print("\nLoading data in chunks (filtering to relevant population)...")
# We'll read only necessary columns to save memory
cols_needed = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'MARST']

# Read data in chunks, filtering as we go
chunksize = 500000
chunks = []

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunksize)):
    # Filter immediately to reduce memory
    # Hispanic-Mexican (HISPAN=1), born in Mexico (BPL=200), non-citizen (CITIZEN=3)
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3)]
    if len(filtered) > 0:
        chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
del chunks
print(f"Total relevant observations loaded: {len(df):,}")

# Check years available
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")

# ============================================================
# STEP 1: Filter to target population
# ============================================================
print("\n" + "="*70)
print("STEP 1: Filter to target population")
print("="*70)

# Already filtered during load:
# Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
# Born in Mexico (BPL == 200 for Mexico)
# Non-citizens only (CITIZEN == 3)
df_mex = df.copy()
print(f"Hispanic-Mexican, Mexican-born, non-citizens: {len(df_mex):,}")

# Working age population (16-64)
df_mex = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 64)].copy()
print(f"After filtering to ages 16-64: {len(df_mex):,}")

# ============================================================
# STEP 2: Define DACA eligibility
# ============================================================
print("\n" + "="*70)
print("STEP 2: Define DACA eligibility")
print("="*70)

"""
DACA eligibility criteria (as of June 15, 2012):
1. Arrived before 16th birthday
2. Born after June 15, 1981 (not yet 31 on June 15, 2012)
3. Present in US since June 15, 2007 (at least 5 years by 2012)
4. Non-citizen (already filtered)

For identification, we use age-based eligibility:
- Treatment group: Those who meet age criteria (born after June 1981, arrived young)
- Control group: Similar non-citizens who don't meet criteria (too old)
"""

# Create binary indicator for post-DACA period (2013-2016)
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)

# Year of immigration - check for continuous presence
# YRIMMIG: Year of immigration (0 = N/A for US-born)
# Need to have arrived by 2007 to meet continuous presence requirement
df_mex['arrived_by_2007'] = (df_mex['YRIMMIG'] <= 2007) & (df_mex['YRIMMIG'] > 0)

# Arrived before age 16
# Calculate approximate age at arrival: YRIMMIG - BIRTHYR
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16) & (df_mex['age_at_arrival'] >= 0)

# Born after June 15, 1981 (under 31 as of June 2012)
# Conservative: use birth year >= 1982 to ensure under 31
# But some born in 1981 after June 15 would qualify
# Since we don't have exact birth date, use BIRTHQTR
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Those born 1981 Q3 or Q4 would be under 31 on June 15, 2012
df_mex['born_after_june1981'] = (
    (df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'].isin([3, 4])))
)

# DACA eligible: meets all criteria
df_mex['daca_eligible'] = (
    df_mex['arrived_before_16'] &
    df_mex['arrived_by_2007'] &
    df_mex['born_after_june1981']
).astype(int)

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df_mex['arrived_before_16'].sum():,}")
print(f"  Arrived by 2007: {df_mex['arrived_by_2007'].sum():,}")
print(f"  Born after June 1981: {df_mex['born_after_june1981'].sum():,}")
print(f"  DACA eligible (all criteria): {df_mex['daca_eligible'].sum():,}")
print(f"  Not DACA eligible: {(~df_mex['daca_eligible'].astype(bool)).sum():,}")

# ============================================================
# STEP 3: Define outcome variable
# ============================================================
print("\n" + "="*70)
print("STEP 3: Define outcome variable")
print("="*70)

# Full-time employment: Usually works 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A or not working)
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Also create employed variable
# EMPSTAT: 1=Employed, 2=Unemployed, 3=Not in labor force
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

print(f"Full-time employment rate (UHRSWORK>=35): {df_mex['fulltime'].mean():.3f}")
print(f"Employment rate (EMPSTAT=1): {df_mex['employed'].mean():.3f}")

# ============================================================
# STEP 4: Create analysis sample
# ============================================================
print("\n" + "="*70)
print("STEP 4: Create analysis sample")
print("="*70)

# Focus on pre-DACA (2008-2011) and post-DACA (2013-2016) periods
# Exclude 2012 as implementation year
df_analysis = df_mex[(df_mex['YEAR'] != 2012)].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# Restrict to 2008-2016 for cleaner pre/post comparison
df_analysis = df_analysis[(df_analysis['YEAR'] >= 2008) & (df_analysis['YEAR'] <= 2016)].copy()
print(f"After restricting to 2008-2016: {len(df_analysis):,}")

# Create age controls
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Education categories
df_analysis['educ_cat'] = pd.cut(df_analysis['EDUC'],
                                  bins=[-1, 2, 6, 10, 11],
                                  labels=['Less than HS', 'HS/Some College', 'College', 'Graduate'])

# Marital status
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Summary by group and period
print("\n" + "-"*50)
print("Sample composition:")
print("-"*50)
summary = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'PERWT': 'sum'
}).round(3)
print(summary)

# ============================================================
# STEP 5: Difference-in-Differences Analysis
# ============================================================
print("\n" + "="*70)
print("STEP 5: Difference-in-Differences Analysis")
print("="*70)

# Create interaction term
df_analysis['did'] = df_analysis['daca_eligible'] * df_analysis['post']

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with state fixed effects
print("\n--- Model 3: DiD with state fixed effects ---")
model3 = smf.wls('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married + C(STATEFIP)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Extract key coefficients
did_coef = model3.params['did']
did_se = model3.bse['did']
did_pval = model3.pvalues['did']
did_ci = model3.conf_int().loc['did']

print(f"\nDiD coefficient (did): {did_coef:.4f}")
print(f"Standard error: {did_se:.4f}")
print(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"P-value: {did_pval:.4f}")

# Model 4: Full model with year FE
print("\n--- Model 4: DiD with year and state FE ---")
model4 = smf.wls('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

did_coef4 = model4.params['did']
did_se4 = model4.bse['did']
did_pval4 = model4.pvalues['did']
did_ci4 = model4.conf_int().loc['did']

print(f"\nDiD coefficient (did): {did_coef4:.4f}")
print(f"Standard error: {did_se4:.4f}")
print(f"95% CI: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"P-value: {did_pval4:.4f}")

# ============================================================
# STEP 6: Robustness Checks
# ============================================================
print("\n" + "="*70)
print("STEP 6: Robustness Checks")
print("="*70)

# A. Employment (any work) as outcome
print("\n--- Robustness A: Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ daca_eligible + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                     data=df_analysis,
                     weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD on employment: {model_emp.params['did']:.4f} (SE: {model_emp.bse['did']:.4f})")

# B. Restrict to specific age range (more comparable treatment/control)
print("\n--- Robustness B: Restricted age range (18-45) ---")
df_restricted = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 45)]
model_age = smf.wls('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)',
                     data=df_restricted,
                     weights=df_restricted['PERWT']).fit(cov_type='HC1')
print(f"DiD on full-time (ages 18-45): {model_age.params['did']:.4f} (SE: {model_age.bse['did']:.4f})")

# C. By gender
print("\n--- Robustness C: By gender ---")
df_male = df_analysis[df_analysis['female'] == 0]
df_female = df_analysis[df_analysis['female'] == 1]

model_male = smf.wls('fulltime ~ daca_eligible + did + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
                      data=df_male,
                      weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ daca_eligible + did + AGE + age_sq + married + C(YEAR) + C(STATEFIP)',
                        data=df_female,
                        weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"DiD for males: {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f})")
print(f"DiD for females: {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f})")

# ============================================================
# STEP 7: Event Study / Pre-trends
# ============================================================
print("\n" + "="*70)
print("STEP 7: Event Study Analysis")
print("="*70)

# Create year dummies interacted with treatment
df_analysis['year_centered'] = df_analysis['YEAR'] - 2012

# Create event study indicators
for year in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
    df_analysis[f'daca_x_{year}'] = df_analysis['daca_eligible'] * df_analysis[f'year_{year}']

# Event study regression (2011 as reference)
event_formula = 'fulltime ~ daca_eligible + daca_x_2008 + daca_x_2009 + daca_x_2010 + daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 + AGE + age_sq + female + married + C(YEAR) + C(STATEFIP)'
model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent study coefficients (relative to 2011):")
event_vars = ['daca_x_2008', 'daca_x_2009', 'daca_x_2010', 'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    year = var.split('_')[-1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

# ============================================================
# STEP 8: Summary Statistics Tables
# ============================================================
print("\n" + "="*70)
print("STEP 8: Summary Statistics")
print("="*70)

# Table 1: Summary by treatment status and period
print("\n--- Table 1: Sample Means by Treatment/Period ---")

def weighted_stats(df, weight_col='PERWT'):
    """Calculate weighted means"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = {}
    for col in ['fulltime', 'employed', 'AGE', 'female', 'married']:
        if col in df.columns:
            weights = df[weight_col]
            result[col] = np.average(df[col], weights=weights)
    result['N'] = len(df)
    result['weighted_N'] = df[weight_col].sum()
    return pd.Series(result)

summary_table = df_analysis.groupby(['daca_eligible', 'post']).apply(weighted_stats)
print(summary_table.round(3))

# ============================================================
# PREFERRED ESTIMATE
# ============================================================
print("\n" + "="*70)
print("PREFERRED ESTIMATE (Model 4 with Year and State FE)")
print("="*70)
print(f"Effect size (DiD coefficient): {did_coef4:.4f}")
print(f"Standard error: {did_se4:.4f}")
print(f"95% Confidence interval: [{did_ci4[0]:.4f}, {did_ci4[1]:.4f}]")
print(f"Sample size: {len(df_analysis):,}")
print(f"P-value: {did_pval4:.4f}")

# Save key results for report
results = {
    'did_coef': float(did_coef4),
    'did_se': float(did_se4),
    'did_ci_lower': float(did_ci4[0]),
    'did_ci_upper': float(did_ci4[1]),
    'did_pval': float(did_pval4),
    'n_obs': int(len(df_analysis)),
    'n_treated': int(df_analysis['daca_eligible'].sum()),
    'n_control': int(len(df_analysis) - df_analysis['daca_eligible'].sum())
}

# Save results to file for LaTeX
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to results.json")

# ============================================================
# Generate additional outputs for report
# ============================================================

# Trends by year
print("\n--- Full-time employment trends by year and eligibility ---")
trends = df_analysis.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x)
    })
).round(4)
print(trends)

# Save trends
trends.to_csv('trends.csv')
print("\nTrends saved to trends.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

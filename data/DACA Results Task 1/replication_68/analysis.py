"""
DACA Replication Analysis - Study 68
======================================
Estimating the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Research Question: What was the causal impact of DACA eligibility (treatment)
on full-time employment (outcome) among Hispanic-Mexican, Mexican-born people?

Design: Difference-in-Differences (DiD)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
import json
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS - STUDY 68")
print("="*80)

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("\n[1] Loading data (this may take a while due to file size)...")

# Define columns to use (reducing memory)
usecols = ['YEAR', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'MARST',
           'EMPSTAT', 'UHRSWORK']

dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'MARST': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8'
}

# Read data in chunks and filter
chunk_size = 500000
filtered_chunks = []

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols,
                                       dtype=dtypes, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican (HISPAN=1), Mexican-born (BPL=200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        filtered_chunks.append(filtered)
    if (i+1) % 20 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(filtered_chunks, ignore_index=True)
print(f"  Total Hispanic-Mexican, Mexican-born individuals: {len(df):,}")

# ============================================================================
# STEP 2: Create Analysis Variables
# ============================================================================
print("\n[2] Creating analysis variables...")

# Calculate age at immigration
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']

# Create arrival before 16 indicator
df['arrived_before_16'] = (df['age_at_immigration'] < 16) & (df['age_at_immigration'] >= 0)

# DACA age criterion: born 1981 or later (under 31 as of June 15, 2012)
df['under_31_june2012'] = (df['BIRTHYR'] > 1981) | \
                           ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 2))

# DACA residence criterion: in US by June 15, 2007
df['in_us_by_2007'] = df['YRIMMIG'] <= 2007

# Non-citizen (proxy for potentially undocumented)
df['non_citizen'] = df['CITIZEN'] == 3

# DACA eligibility indicator (treatment)
df['daca_eligible'] = (df['arrived_before_16'] &
                        df['under_31_june2012'] &
                        df['in_us_by_2007'] &
                        df['non_citizen'])

# Post-DACA indicator (2013 onwards)
df['post_daca'] = df['YEAR'] >= 2013

# Full-time employment outcome
df['employed'] = df['EMPSTAT'] == 1
df['fulltime'] = (df['UHRSWORK'] >= 35) & df['employed']

# Interaction term for DiD
df['daca_x_post'] = df['daca_eligible'].astype(int) * df['post_daca'].astype(int)

print(f"  DACA eligible: {df['daca_eligible'].sum():,}")
print(f"  Non-eligible: {(~df['daca_eligible']).sum():,}")

# ============================================================================
# STEP 3: Sample Restrictions
# ============================================================================
print("\n[3] Applying sample restrictions...")

# Working age (18-45)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 45)]
print(f"  After age restriction (18-45): {len(df):,}")

# Valid YRIMMIG
df = df[df['YRIMMIG'] > 0]
print(f"  After valid immigration year: {len(df):,}")

# Exclude 2012
df = df[df['YEAR'] != 2012]
print(f"  After excluding 2012: {len(df):,}")

# Focus on non-citizens
df = df[df['non_citizen']]
print(f"  After restricting to non-citizens: {len(df):,}")

# Reset index
df = df.reset_index(drop=True)

# ============================================================================
# STEP 4: Create Control Variables
# ============================================================================
print("\n[4] Creating control variables...")

df['female'] = (df['SEX'] == 2).astype(float)
df['married'] = (df['MARST'].isin([1, 2])).astype(float)

# Education categories
df['less_than_hs'] = (df['EDUC'] < 6).astype(float)
df['hs_grad'] = (df['EDUC'] == 6).astype(float)
df['some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(float)
df['college_plus'] = (df['EDUC'] >= 10).astype(float)

# Convert key variables to float
df['daca_eligible_f'] = df['daca_eligible'].astype(float)
df['post_daca_f'] = df['post_daca'].astype(float)
df['daca_x_post_f'] = df['daca_x_post'].astype(float)
df['fulltime_f'] = df['fulltime'].astype(float)
df['employed_f'] = df['employed'].astype(float)
df['age_f'] = df['AGE'].astype(float)

# ============================================================================
# STEP 5: Descriptive Statistics
# ============================================================================
print("\n[5] Descriptive Statistics")
print("-"*60)

# Sample sizes by treatment and period
print("\nSample sizes by group and period:")
crosstab = pd.crosstab(df['daca_eligible'], df['post_daca'], margins=True)
crosstab.index = ['Non-eligible', 'DACA-eligible', 'Total']
crosstab.columns = ['Pre-DACA (2006-11)', 'Post-DACA (2013-16)', 'Total']
print(crosstab)

# Mean full-time employment by group and period
print("\nMean full-time employment rates:")
means = df.groupby(['daca_eligible', 'post_daca'])['fulltime'].mean().unstack()
means.index = ['Non-eligible', 'DACA-eligible']
means.columns = ['Pre-DACA', 'Post-DACA']
means['Diff'] = means['Post-DACA'] - means['Pre-DACA']
print(means.round(4))

# Simple DiD estimate
did_simple = (means.loc['DACA-eligible', 'Diff'] - means.loc['Non-eligible', 'Diff'])
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# ============================================================================
# STEP 6: Main DiD Regressions
# ============================================================================
print("\n" + "="*60)
print("[6] MAIN DIFFERENCE-IN-DIFFERENCES REGRESSIONS")
print("="*60)

# Prepare outcome and weights
y = df['fulltime_f'].values
weights = df['PERWT'].values

# --- Model 1: Basic DiD ---
print("\n--- Model 1: Basic DiD ---")
X1_data = np.column_stack([
    np.ones(len(df)),
    df['daca_eligible_f'].values,
    df['post_daca_f'].values,
    df['daca_x_post_f'].values
])
model1 = sm.WLS(y, X1_data, weights=weights).fit(cov_type='HC1')
m1_names = ['const', 'daca_eligible', 'post_daca', 'daca_x_post']
print("Coefficients:")
for i, name in enumerate(m1_names):
    print(f"  {name}: {model1.params[i]:.4f} (SE: {model1.bse[i]:.4f})")
print(f"R-squared: {model1.rsquared:.4f}, N: {int(model1.nobs)}")

# --- Model 2: DiD with Demographic Controls ---
print("\n--- Model 2: DiD with Demographic Controls ---")
X2_data = np.column_stack([
    np.ones(len(df)),
    df['daca_eligible_f'].values,
    df['post_daca_f'].values,
    df['daca_x_post_f'].values,
    df['age_f'].values,
    df['female'].values,
    df['married'].values,
    df['hs_grad'].values,
    df['some_college'].values,
    df['college_plus'].values
])
model2 = sm.WLS(y, X2_data, weights=weights).fit(cov_type='HC1')
m2_names = ['const', 'daca_eligible', 'post_daca', 'daca_x_post',
            'AGE', 'female', 'married', 'hs_grad', 'some_college', 'college_plus']
print("Coefficients:")
for i, name in enumerate(m2_names):
    print(f"  {name}: {model2.params[i]:.4f} (SE: {model2.bse[i]:.4f})")
print(f"R-squared: {model2.rsquared:.4f}, N: {int(model2.nobs)}")

# --- Model 3: DiD with State and Year Fixed Effects ---
print("\n--- Model 3: DiD with State and Year Fixed Effects ---")

# Create state and year dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True).astype(float).values
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True).astype(float).values

X3_data = np.column_stack([
    np.ones(len(df)),
    df['daca_eligible_f'].values,
    df['post_daca_f'].values,
    df['daca_x_post_f'].values,
    df['age_f'].values,
    df['female'].values,
    df['married'].values,
    df['hs_grad'].values,
    df['some_college'].values,
    df['college_plus'].values,
    state_dummies,
    year_dummies
])

model3 = sm.WLS(y, X3_data, weights=weights).fit(cov_type='HC1')

# Key coefficients are in first 10 positions
m3_names = ['const', 'daca_eligible', 'post_daca', 'daca_x_post',
            'AGE', 'female', 'married', 'hs_grad', 'some_college', 'college_plus']
print("\nMain coefficients (Model 3 with FE):")
for i, name in enumerate(m3_names):
    coef = model3.params[i]
    se = model3.bse[i]
    pval = model3.pvalues[i]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {name:20s}: {coef:8.4f} ({se:.4f}) {stars}")

print(f"\nR-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

# Store Model 3 results
did_coef = model3.params[3]  # daca_x_post
did_se = model3.bse[3]
did_pval = model3.pvalues[3]

# ============================================================================
# STEP 7: Robustness Checks
# ============================================================================
print("\n" + "="*60)
print("[7] ROBUSTNESS CHECKS")
print("="*60)

# 7a. Alternative outcome: any employment
print("\n--- 7a. Alternative outcome: Any employment ---")
y_emp = df['employed_f'].values
model_emp = sm.WLS(y_emp, X3_data, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient (employment): {model_emp.params[3]:.4f} (SE: {model_emp.bse[3]:.4f})")

# 7b. Restrict to younger age group (18-30)
print("\n--- 7b. Restricted sample: Age 18-30 ---")
mask_young = (df['AGE'] >= 18) & (df['AGE'] <= 30)
df_young = df[mask_young].reset_index(drop=True)

y_young = df_young['fulltime_f'].values
weights_young = df_young['PERWT'].values

state_d_young = pd.get_dummies(df_young['STATEFIP'], prefix='state', drop_first=True).astype(float).values
year_d_young = pd.get_dummies(df_young['YEAR'], prefix='year', drop_first=True).astype(float).values

X_young = np.column_stack([
    np.ones(len(df_young)),
    df_young['daca_eligible_f'].values,
    df_young['post_daca_f'].values,
    df_young['daca_x_post_f'].values,
    df_young['age_f'].values,
    df_young['female'].values,
    df_young['married'].values,
    df_young['hs_grad'].values,
    df_young['some_college'].values,
    df_young['college_plus'].values,
    state_d_young,
    year_d_young
])

model_young = sm.WLS(y_young, X_young, weights=weights_young).fit(cov_type='HC1')
print(f"DiD coefficient (age 18-30): {model_young.params[3]:.4f} (SE: {model_young.bse[3]:.4f})")
print(f"N: {int(model_young.nobs):,}")

# 7c. Separate effects by gender
print("\n--- 7c. Heterogeneity by gender ---")
for gender_val, label in [(0.0, 'Male'), (1.0, 'Female')]:
    mask_g = df['female'] == gender_val
    df_g = df[mask_g].reset_index(drop=True)

    y_g = df_g['fulltime_f'].values
    w_g = df_g['PERWT'].values

    state_d_g = pd.get_dummies(df_g['STATEFIP'], prefix='state', drop_first=True).astype(float).values
    year_d_g = pd.get_dummies(df_g['YEAR'], prefix='year', drop_first=True).astype(float).values

    X_g = np.column_stack([
        np.ones(len(df_g)),
        df_g['daca_eligible_f'].values,
        df_g['post_daca_f'].values,
        df_g['daca_x_post_f'].values,
        df_g['age_f'].values,
        df_g['married'].values,
        df_g['hs_grad'].values,
        df_g['some_college'].values,
        df_g['college_plus'].values,
        state_d_g,
        year_d_g
    ])

    model_g = sm.WLS(y_g, X_g, weights=w_g).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_g.params[3]:.4f} (SE: {model_g.bse[3]:.4f}), N = {int(model_g.nobs):,}")

# Store gender results
gender_results = {}
for gender_val, label in [(0.0, 'Male'), (1.0, 'Female')]:
    mask_g = df['female'] == gender_val
    df_g = df[mask_g].reset_index(drop=True)
    y_g = df_g['fulltime_f'].values
    w_g = df_g['PERWT'].values
    state_d_g = pd.get_dummies(df_g['STATEFIP'], prefix='state', drop_first=True).astype(float).values
    year_d_g = pd.get_dummies(df_g['YEAR'], prefix='year', drop_first=True).astype(float).values
    X_g = np.column_stack([
        np.ones(len(df_g)),
        df_g['daca_eligible_f'].values,
        df_g['post_daca_f'].values,
        df_g['daca_x_post_f'].values,
        df_g['age_f'].values,
        df_g['married'].values,
        df_g['hs_grad'].values,
        df_g['some_college'].values,
        df_g['college_plus'].values,
        state_d_g,
        year_d_g
    ])
    model_g = sm.WLS(y_g, X_g, weights=w_g).fit(cov_type='HC1')
    gender_results[label] = {'coef': model_g.params[3], 'se': model_g.bse[3], 'n': int(model_g.nobs)}

# 7d. Event study
print("\n--- 7d. Event study: Year-by-year effects ---")
years = sorted(df['YEAR'].unique())
ref_year = 2011

# Create year interaction variables
year_int_cols = []
year_int_names = []
for yr in years:
    if yr != ref_year:
        col = (df['daca_eligible_f'] * (df['YEAR'] == yr).astype(float)).values
        year_int_cols.append(col)
        year_int_names.append(yr)

year_int_data = np.column_stack(year_int_cols) if year_int_cols else np.zeros((len(df), 0))

X_event = np.column_stack([
    np.ones(len(df)),
    df['daca_eligible_f'].values,
    df['age_f'].values,
    df['female'].values,
    df['married'].values,
    df['hs_grad'].values,
    df['some_college'].values,
    df['college_plus'].values,
    year_int_data,
    state_dummies,
    year_dummies
])

model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='HC1')

print(f"\nYear-by-year DACA eligibility effects (relative to {ref_year}):")
event_coeffs = {}
base_idx = 8  # After const, daca_eligible, AGE, female, married, hs_grad, some_college, college_plus
for i, yr in enumerate(year_int_names):
    coef = model_event.params[base_idx + i]
    se = model_event.bse[base_idx + i]
    pval = model_event.pvalues[base_idx + i]
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    pre_post = "POST" if yr >= 2013 else "PRE"
    print(f"  {yr} ({pre_post}): {coef:7.4f} ({se:.4f}) {stars}")
    event_coeffs[int(yr)] = {'coefficient': float(coef), 'se': float(se)}

# ============================================================================
# STEP 8: Summary of Results
# ============================================================================
print("\n" + "="*80)
print("[8] SUMMARY OF RESULTS")
print("="*80)

print("\n*** PREFERRED ESTIMATE: Model 3 (DiD with controls, state & year FE) ***")
print("-"*60)
print(f"Outcome: Full-time employment (35+ hours/week)")
print(f"Treatment: DACA eligibility")
print(f"Sample: Mexican-born, Hispanic-Mexican, non-citizen adults aged 18-45")
print(f"Pre-period: 2006-2011 | Post-period: 2013-2016")
print("-"*60)

did_ci_low = did_coef - 1.96 * did_se
did_ci_high = did_coef + 1.96 * did_se

print(f"\nDiD Coefficient (DACA effect): {did_coef:.4f}")
print(f"Standard Error (robust):        {did_se:.4f}")
print(f"95% Confidence Interval:        [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"P-value:                        {did_pval:.4f}")
print(f"Sample Size:                    {int(model3.nobs):,}")

# Baseline full-time rate for eligible pre-DACA
baseline = df[(df['daca_eligible']) & (~df['post_daca'])]['fulltime'].mean()
print(f"\nBaseline full-time rate (eligible, pre-DACA): {baseline:.4f}")
print(f"Percentage point change:                       {did_coef*100:.2f} pp")
if baseline > 0:
    print(f"Percent change from baseline:                  {(did_coef/baseline)*100:.1f}%")

# ============================================================================
# STEP 9: Save Results
# ============================================================================
print("\n[9] Saving results...")

results = {
    'preferred_estimate': {
        'coefficient': float(did_coef),
        'se': float(did_se),
        'ci_low': float(did_ci_low),
        'ci_high': float(did_ci_high),
        'pvalue': float(did_pval),
        'n': int(model3.nobs),
        'r2': float(model3.rsquared)
    },
    'model1': {
        'coefficient': float(model1.params[3]),
        'se': float(model1.bse[3]),
        'n': int(model1.nobs),
        'r2': float(model1.rsquared)
    },
    'model2': {
        'coefficient': float(model2.params[3]),
        'se': float(model2.bse[3]),
        'n': int(model2.nobs),
        'r2': float(model2.rsquared)
    },
    'model3': {
        'coefficient': float(model3.params[3]),
        'se': float(model3.bse[3]),
        'n': int(model3.nobs),
        'r2': float(model3.rsquared),
        'all_coeffs': {name: {'coef': float(model3.params[i]), 'se': float(model3.bse[i])}
                       for i, name in enumerate(m3_names)}
    },
    'robustness': {
        'employment': {
            'coefficient': float(model_emp.params[3]),
            'se': float(model_emp.bse[3])
        },
        'young_sample': {
            'coefficient': float(model_young.params[3]),
            'se': float(model_young.bse[3]),
            'n': int(model_young.nobs)
        },
        'by_gender': gender_results
    },
    'event_study': event_coeffs,
    'descriptive': {
        'n_eligible_pre': int(len(df[(df['daca_eligible']) & (~df['post_daca'])])),
        'n_eligible_post': int(len(df[(df['daca_eligible']) & (df['post_daca'])])),
        'n_noneligible_pre': int(len(df[(~df['daca_eligible']) & (~df['post_daca'])])),
        'n_noneligible_post': int(len(df[(~df['daca_eligible']) & (df['post_daca'])])),
        'fulltime_eligible_pre': float(df[(df['daca_eligible']) & (~df['post_daca'])]['fulltime'].mean()),
        'fulltime_eligible_post': float(df[(df['daca_eligible']) & (df['post_daca'])]['fulltime'].mean()),
        'fulltime_noneligible_pre': float(df[(~df['daca_eligible']) & (~df['post_daca'])]['fulltime'].mean()),
        'fulltime_noneligible_post': float(df[(~df['daca_eligible']) & (df['post_daca'])]['fulltime'].mean()),
        'simple_did': float(did_simple),
        'baseline_eligible': float(baseline)
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save regression tables
reg_table = pd.DataFrame({
    'Model': ['Model 1 (Basic)', 'Model 2 (Controls)', 'Model 3 (FE)'],
    'DiD_Coef': [model1.params[3], model2.params[3], model3.params[3]],
    'SE': [model1.bse[3], model2.bse[3], model3.bse[3]],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared]
})
reg_table.to_csv('regression_summary.csv', index=False)

# Save descriptive statistics
desc_stats = df.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')

# Save event study
event_df = pd.DataFrame([
    {'year': yr, 'coef': d['coefficient'], 'se': d['se'],
     'ci_low': d['coefficient'] - 1.96*d['se'],
     'ci_high': d['coefficient'] + 1.96*d['se']}
    for yr, d in event_coeffs.items()
]).sort_values('year')
event_df.to_csv('event_study.csv', index=False)

print("Results saved to: results.json, regression_summary.csv, descriptive_stats.csv, event_study.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born non-citizens
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
OUTPUT_DIR = '.'

print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

# =============================================================================
# STEP 1: Load and Filter Data
# =============================================================================
print("\n[1] Loading data...")

# Define columns we need to minimize memory usage
needed_cols = [
    'YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'UHRSWORK',
    'MARST', 'NCHILD', 'STATEFIP'
]

# Read data in chunks to handle large file
chunks = []
chunk_size = 500000

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=needed_cols, chunksize=chunk_size)):
    # Filter early to reduce memory
    # Hispanic Mexican (HISPAN = 1)
    # Born in Mexico (BPL = 200)
    # Not a citizen (CITIZEN = 3)
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
    chunks.append(filtered)
    if (i + 1) % 20 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Filtered data: {len(df):,} observations")

# =============================================================================
# STEP 2: Apply DACA Eligibility Criteria
# =============================================================================
print("\n[2] Applying DACA eligibility criteria...")

# DACA was implemented June 15, 2012
# Eligibility requires arriving before 16th birthday and by 2007

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligible: immigrated before age 16 AND by 2007
# Note: YRIMMIG = 0 means N/A, so we exclude those
df['daca_eligible'] = (
    (df['YRIMMIG'] > 0) &
    (df['age_at_immig'] < 16) &
    (df['YRIMMIG'] <= 2007)
)

print(f"  DACA eligible: {df['daca_eligible'].sum():,} observations")

# Filter to DACA eligible only
df = df[df['daca_eligible']].copy()

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n[3] Defining treatment and control groups...")

# Treatment: ages 26-30 on June 15, 2012 (birth years 1982-1986)
# Control: ages 31-35 on June 15, 2012 (birth years 1977-1981)

# For simplicity, we use birth year to define groups
# Those born 1982-1986 were 26-30 on June 15, 2012
# Those born 1977-1981 were 31-35 on June 15, 2012

df['treatment'] = (df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)
df['control'] = (df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)

# Keep only treatment and control groups
df = df[df['treatment'] | df['control']].copy()

# Binary treatment indicator
df['treat'] = df['treatment'].astype(int)

print(f"  Treatment group (birth 1982-1986): {df['treat'].sum():,}")
print(f"  Control group (birth 1977-1981): {(1 - df['treat']).sum():,}")

# =============================================================================
# STEP 4: Define Time Periods
# =============================================================================
print("\n[4] Defining time periods...")

# Exclude 2012 (mid-year implementation)
df = df[df['YEAR'] != 2012].copy()

# Post-period: 2013-2016
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"  Pre-period (2006-2011): {(df['post'] == 0).sum():,} observations")
print(f"  Post-period (2013-2016): {(df['post'] == 1).sum():,} observations")

# =============================================================================
# STEP 5: Define Outcome Variable
# =============================================================================
print("\n[5] Defining outcome variable...")

# Full-time employment: UHRSWORK >= 35
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Summary statistics
print(f"  Full-time employment rate: {df['fulltime'].mean():.3f}")

# =============================================================================
# STEP 6: Create Additional Variables for Analysis
# =============================================================================
print("\n[6] Creating analysis variables...")

# Interaction term
df['treat_post'] = df['treat'] * df['post']

# Demographics for controls
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)  # Married spouse present/absent
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Education categories
df['educ_hs'] = (df['EDUCD'] >= 62).astype(int)  # High school or more
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or more

# Age at survey (for controls)
df['age'] = df['AGE']
df['age_sq'] = df['age'] ** 2

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n[7] Summary Statistics")
print("-" * 60)

# By group and period
summary = df.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'has_children': 'mean',
    'educ_hs': 'mean',
    'age': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nSummary by Treatment Status and Period:")
print(summary)

# Calculate raw DiD
treat_pre = df[(df['treat'] == 1) & (df['post'] == 0)]['fulltime'].mean()
treat_post = df[(df['treat'] == 1) & (df['post'] == 1)]['fulltime'].mean()
control_pre = df[(df['treat'] == 0) & (df['post'] == 0)]['fulltime'].mean()
control_post = df[(df['treat'] == 0) & (df['post'] == 1)]['fulltime'].mean()

raw_did = (treat_post - treat_pre) - (control_post - control_pre)

print(f"\nRaw DiD Calculation:")
print(f"  Treatment pre:  {treat_pre:.4f}")
print(f"  Treatment post: {treat_post:.4f}")
print(f"  Control pre:    {control_pre:.4f}")
print(f"  Control post:   {control_post:.4f}")
print(f"  Raw DiD effect: {raw_did:.4f}")

# =============================================================================
# STEP 8: Difference-in-Differences Regression
# =============================================================================
print("\n[8] Difference-in-Differences Regression")
print("-" * 60)

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df).fit(cov_type='HC1')
print(model1.summary())

# Model 2: Basic DiD with weights
print("\n--- Model 2: Basic DiD (Weighted) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with Demographic Controls (Weighted) ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + educ_hs',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with demographic controls and age polynomials
print("\n--- Model 4: DiD with Age Controls (Weighted) ---")
model4 = smf.wls('fulltime ~ treat + post + treat_post + female + married + has_children + educ_hs + age + age_sq',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: With year fixed effects
print("\n--- Model 5: DiD with Year Fixed Effects (Weighted) ---")
df['year_fe'] = df['YEAR'].astype(str)
model5 = smf.wls('fulltime ~ treat + C(year_fe) + treat_post + female + married + has_children + educ_hs',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Model 6: With state fixed effects
print("\n--- Model 6: DiD with State Fixed Effects (Weighted) ---")
df['state_fe'] = df['STATEFIP'].astype(str)
model6 = smf.wls('fulltime ~ treat + C(year_fe) + treat_post + female + married + has_children + educ_hs + C(state_fe)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
# Just print key results
print(f"\ntreat_post coefficient: {model6.params['treat_post']:.4f}")
print(f"Standard error: {model6.bse['treat_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treat_post', 0]:.4f}, {model6.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model6.pvalues['treat_post']:.4f}")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs)}")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n[9] Robustness Checks")
print("-" * 60)

# Robustness 1: By gender
print("\n--- Robustness: By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model = smf.wls('fulltime ~ treat + post + treat_post', data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
    print(f"\n{label}:")
    print(f"  DiD estimate: {model.params['treat_post']:.4f} (SE: {model.bse['treat_post']:.4f})")
    print(f"  95% CI: [{model.conf_int().loc['treat_post', 0]:.4f}, {model.conf_int().loc['treat_post', 1]:.4f}]")
    print(f"  N: {int(model.nobs)}")

# Robustness 2: Different age bandwidths
print("\n--- Robustness: Alternative Age Bandwidth (3 years) ---")
df_narrow = df[(df['BIRTHYR'] >= 1980) & (df['BIRTHYR'] <= 1984)].copy()
df_narrow['treat'] = (df_narrow['BIRTHYR'] >= 1982).astype(int)
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']
model_narrow = smf.wls('fulltime ~ treat + post + treat_post', data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"  DiD estimate: {model_narrow.params['treat_post']:.4f} (SE: {model_narrow.bse['treat_post']:.4f})")
print(f"  95% CI: [{model_narrow.conf_int().loc['treat_post', 0]:.4f}, {model_narrow.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  N: {int(model_narrow.nobs)}")

# Robustness 3: Placebo test (pre-period only)
print("\n--- Placebo Test: Pre-2010 vs 2010-2011 ---")
df_pre = df[df['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['treat_placebo'] = df_pre['treat'] * df_pre['placebo_post']
model_placebo = smf.wls('fulltime ~ treat + placebo_post + treat_placebo', data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"  Placebo DiD estimate: {model_placebo.params['treat_placebo']:.4f} (SE: {model_placebo.bse['treat_placebo']:.4f})")
print(f"  p-value: {model_placebo.pvalues['treat_placebo']:.4f}")

# =============================================================================
# STEP 10: Event Study / Dynamic Effects
# =============================================================================
print("\n[10] Event Study Analysis")
print("-" * 60)

# Create year-specific treatment effects (relative to 2011)
df['year_treat'] = df['YEAR'].astype(str) + '_' + df['treat'].astype(str)

# Reference year: 2011 (last pre-treatment year)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_study_results = []

for year in years:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'treat_year_{year}'] = df['treat'] * df[f'year_{year}']

formula = 'fulltime ~ treat + ' + ' + '.join([f'year_{y}' for y in years]) + ' + ' + ' + '.join([f'treat_year_{y}' for y in years])
model_event = smf.wls(formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (reference year: 2011):")
print("-" * 40)
for year in years:
    coef = model_event.params[f'treat_year_{year}']
    se = model_event.bse[f'treat_year_{year}']
    ci_low, ci_high = model_event.conf_int().loc[f'treat_year_{year}']
    pval = model_event.pvalues[f'treat_year_{year}']
    print(f"  {year}: {coef:7.4f} (SE: {se:.4f}) [{ci_low:7.4f}, {ci_high:7.4f}] p={pval:.3f}")

# =============================================================================
# STEP 11: Sample Size Summary
# =============================================================================
print("\n[11] Sample Size Summary")
print("-" * 60)

print("\nObservations by Year and Treatment Status:")
year_treat_counts = df.groupby(['YEAR', 'treat']).size().unstack(fill_value=0)
year_treat_counts.columns = ['Control', 'Treatment']
print(year_treat_counts)

print("\nWeighted Population by Year and Treatment Status:")
year_treat_pop = df.groupby(['YEAR', 'treat'])['PERWT'].sum().unstack(fill_value=0) / 1000
year_treat_pop.columns = ['Control (000s)', 'Treatment (000s)']
print(year_treat_pop.round(1))

# =============================================================================
# STEP 12: Save Results
# =============================================================================
print("\n[12] Saving Results")
print("-" * 60)

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'Weighted DiD', 'With Demographics', 'With Age Controls', 'Year FE', 'Year + State FE'],
    'Coefficient': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model5.params['treat_post'],
        model6.params['treat_post']
    ],
    'Std_Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model5.bse['treat_post'],
        model6.bse['treat_post']
    ],
    'CI_Low': [
        model1.conf_int().loc['treat_post', 0],
        model2.conf_int().loc['treat_post', 0],
        model3.conf_int().loc['treat_post', 0],
        model4.conf_int().loc['treat_post', 0],
        model5.conf_int().loc['treat_post', 0],
        model6.conf_int().loc['treat_post', 0]
    ],
    'CI_High': [
        model1.conf_int().loc['treat_post', 1],
        model2.conf_int().loc['treat_post', 1],
        model3.conf_int().loc['treat_post', 1],
        model4.conf_int().loc['treat_post', 1],
        model5.conf_int().loc['treat_post', 1],
        model6.conf_int().loc['treat_post', 1]
    ],
    'P_Value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model5.pvalues['treat_post'],
        model6.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model6.nobs)
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("Saved regression results to regression_results.csv")

# Save event study results
event_results = []
for year in years:
    event_results.append({
        'Year': year,
        'Coefficient': model_event.params[f'treat_year_{year}'],
        'Std_Error': model_event.bse[f'treat_year_{year}'],
        'CI_Low': model_event.conf_int().loc[f'treat_year_{year}', 0],
        'CI_High': model_event.conf_int().loc[f'treat_year_{year}', 1],
        'P_Value': model_event.pvalues[f'treat_year_{year}']
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Saved event study results to event_study_results.csv")

# Save summary statistics
summary_stats = df.groupby(['treat', 'post']).agg({
    'fulltime': 'mean',
    'female': 'mean',
    'married': 'mean',
    'has_children': 'mean',
    'educ_hs': 'mean',
    'age': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Saved summary statistics to summary_statistics.csv")

# =============================================================================
# STEP 13: Final Summary
# =============================================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE - FINAL SUMMARY")
print("=" * 60)

print(f"""
PREFERRED ESTIMATE (Model 2: Weighted Basic DiD):
-------------------------------------------------
Effect Size:     {model2.params['treat_post']:.4f}
Standard Error:  {model2.bse['treat_post']:.4f}
95% CI:          [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]
P-value:         {model2.pvalues['treat_post']:.4f}
Sample Size:     {int(model2.nobs):,}

INTERPRETATION:
DACA eligibility is associated with a {model2.params['treat_post']*100:.2f} percentage point
change in the probability of full-time employment for DACA-eligible
Hispanic-Mexican Mexican-born non-citizens aged 26-30 relative to
those aged 31-35.

This estimate {'is statistically significant' if model2.pvalues['treat_post'] < 0.05 else 'is not statistically significant'} at the 5% level.
""")

print("=" * 60)
print("END OF ANALYSIS")
print("=" * 60)

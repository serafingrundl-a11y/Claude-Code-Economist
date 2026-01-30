"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States

Author: Anonymous (Replication 16)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = 'data/data.csv'
OUTPUT_DIR = '.'

# ============================================================================
# STEP 1: LOAD AND FILTER DATA
# ============================================================================

print("=" * 70)
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("=" * 70)

print("\n[1] Loading data...")
# Read data in chunks to manage memory
chunks = []
chunksize = 1000000

for chunk in pd.read_csv(DATA_PATH, chunksize=chunksize):
    # Filter to Hispanic-Mexican (HISPAN == 1) and born in Mexico (BPL == 200)
    # and non-citizens (CITIZEN == 3)
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ]
    chunks.append(filtered)
    print(f"  Processed chunk, found {len(filtered)} matching records...")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal records after initial filter: {len(df):,}")

# ============================================================================
# STEP 2: CREATE VARIABLES FOR ANALYSIS
# ============================================================================

print("\n[2] Creating analysis variables...")

# Exclude 2012 (mid-year policy implementation)
df = df[df['YEAR'] != 2012].copy()
print(f"  Records after excluding 2012: {len(df):,}")

# Create post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Create full-time employment indicator (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Age at time of survey
df['age'] = df['AGE']

# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# ============================================================================
# STEP 3: DETERMINE DACA ELIGIBILITY
# ============================================================================

print("\n[3] Determining DACA eligibility...")

# DACA eligibility criteria (as of June 15, 2012):
# 1. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 2. Arrived before age 16
# 3. Continuously present since June 15, 2007 (arrived by 2007)
# 4. Not a citizen (already filtered)

# Criterion 1: Born after June 15, 1981 (under 31 as of June 15, 2012)
# Using birth year and quarter: born in 1982+, or born in 1981 Q3-Q4
# (Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec)
# June 15 falls in Q2, so those born in Q3 or Q4 of 1981 would be under 31
df['under_31_2012'] = (
    (df['BIRTHYR'] >= 1982) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 2: Arrived before age 16
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)

# Criterion 3: Arrived by 2007 (continuous presence since June 15, 2007)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# DACA eligible: meets all criteria
df['daca_eligible'] = (
    (df['under_31_2012'] == 1) &
    (df['arrived_before_16'] == 1) &
    (df['arrived_by_2007'] == 1)
).astype(int)

print(f"  DACA eligible: {df['daca_eligible'].sum():,}")
print(f"  DACA ineligible: {(1 - df['daca_eligible']).sum():,}")

# ============================================================================
# STEP 4: RESTRICT SAMPLE FOR ANALYSIS
# ============================================================================

print("\n[4] Restricting sample for analysis...")

# Restrict to working-age adults (18-45)
# This ensures we have comparable age cohorts around the DACA age cutoff
df_analysis = df[(df['age'] >= 18) & (df['age'] <= 45)].copy()
print(f"  Records with age 18-45: {len(df_analysis):,}")

# Also filter to valid YRIMMIG (not 0)
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
print(f"  Records with valid immigration year: {len(df_analysis):,}")

# ============================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# ============================================================================

print("\n[5] Generating descriptive statistics...")

# Summary by treatment status and period
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'age': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Treatment Status and Period:")
print(summary_stats)

# Year-by-year trends
yearly_stats = df_analysis.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).round(4)
print("\nYearly Full-time Employment Rate by Eligibility Status:")
print(yearly_stats)

# ============================================================================
# STEP 6: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# ============================================================================

print("\n[6] Running Difference-in-Differences regression...")

# Basic DiD: fulltime = β0 + β1*post + β2*eligible + β3*post*eligible + ε
df_analysis['post_x_eligible'] = df_analysis['post'] * df_analysis['daca_eligible']

# Model 1: Basic DiD (unweighted)
model1 = smf.ols('fulltime ~ post + daca_eligible + post_x_eligible', data=df_analysis).fit()

print("\n--- Model 1: Basic DiD (OLS, unweighted) ---")
print(model1.summary())

# Model 2: DiD with weights
model2 = smf.wls('fulltime ~ post + daca_eligible + post_x_eligible',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n--- Model 2: DiD with survey weights (WLS) ---")
print(model2.summary())

# Model 3: DiD with controls
df_analysis['age_sq'] = df_analysis['age'] ** 2
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)

model3 = smf.wls('fulltime ~ post + daca_eligible + post_x_eligible + age + age_sq + male',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n--- Model 3: DiD with controls (WLS) ---")
print(model3.summary())

# Model 4: DiD with year fixed effects
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model4 = smf.wls('fulltime ~ C(YEAR) + daca_eligible + post_x_eligible + age + age_sq + male',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n--- Model 4: DiD with year FE and controls (WLS) ---")
print(model4.summary())

# Model 5: Add state fixed effects
model5 = smf.wls('fulltime ~ C(YEAR) + C(STATEFIP) + daca_eligible + post_x_eligible + age + age_sq + male',
                 data=df_analysis, weights=df_analysis['PERWT']).fit()

print("\n--- Model 5: DiD with year and state FE (WLS) ---")
# Only print coefficients of interest
print("\nKey coefficients from Model 5:")
print(f"  post_x_eligible (DiD estimate): {model5.params['post_x_eligible']:.6f}")
print(f"  SE: {model5.bse['post_x_eligible']:.6f}")
print(f"  t-stat: {model5.tvalues['post_x_eligible']:.4f}")
print(f"  p-value: {model5.pvalues['post_x_eligible']:.6f}")
conf_int = model5.conf_int().loc['post_x_eligible']
print(f"  95% CI: [{conf_int[0]:.6f}, {conf_int[1]:.6f}]")

# ============================================================================
# STEP 7: ROBUST STANDARD ERRORS
# ============================================================================

print("\n[7] Computing robust standard errors...")

# Model with robust (HC1) standard errors
model_robust = smf.wls('fulltime ~ C(YEAR) + C(STATEFIP) + daca_eligible + post_x_eligible + age + age_sq + male',
                       data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nModel with robust standard errors (HC1):")
print(f"  post_x_eligible: {model_robust.params['post_x_eligible']:.6f}")
print(f"  Robust SE: {model_robust.bse['post_x_eligible']:.6f}")
print(f"  t-stat: {model_robust.tvalues['post_x_eligible']:.4f}")
print(f"  p-value: {model_robust.pvalues['post_x_eligible']:.6f}")
conf_int_robust = model_robust.conf_int().loc['post_x_eligible']
print(f"  95% CI: [{conf_int_robust[0]:.6f}, {conf_int_robust[1]:.6f}]")

# ============================================================================
# STEP 8: EVENT STUDY / DYNAMIC EFFECTS
# ============================================================================

print("\n[8] Running event study specification...")

# Create year dummies interacted with eligibility
for year in df_analysis['YEAR'].unique():
    df_analysis[f'eligible_x_{year}'] = (
        (df_analysis['YEAR'] == year) * df_analysis['daca_eligible']
    ).astype(int)

# Use 2011 as reference year (last pre-treatment year)
year_vars = [f'eligible_x_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2011]
formula = 'fulltime ~ C(YEAR) + C(STATEFIP) + daca_eligible + ' + ' + '.join(year_vars) + ' + age + age_sq + male'

model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in sorted(df_analysis['YEAR'].unique()):
    if year != 2011:
        coef = model_event.params[f'eligible_x_{year}']
        se = model_event.bse[f'eligible_x_{year}']
        print(f"  {year}: {coef:.6f} (SE: {se:.6f})")

# ============================================================================
# STEP 9: SAVE RESULTS
# ============================================================================

print("\n[9] Saving results...")

# Create results summary
results_summary = {
    'Model': ['Basic DiD (OLS)', 'DiD with weights', 'DiD + controls',
              'DiD + year FE', 'DiD + year & state FE', 'DiD robust SE'],
    'DiD Coefficient': [model1.params['post_x_eligible'],
                       model2.params['post_x_eligible'],
                       model3.params['post_x_eligible'],
                       model4.params['post_x_eligible'],
                       model5.params['post_x_eligible'],
                       model_robust.params['post_x_eligible']],
    'Standard Error': [model1.bse['post_x_eligible'],
                      model2.bse['post_x_eligible'],
                      model3.bse['post_x_eligible'],
                      model4.bse['post_x_eligible'],
                      model5.bse['post_x_eligible'],
                      model_robust.bse['post_x_eligible']],
    'p-value': [model1.pvalues['post_x_eligible'],
               model2.pvalues['post_x_eligible'],
               model3.pvalues['post_x_eligible'],
               model4.pvalues['post_x_eligible'],
               model5.pvalues['post_x_eligible'],
               model_robust.pvalues['post_x_eligible']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model_robust.nobs)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv(f'{OUTPUT_DIR}/regression_results.csv', index=False)
print("\nRegression results saved to regression_results.csv")

# Save descriptive statistics
yearly_stats_reset = df_analysis.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).reset_index()
yearly_stats_reset.to_csv(f'{OUTPUT_DIR}/yearly_stats.csv', index=False)
print("Yearly statistics saved to yearly_stats.csv")

# Save event study coefficients
event_study_results = []
for year in sorted(df_analysis['YEAR'].unique()):
    if year == 2011:
        event_study_results.append({
            'year': year,
            'coefficient': 0,
            'std_error': 0,
            'reference': True
        })
    else:
        event_study_results.append({
            'year': year,
            'coefficient': model_event.params[f'eligible_x_{year}'],
            'std_error': model_event.bse[f'eligible_x_{year}'],
            'reference': False
        })
event_study_df = pd.DataFrame(event_study_results)
event_study_df.to_csv(f'{OUTPUT_DIR}/event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# ============================================================================
# STEP 10: FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\nPreferred Estimate (Model 5 with robust SE):")
print(f"  Effect of DACA eligibility on full-time employment: {model_robust.params['post_x_eligible']:.4f}")
print(f"  Standard Error (robust): {model_robust.bse['post_x_eligible']:.4f}")
print(f"  95% Confidence Interval: [{conf_int_robust[0]:.4f}, {conf_int_robust[1]:.4f}]")
print(f"  p-value: {model_robust.pvalues['post_x_eligible']:.4f}")
print(f"  Sample Size: {int(model_robust.nobs):,}")

print(f"\nInterpretation:")
effect = model_robust.params['post_x_eligible']
if effect > 0:
    print(f"  DACA eligibility is associated with a {effect*100:.2f} percentage point")
    print(f"  INCREASE in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(effect)*100:.2f} percentage point")
    print(f"  DECREASE in the probability of full-time employment.")

if model_robust.pvalues['post_x_eligible'] < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

# Additional sample statistics
print(f"\n\nSample Composition:")
print(f"  Total observations: {len(df_analysis):,}")
print(f"  DACA-eligible observations: {df_analysis['daca_eligible'].sum():,}")
print(f"  DACA-ineligible observations: {(1-df_analysis['daca_eligible']).sum():,}")
print(f"  Pre-period observations (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"  Post-period observations (2013-2016): {(df_analysis['post']==1).sum():,}")

# Mean outcomes
print(f"\n\nMean Full-time Employment Rates:")
for eligible in [0, 1]:
    for post in [0, 1]:
        subset = df_analysis[(df_analysis['daca_eligible']==eligible) & (df_analysis['post']==post)]
        mean_ft = subset['fulltime'].mean()
        n = len(subset)
        eligible_str = "Eligible" if eligible else "Ineligible"
        period_str = "Post" if post else "Pre"
        print(f"  {eligible_str}, {period_str}: {mean_ft:.4f} (n={n:,})")

# Simple DiD calculation
pre_elig = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_elig = df_analysis[(df_analysis['daca_eligible']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_inelig = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_inelig = df_analysis[(df_analysis['daca_eligible']==0) & (df_analysis['post']==1)]['fulltime'].mean()

simple_did = (post_elig - pre_elig) - (post_inelig - pre_inelig)
print(f"\n\nSimple DiD Calculation (unweighted, no controls):")
print(f"  Change for eligible: {post_elig:.4f} - {pre_elig:.4f} = {post_elig - pre_elig:.4f}")
print(f"  Change for ineligible: {post_inelig:.4f} - {pre_inelig:.4f} = {post_inelig - pre_inelig:.4f}")
print(f"  DiD = {simple_did:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

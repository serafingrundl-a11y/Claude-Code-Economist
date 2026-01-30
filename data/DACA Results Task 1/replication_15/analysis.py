"""
DACA Replication Analysis
Research Question: What was the causal impact of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican, Mexican-born people living in the US?

This script:
1. Loads ACS data (2006-2016)
2. Identifies DACA-eligible individuals
3. Implements a difference-in-differences design
4. Estimates the effect on full-time employment (working 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_15")

print("="*70)
print("DACA REPLICATION ANALYSIS")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. Loading ACS data (2006-2016)...")

# Load data in chunks due to large size, then filter
# We only need Hispanic-Mexican (HISPAN=1) born in Mexico (BPL=200)

chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', chunksize=chunk_size, low_memory=False):
    # Filter to Hispanic-Mexican (HISPAN=1) AND born in Mexico (BPL=200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    print(f"  Processed chunk... current sample size: {sum(len(c) for c in chunks)}")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal observations after initial filter (Hispanic-Mexican, born in Mexico): {len(df)}")

# =============================================================================
# 2. DEFINE DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n2. Defining DACA eligibility criteria...")

"""
DACA eligibility requirements (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Under age 31 as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007
4. Present in US on June 15, 2012
5. Not a citizen (undocumented - we proxy this with non-citizen status)

Key dates:
- DACA announced: June 15, 2012
- Applications started: August 15, 2012
- We examine outcomes in 2013-2016

Identification strategy: Difference-in-Differences
- Treatment group: Those who meet DACA eligibility criteria (non-citizens meeting age/arrival requirements)
- Control group: Similar non-citizens who don't meet criteria (e.g., arrived after age 16 or too old)
- Pre-period: 2006-2011 (before DACA)
- Post-period: 2013-2016 (after DACA, excluding 2012 which is ambiguous)
"""

# Calculate age at arrival in US
# YRIMMIG is year of immigration
# BIRTHYR is birth year
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Age in reference year (2012 - when DACA was implemented)
df['age_2012'] = 2012 - df['BIRTHYR']

# Calculate age at time of survey
df['age_at_survey'] = df['YEAR'] - df['BIRTHYR']

# DACA eligibility criteria:
# 1. Arrived before age 16
df['arrived_before_16'] = df['age_at_arrival'] < 16

# 2. Under 31 on June 15, 2012 (born after June 15, 1981)
#    Conservative: born in 1982 or later is definitely eligible
#    For 1981, need to check birth quarter (Q1-Q2 not eligible, Q3-Q4 might be)
#    We'll be slightly conservative: born >= 1982
df['under_31_2012'] = df['BIRTHYR'] >= 1982

# 3. Continuous US presence since June 15, 2007
#    YRIMMIG <= 2007 means they were in US by 2007
df['in_us_by_2007'] = df['YRIMMIG'] <= 2007

# 4. Non-citizen (CITIZEN == 3 means not a citizen)
#    This is our proxy for undocumented status
df['non_citizen'] = df['CITIZEN'] == 3

# Define treatment group: meets all DACA eligibility criteria
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_2012'] &
    df['in_us_by_2007'] &
    df['non_citizen']
)

# Define comparison/control group: Similar non-citizens but NOT DACA-eligible
# Option 1: Non-citizens who arrived at age 16+ (failed the childhood arrival requirement)
df['control_arrived_older'] = (
    (df['age_at_arrival'] >= 16) &
    (df['age_at_arrival'] <= 25) &  # Keep similar ages at arrival
    df['non_citizen'] &
    df['in_us_by_2007']
)

# Option 2: Non-citizens who are too old (born before 1982)
df['control_too_old'] = (
    df['arrived_before_16'] &
    (df['BIRTHYR'] >= 1972) & (df['BIRTHYR'] <= 1981) &  # Slightly older cohort
    df['non_citizen'] &
    df['in_us_by_2007']
)

# Combined control: those who narrowly miss eligibility
df['control_group'] = df['control_arrived_older'] | df['control_too_old']

# =============================================================================
# 3. CREATE OUTCOME VARIABLE
# =============================================================================
print("\n3. Creating outcome variable...")

# Full-time employment: usually works 35+ hours per week
# UHRSWORK: usual hours worked per week (0 = N/A or not working)
df['fulltime_employed'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# =============================================================================
# 4. DEFINE TIME PERIODS
# =============================================================================
print("\n4. Defining time periods...")

# Pre-DACA: 2006-2011
# Post-DACA: 2013-2016 (exclude 2012 - implementation year)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Exclude 2012 from analysis (ambiguous period)
df = df[df['YEAR'] != 2012]

print(f"  Observations after excluding 2012: {len(df)}")

# =============================================================================
# 5. SAMPLE RESTRICTIONS
# =============================================================================
print("\n5. Applying sample restrictions...")

# Working age population (18-64)
# For treatment group, we need those 18+ at time of survey
df['working_age'] = (df['age_at_survey'] >= 18) & (df['age_at_survey'] <= 64)

# Create analysis sample: either treatment or control group, working age
df_analysis = df[(df['daca_eligible'] | df['control_group']) & df['working_age']].copy()

print(f"  Analysis sample size: {len(df_analysis)}")
print(f"  Treatment (DACA-eligible): {df_analysis['daca_eligible'].sum()}")
print(f"  Control group: {df_analysis['control_group'].sum()}")

# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n6. Generating descriptive statistics...")

# Summary by treatment status and period
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime_employed': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Female proportion
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by Treatment and Period:")
print(summary_stats)

# Save descriptive stats
summary_stats.to_csv('descriptive_stats.csv')

# Year-by-year trends
yearly_stats = df_analysis.groupby(['YEAR', 'daca_eligible']).agg({
    'fulltime_employed': 'mean',
    'PERWT': 'sum'
}).round(4)
print("\nYearly Full-Time Employment Rates:")
print(yearly_stats)

yearly_stats.to_csv('yearly_trends.csv')

# =============================================================================
# 7. DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================
print("\n7. Running Difference-in-Differences estimation...")

# Create variables for regression
df_analysis['treat'] = df_analysis['daca_eligible'].astype(int)
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)

# Create age groups
df_analysis['age_group'] = pd.cut(df_analysis['age_at_survey'],
                                   bins=[17, 24, 34, 44, 54, 65],
                                   labels=['18-24', '25-34', '35-44', '45-54', '55-64'])

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (no controls, unweighted) ---")
model1 = smf.ols('fulltime_employed ~ treat + post + treat_post',
                  data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls (unweighted) ---")
model2 = smf.ols('fulltime_employed ~ treat + post + treat_post + female + married + C(age_group)',
                  data=df_analysis).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with state and year fixed effects
print("\n--- Model 3: DiD with state and year fixed effects ---")
model3 = smf.ols('fulltime_employed ~ treat + treat_post + female + married + C(age_group) + C(STATEFIP) + C(YEAR)',
                  data=df_analysis).fit(cov_type='HC1')

print("\nModel 3 - Key coefficients:")
print(f"  Treatment (DACA Eligible): {model3.params['treat']:.4f}")
print(f"  Treat x Post (DiD estimate): {model3.params['treat_post']:.4f}")
print(f"  Std Error: {model3.bse['treat_post']:.4f}")
print(f"  t-stat: {model3.tvalues['treat_post']:.4f}")
print(f"  p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: Weighted DiD (using person weights)
print("\n--- Model 4: Weighted DiD with full controls ---")

# Use WLS with formula interface
model4 = smf.wls('fulltime_employed ~ treat + treat_post + female + married + C(age_group) + C(STATEFIP) + C(YEAR)',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nModel 4 - Key coefficients (Weighted):")
print(f"  Treatment (DACA Eligible): {model4.params['treat']:.4f}")
print(f"  Treat x Post (DiD estimate): {model4.params['treat_post']:.4f}")
print(f"  Std Error: {model4.bse['treat_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n8. Robustness checks...")

# Alternative control group: only those who arrived at age 16+
print("\n--- Robustness 1: Control = arrived at 16+ only ---")
df_robust1 = df_analysis[df_analysis['daca_eligible'] | df_analysis['control_arrived_older']].copy()
df_robust1['treat'] = df_robust1['daca_eligible'].astype(int)
df_robust1['treat_post'] = df_robust1['treat'] * df_robust1['post']
model_r1 = smf.ols('fulltime_employed ~ treat + post + treat_post + female + married + C(YEAR) + C(STATEFIP)',
                   data=df_robust1).fit(cov_type='HC1')
print(f"  DiD estimate: {model_r1.params['treat_post']:.4f} (SE: {model_r1.bse['treat_post']:.4f})")

# Alternative control group: only those too old
print("\n--- Robustness 2: Control = too old (born 1972-1981) only ---")
df_robust2 = df_analysis[df_analysis['daca_eligible'] | df_analysis['control_too_old']].copy()
df_robust2['treat'] = df_robust2['daca_eligible'].astype(int)
df_robust2['treat_post'] = df_robust2['treat'] * df_robust2['post']
model_r2 = smf.ols('fulltime_employed ~ treat + post + treat_post + female + married + C(YEAR) + C(STATEFIP)',
                   data=df_robust2).fit(cov_type='HC1')
print(f"  DiD estimate: {model_r2.params['treat_post']:.4f} (SE: {model_r2.bse['treat_post']:.4f})")

# Outcome: Any employment
print("\n--- Robustness 3: Outcome = Any employment ---")
model_r3 = smf.ols('employed ~ treat + post + treat_post + female + married + C(YEAR) + C(STATEFIP)',
                   data=df_analysis).fit(cov_type='HC1')
print(f"  DiD estimate: {model_r3.params['treat_post']:.4f} (SE: {model_r3.bse['treat_post']:.4f})")

# =============================================================================
# 9. EVENT STUDY
# =============================================================================
print("\n9. Event study analysis...")

# Create year-specific treatment effects (relative to 2011)
years = sorted(df_analysis['YEAR'].unique())
ref_year = 2011  # Last pre-treatment year

for yr in years:
    if yr != ref_year:
        df_analysis[f'treat_x_{yr}'] = df_analysis['treat'] * (df_analysis['YEAR'] == yr).astype(int)

# Run event study regression
event_formula = 'fulltime_employed ~ treat + female + married + C(YEAR) + C(STATEFIP)'
for yr in years:
    if yr != ref_year:
        event_formula += f' + treat_x_{yr}'

model_event = smf.ols(event_formula, data=df_analysis).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
event_results = []
for yr in years:
    if yr != ref_year:
        coef = model_event.params[f'treat_x_{yr}']
        se = model_event.bse[f'treat_x_{yr}']
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        print(f"  Year {yr}: {coef:.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}]")
        event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# =============================================================================
# 10. SAVE FINAL RESULTS
# =============================================================================
print("\n10. Saving results...")

# Main results table
results = {
    'Model': ['Basic DiD', 'DiD + Demographics', 'DiD + FE', 'Weighted DiD + FE'],
    'DiD_Estimate': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post']
    ],
    'Std_Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post']
    ],
    'p_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs)
    ]
}

results_df = pd.DataFrame(results)
results_df['CI_lower'] = results_df['DiD_Estimate'] - 1.96 * results_df['Std_Error']
results_df['CI_upper'] = results_df['DiD_Estimate'] + 1.96 * results_df['Std_Error']
print("\n" + "="*70)
print("MAIN RESULTS TABLE")
print("="*70)
print(results_df.to_string(index=False))
results_df.to_csv('main_results.csv', index=False)

# Robustness results
robust_results = {
    'Specification': ['Control: Arrived 16+', 'Control: Too Old', 'Outcome: Any Employment'],
    'DiD_Estimate': [
        model_r1.params['treat_post'],
        model_r2.params['treat_post'],
        model_r3.params['treat_post']
    ],
    'Std_Error': [
        model_r1.bse['treat_post'],
        model_r2.bse['treat_post'],
        model_r3.bse['treat_post']
    ],
    'N': [
        int(model_r1.nobs),
        int(model_r2.nobs),
        int(model_r3.nobs)
    ]
}

robust_df = pd.DataFrame(robust_results)
robust_df['CI_lower'] = robust_df['DiD_Estimate'] - 1.96 * robust_df['Std_Error']
robust_df['CI_upper'] = robust_df['DiD_Estimate'] + 1.96 * robust_df['Std_Error']
print("\n" + "="*70)
print("ROBUSTNESS RESULTS")
print("="*70)
print(robust_df.to_string(index=False))
robust_df.to_csv('robustness_results.csv', index=False)

# =============================================================================
# 11. SUMMARY
# =============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

# Use Model 3 as preferred (unweighted with fixed effects - more standard)
preferred_estimate = model3.params['treat_post']
preferred_se = model3.bse['treat_post']
preferred_ci_low = preferred_estimate - 1.96 * preferred_se
preferred_ci_high = preferred_estimate + 1.96 * preferred_se
sample_size = int(model3.nobs)

print(f"""
PREFERRED ESTIMATE (DiD with State and Year FE):
  Effect Size: {preferred_estimate:.4f} ({preferred_estimate*100:.2f} percentage points)
  Standard Error: {preferred_se:.4f}
  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]
  Sample Size: {sample_size:,}

INTERPRETATION:
  DACA eligibility is associated with a {preferred_estimate*100:.2f} percentage point
  increase in the probability of full-time employment (working 35+ hours/week)
  among Hispanic-Mexican individuals born in Mexico.

  The effect is {"statistically significant" if model3.pvalues['treat_post'] < 0.05 else "not statistically significant"} at the 5% level
  (p-value: {model3.pvalues['treat_post']:.4f})
""")

# Save summary for report
with open('analysis_summary.txt', 'w') as f:
    f.write(f"Preferred Estimate: {preferred_estimate:.4f}\n")
    f.write(f"Standard Error: {preferred_se:.4f}\n")
    f.write(f"95% CI Lower: {preferred_ci_low:.4f}\n")
    f.write(f"95% CI Upper: {preferred_ci_high:.4f}\n")
    f.write(f"Sample Size: {sample_size}\n")
    f.write(f"p-value: {model3.pvalues['treat_post']:.4f}\n")

# Save detailed stats for plotting
yearly_means = df_analysis.groupby(['YEAR', 'treat'])['fulltime_employed'].mean().unstack()
yearly_means.columns = ['Control', 'Treatment']
yearly_means.to_csv('yearly_means_for_plot.csv')

print("\nAnalysis complete! Results saved to CSV files.")

"""
DACA Replication Analysis - Task 75
Independent replication of DACA eligibility impact on full-time employment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = "data/data.csv"
OUTPUT_DIR = "."

# DACA was implemented on June 15, 2012
DACA_DATE = (2012, 6, 15)

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_age_on_date(birth_year, birth_qtr, target_year, target_month, target_day):
    """
    Calculate age on a specific date given birth year and quarter.
    Birth quarters: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
    We use midpoint of quarter for birth month approximation.
    """
    # Approximate birth month from quarter (midpoint)
    qtr_to_month = {1: 2, 2: 5, 3: 8, 4: 11}  # Feb, May, Aug, Nov
    birth_month = birth_qtr.map(qtr_to_month)

    # Calculate age
    age = target_year - birth_year
    # Adjust if birthday hasn't occurred yet
    # If target month < birth month, subtract 1
    # If target month == birth month and target day < 15, subtract 1
    adjustment = ((target_month < birth_month) |
                  ((target_month == birth_month) & (target_day < 15))).astype(int)
    return age - adjustment

def calculate_age_at_immigration(birth_year, yrimmig):
    """Calculate age when person immigrated."""
    return yrimmig - birth_year

print("=" * 70)
print("DACA REPLICATION ANALYSIS - TASK 75")
print("=" * 70)

# ============================================================================
# Step 1: Load and Filter Data
# ============================================================================
print("\nStep 1: Loading data...")

# Define columns we need
usecols = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EMPSTAT', 'UHRSWORK', 'EDUC', 'EDUCD', 'STATEFIP', 'LABFORCE']

# Read data in chunks for memory efficiency
print("Reading data (this may take a while due to file size)...")
chunks = []
for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=1000000):
    # Initial filter: Hispanic-Mexican born in Mexico, non-citizen
    mask = (
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &    # Born in Mexico
        (chunk['CITIZEN'] == 3)    # Not a citizen
    )
    filtered = chunk[mask].copy()
    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"After initial filter (Hispanic-Mexican, Mexico-born, non-citizen): {len(df):,} observations")

# ============================================================================
# Step 2: Calculate Ages and Apply DACA Eligibility Criteria
# ============================================================================
print("\nStep 2: Applying DACA eligibility criteria...")

# Calculate age on June 15, 2012 (DACA implementation date)
df['age_at_daca'] = calculate_age_on_date(
    df['BIRTHYR'], df['BIRTHQTR'], 2012, 6, 15
)

# Calculate age at immigration
df['age_at_immigration'] = calculate_age_at_immigration(df['BIRTHYR'], df['YRIMMIG'])

# DACA eligibility criteria (other than age):
# 1. Arrived before 16th birthday: age_at_immigration < 16
# 2. Arrived by June 15, 2007 (lived continuously since): YRIMMIG <= 2007
# Note: We cannot directly verify presence on June 15, 2012, so we assume
# anyone in the ACS sample who meets other criteria was present

eligibility_mask = (
    (df['age_at_immigration'] < 16) &  # Arrived before 16th birthday
    (df['YRIMMIG'] <= 2007) &           # Arrived by June 15, 2007
    (df['YRIMMIG'] > 0)                 # Valid immigration year
)

df_eligible = df[eligibility_mask].copy()
print(f"After DACA eligibility filter (arrived <16, by 2007): {len(df_eligible):,} observations")

# ============================================================================
# Step 3: Define Treatment and Control Groups
# ============================================================================
print("\nStep 3: Defining treatment and control groups...")

# Treatment group: Ages 26-30 on June 15, 2012
# Control group: Ages 31-35 on June 15, 2012

df_eligible['treated'] = (
    (df_eligible['age_at_daca'] >= 26) &
    (df_eligible['age_at_daca'] <= 30)
).astype(int)

df_eligible['control'] = (
    (df_eligible['age_at_daca'] >= 31) &
    (df_eligible['age_at_daca'] <= 35)
).astype(int)

# Keep only treatment and control groups
df_analysis = df_eligible[
    (df_eligible['treated'] == 1) | (df_eligible['control'] == 1)
].copy()

print(f"Treatment group (ages 26-30 at DACA): {df_analysis['treated'].sum():,}")
print(f"Control group (ages 31-35 at DACA): {(df_analysis['control'] == 1).sum():,}")
print(f"Total analysis sample: {len(df_analysis):,}")

# ============================================================================
# Step 4: Define Outcome Variable and Time Periods
# ============================================================================
print("\nStep 4: Creating outcome variable and time indicators...")

# Outcome: Full-time employment (35+ hours per week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator
# Pre-period: 2006-2011
# Post-period: 2013-2016 (excluding 2012 due to mid-year implementation)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Exclude 2012 (mid-year implementation makes it ambiguous)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()

print(f"After excluding 2012: {len(df_analysis):,} observations")
print(f"Pre-period observations (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period observations (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# Step 5: Summary Statistics
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

# Overall summary
print("\n--- Overall Sample Characteristics ---")
print(f"Total observations: {len(df_analysis):,}")
print(f"Unique years: {sorted(df_analysis['YEAR'].unique())}")
print(f"Treatment group (26-30 at DACA): {(df_analysis['treated'] == 1).sum():,}")
print(f"Control group (31-35 at DACA): {(df_analysis['treated'] == 0).sum():,}")

# Full-time employment rates by group and period
print("\n--- Full-Time Employment Rates (Unweighted) ---")
for period, period_name in [(0, 'Pre-DACA (2006-2011)'), (1, 'Post-DACA (2013-2016)')]:
    for treat, group_name in [(1, 'Treatment (26-30)'), (0, 'Control (31-35)')]:
        mask = (df_analysis['post'] == period) & (df_analysis['treated'] == treat)
        rate = df_analysis.loc[mask, 'fulltime'].mean()
        n = mask.sum()
        print(f"  {group_name}, {period_name}: {rate:.4f} (n={n:,})")

# Weighted employment rates
print("\n--- Full-Time Employment Rates (Weighted) ---")
for period, period_name in [(0, 'Pre-DACA (2006-2011)'), (1, 'Post-DACA (2013-2016)')]:
    for treat, group_name in [(1, 'Treatment (26-30)'), (0, 'Control (31-35)')]:
        mask = (df_analysis['post'] == period) & (df_analysis['treated'] == treat)
        subset = df_analysis.loc[mask]
        rate = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = mask.sum()
        pop = subset['PERWT'].sum()
        print(f"  {group_name}, {period_name}: {rate:.4f} (n={n:,}, pop={pop:,.0f})")

# ============================================================================
# Step 6: Difference-in-Differences Analysis
# ============================================================================
print("\n" + "=" * 70)
print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treated'] * df_analysis['post']

# --- Model 1: Basic DiD (unweighted) ---
print("\n--- Model 1: Basic DiD (Unweighted OLS) ---")
model1 = smf.ols('fulltime ~ treated + post + treat_post', data=df_analysis).fit()
print(model1.summary())

# --- Model 2: Basic DiD (weighted) ---
print("\n--- Model 2: Basic DiD (Weighted OLS) ---")
model2 = smf.wls('fulltime ~ treated + post + treat_post',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(model2.summary())

# --- Model 3: DiD with covariates (weighted) ---
print("\n--- Model 3: DiD with Covariates (Weighted OLS) ---")
# Add covariates: sex, education, marital status, state fixed effects
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUCD'] >= 62).astype(int)  # HS diploma or higher
df_analysis['educ_college'] = (df_analysis['EDUCD'] >= 101).astype(int)  # College or higher

model3 = smf.wls('fulltime ~ treated + post + treat_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(model3.summary())

# --- Model 4: DiD with year fixed effects (weighted) ---
print("\n--- Model 4: DiD with Year Fixed Effects (Weighted OLS) ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treated + C(year_factor) + treat_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit()
print(f"\nDiD Coefficient (treat_post): {model4.params['treat_post']:.6f}")
print(f"Standard Error: {model4.bse['treat_post']:.6f}")
print(f"t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"p-value: {model4.pvalues['treat_post']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.6f}, {model4.conf_int().loc['treat_post', 1]:.6f}]")

# --- Model 5: DiD with state and year fixed effects ---
print("\n--- Model 5: DiD with State and Year Fixed Effects (Preferred Model) ---")
df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treated + C(year_factor) + C(state_factor) + treat_post + female + married + educ_hs + educ_college',
                 data=df_analysis,
                 weights=df_analysis['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print(f"\nPreferred Model Results:")
print(f"DiD Coefficient (treat_post): {model5.params['treat_post']:.6f}")
print(f"Clustered Standard Error (state): {model5.bse['treat_post']:.6f}")
print(f"t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"p-value: {model5.pvalues['treat_post']:.6f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.6f}, {model5.conf_int().loc['treat_post', 1]:.6f}]")
print(f"N: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

# ============================================================================
# Step 7: Robustness Checks
# ============================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)

# --- Check 1: Separate effects by gender ---
print("\n--- Gender-Specific Effects ---")
for gender, gender_name in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender].copy()
    model_gender = smf.wls('fulltime ~ treated + C(year_factor) + treat_post + married + educ_hs + educ_college',
                           data=df_gender,
                           weights=df_gender['PERWT']).fit()
    print(f"  {gender_name}: DiD = {model_gender.params['treat_post']:.4f} (SE: {model_gender.bse['treat_post']:.4f}), n={len(df_gender):,}")

# --- Check 2: Placebo test - use 2009 as fake treatment year ---
print("\n--- Placebo Test (Fake Treatment in 2009) ---")
df_pre = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['fake_treat_post'] = df_pre['treated'] * df_pre['fake_post']

model_placebo = smf.wls('fulltime ~ treated + fake_post + fake_treat_post',
                        data=df_pre,
                        weights=df_pre['PERWT']).fit()
print(f"Placebo DiD Coefficient: {model_placebo.params['fake_treat_post']:.6f}")
print(f"Standard Error: {model_placebo.bse['fake_treat_post']:.6f}")
print(f"p-value: {model_placebo.pvalues['fake_treat_post']:.6f}")

# --- Check 3: Year-by-year effects ---
print("\n--- Year-by-Year Effects ---")
yearly_effects = []
for year in sorted(df_analysis['YEAR'].unique()):
    df_year = df_analysis[df_analysis['YEAR'] == year].copy()
    treat_mean = np.average(df_year[df_year['treated'] == 1]['fulltime'],
                           weights=df_year[df_year['treated'] == 1]['PERWT'])
    control_mean = np.average(df_year[df_year['treated'] == 0]['fulltime'],
                             weights=df_year[df_year['treated'] == 0]['PERWT'])
    diff = treat_mean - control_mean
    yearly_effects.append({'year': year, 'treatment': treat_mean, 'control': control_mean, 'diff': diff})
    print(f"  {year}: Treatment={treat_mean:.4f}, Control={control_mean:.4f}, Diff={diff:.4f}")

# ============================================================================
# Step 8: Event Study Analysis
# ============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY ANALYSIS")
print("=" * 70)

# Create year dummies interacted with treatment
df_analysis['year_relative'] = df_analysis['YEAR'] - 2012
years = sorted(df_analysis['year_relative'].unique())
ref_year = -1  # Use 2011 as reference year

print("\nEvent study coefficients (relative to 2011):")
event_study_results = []

# Create column names that don't have negative numbers
for yr in years:
    if yr != ref_year:
        col_suffix = f'neg{abs(yr)}' if yr < 0 else f'pos{yr}'
        df_analysis[f'year_{col_suffix}'] = (df_analysis['year_relative'] == yr).astype(int)
        df_analysis[f'treat_year_{col_suffix}'] = df_analysis['treated'] * df_analysis[f'year_{col_suffix}']

# Build formula with all interactions except reference year
year_dummies = ' + '.join([f'year_neg{abs(yr)}' if yr < 0 else f'year_pos{yr}' for yr in years if yr != ref_year])
interactions = ' + '.join([f'treat_year_neg{abs(yr)}' if yr < 0 else f'treat_year_pos{yr}' for yr in years if yr != ref_year])
formula = f'fulltime ~ treated + {year_dummies} + {interactions} + female + married + educ_hs + educ_college'

model_event = smf.wls(formula, data=df_analysis, weights=df_analysis['PERWT']).fit()

for yr in years:
    if yr != ref_year:
        col_suffix = f'neg{abs(yr)}' if yr < 0 else f'pos{yr}'
        coef = model_event.params[f'treat_year_{col_suffix}']
        se = model_event.bse[f'treat_year_{col_suffix}']
        pval = model_event.pvalues[f'treat_year_{col_suffix}']
        actual_year = yr + 2012
        event_study_results.append({
            'year': actual_year,
            'relative_year': yr,
            'coefficient': coef,
            'se': se,
            'pvalue': pval
        })
        sig = '*' if pval < 0.1 else ''
        sig = '**' if pval < 0.05 else sig
        sig = '***' if pval < 0.01 else sig
        print(f"  Year {actual_year} (t={yr:+d}): {coef:.4f} (SE: {se:.4f}) {sig}")

# ============================================================================
# Step 9: Save Results
# ============================================================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

# Create results summary
results_summary = {
    'sample_size': int(model5.nobs),
    'treatment_n': int((df_analysis['treated'] == 1).sum()),
    'control_n': int((df_analysis['treated'] == 0).sum()),
    'did_coefficient': model5.params['treat_post'],
    'did_se': model5.bse['treat_post'],
    'did_tstat': model5.tvalues['treat_post'],
    'did_pvalue': model5.pvalues['treat_post'],
    'did_ci_lower': model5.conf_int().loc['treat_post', 0],
    'did_ci_upper': model5.conf_int().loc['treat_post', 1],
    'r_squared': model5.rsquared
}

# Save to file
results_df = pd.DataFrame([results_summary])
results_df.to_csv(f'{OUTPUT_DIR}/results_summary.csv', index=False)

# Save yearly effects for plotting
yearly_df = pd.DataFrame(yearly_effects)
yearly_df.to_csv(f'{OUTPUT_DIR}/yearly_effects.csv', index=False)

# Save event study results
event_df = pd.DataFrame(event_study_results)
event_df.to_csv(f'{OUTPUT_DIR}/event_study.csv', index=False)

# Save model coefficients
coef_df = pd.DataFrame({
    'variable': model5.params.index[:20],  # First 20 to avoid too many state dummies
    'coefficient': model5.params.values[:20],
    'std_error': model5.bse.values[:20],
    'pvalue': model5.pvalues.values[:20]
})
coef_df.to_csv(f'{OUTPUT_DIR}/model_coefficients.csv', index=False)

print(f"\nResults saved to {OUTPUT_DIR}/")
print(f"  - results_summary.csv")
print(f"  - yearly_effects.csv")
print(f"  - event_study.csv")
print(f"  - model_coefficients.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"""
Research Question: Effect of DACA eligibility on full-time employment

Sample: Hispanic-Mexican Mexican-born non-citizens who arrived before age 16
        and by 2007, ages 26-35 at DACA implementation

Design: Difference-in-Differences
  - Treatment: Ages 26-30 at June 15, 2012 (DACA eligible)
  - Control: Ages 31-35 at June 15, 2012 (too old for DACA)
  - Pre-period: 2006-2011
  - Post-period: 2013-2016

Preferred Estimate (Model 5: State + Year FE, covariates, clustered SEs):
  DiD Coefficient:     {results_summary['did_coefficient']:.4f}
  Standard Error:      {results_summary['did_se']:.4f}
  95% Confidence Int:  [{results_summary['did_ci_lower']:.4f}, {results_summary['did_ci_upper']:.4f}]
  p-value:             {results_summary['did_pvalue']:.4f}
  Sample Size:         {results_summary['sample_size']:,}
  R-squared:           {results_summary['r_squared']:.4f}

Interpretation:
DACA eligibility {'increased' if results_summary['did_coefficient'] > 0 else 'decreased'}
the probability of full-time employment by {abs(results_summary['did_coefficient']):.2%}
({abs(results_summary['did_coefficient'])*100:.2f} percentage points).
This effect is {'statistically significant' if results_summary['did_pvalue'] < 0.05 else 'not statistically significant'}
at the 5% level (p = {results_summary['did_pvalue']:.4f}).
""")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

"""
DACA Replication Study - Analysis Script
Effect of DACA on Full-Time Employment among Hispanic-Mexican, Mexican-Born Individuals

Research Design: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Outcome: Full-time employment (35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = "data/data.csv"
OUTPUT_PREFIX = "replication_39"

# DACA implementation date: June 15, 2012
DACA_YEAR = 2012
DACA_MONTH = 6
DACA_DAY = 15

# Age groups at DACA implementation
TREAT_AGE_MIN = 26
TREAT_AGE_MAX = 30
CONTROL_AGE_MIN = 31
CONTROL_AGE_MAX = 35

# DACA eligibility: must have arrived before 31st birthday as of June 15, 2012
# and must have arrived before 16th birthday
# and must have lived in US since June 15, 2007

# =============================================================================
# DATA LOADING
# =============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 70)
print("\n1. Loading data...")

# Load data in chunks due to large file size
chunks = []
chunk_size = 500000
for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False):
    chunks.append(chunk)
    print(f"   Loaded chunk with {len(chunk):,} rows")

df = pd.concat(chunks, ignore_index=True)
print(f"\n   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# SAMPLE RESTRICTION
# =============================================================================
print("\n2. Applying sample restrictions...")

# Initial count
n_initial = len(df)
print(f"   Initial observations: {n_initial:,}")

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN = 1)
df = df[df['HISPAN'] == 1]
n_after_hispan = len(df)
print(f"   After Hispanic-Mexican restriction: {n_after_hispan:,} (dropped {n_initial - n_after_hispan:,})")

# Restriction 2: Born in Mexico (BPL = 200)
df = df[df['BPL'] == 200]
n_after_bpl = len(df)
print(f"   After Mexico birthplace restriction: {n_after_bpl:,} (dropped {n_after_hispan - n_after_bpl:,})")

# Restriction 3: Not a citizen (CITIZEN = 3)
# This proxies for undocumented status
df = df[df['CITIZEN'] == 3]
n_after_citizen = len(df)
print(f"   After non-citizen restriction: {n_after_citizen:,} (dropped {n_after_bpl - n_after_citizen:,})")

# Restriction 4: Exclude year 2012 (cannot distinguish pre/post DACA)
df = df[df['YEAR'] != 2012]
n_after_2012 = len(df)
print(f"   After excluding 2012: {n_after_2012:,} (dropped {n_after_citizen - n_after_2012:,})")

# =============================================================================
# CONSTRUCT KEY VARIABLES
# =============================================================================
print("\n3. Constructing key variables...")

# Calculate age as of June 15, 2012
# We need to determine birth cohorts for treatment/control assignment
# Birth year determines the cohort

# For someone to be age X on June 15, 2012:
# - If born in Q1 (Jan-Mar) or Q2 (Apr-Jun): they turn X in 2012, so BIRTHYR = 2012 - X
# - If born in Q3 (Jul-Sep) or Q4 (Oct-Dec): they haven't turned X yet in June, so BIRTHYR = 2012 - X

# Simpler approach: use birth year to define cohorts
# Ages 26-30 on June 15, 2012 means born roughly 1982-1986
# Ages 31-35 on June 15, 2012 means born roughly 1977-1981

# More precisely:
# Age 26 on June 15, 2012: Born between June 16, 1985 and June 15, 1986
# Age 30 on June 15, 2012: Born between June 16, 1981 and June 15, 1982
# Age 31 on June 15, 2012: Born between June 16, 1980 and June 15, 1981
# Age 35 on June 15, 2012: Born between June 16, 1976 and June 15, 1977

# For simplicity and following standard practice, we use birth year:
# Treatment (26-30): BIRTHYR 1982-1986
# Control (31-35): BIRTHYR 1977-1981

df['treat_group'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control_group'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Calculate approximate age at DACA implementation
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Keep only treatment and control groups
df = df[(df['treat_group'] == 1) | (df['control_group'] == 1)]
n_after_age = len(df)
print(f"   After age group restriction (26-35 in 2012): {n_after_age:,}")

# Check DACA eligibility criteria
# Must have arrived before 16th birthday
# Age at arrival = YRIMMIG - BIRTHYR
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['YRIMMIG'] > 0)

# Must have been in US since June 15, 2007 (arrived by 2007)
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# Apply DACA eligibility restrictions
df = df[df['arrived_before_16'] == True]
n_after_arrival = len(df)
print(f"   After arrived-before-16 restriction: {n_after_arrival:,}")

df = df[df['in_us_since_2007'] == True]
n_after_2007 = len(df)
print(f"   After in-US-since-2007 restriction: {n_after_2007:,}")

# =============================================================================
# OUTCOME VARIABLE
# =============================================================================
print("\n4. Creating outcome variable...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK = 0 means N/A (not employed or other)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employment indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"   Full-time employment rate: {df['fulltime'].mean():.3f}")
print(f"   Employment rate: {df['employed'].mean():.3f}")

# =============================================================================
# TREATMENT INDICATORS
# =============================================================================
print("\n5. Creating treatment indicators...")

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Treatment group indicator (ages 26-30 in 2012)
df['treated'] = df['treat_group']

# Interaction term (DiD estimator)
df['treated_post'] = df['treated'] * df['post']

print(f"   Pre-period years: {sorted(df[df['post']==0]['YEAR'].unique())}")
print(f"   Post-period years: {sorted(df[df['post']==1]['YEAR'].unique())}")
print(f"   Treatment group (ages 26-30): {df['treated'].sum():,} observations")
print(f"   Control group (ages 31-35): {(1-df['treated']).sum():,} observations")

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

# Sample sizes by group and period
print("\n6. Sample sizes by group and period:")
sample_table = df.groupby(['treated', 'post']).size().unstack()
sample_table.index = ['Control (31-35)', 'Treatment (26-30)']
sample_table.columns = ['Pre-DACA', 'Post-DACA']
print(sample_table)

# Mean outcome by group and period
print("\n7. Full-time employment rates by group and period:")
outcome_table = df.groupby(['treated', 'post'])['fulltime'].mean().unstack()
outcome_table.index = ['Control (31-35)', 'Treatment (26-30)']
outcome_table.columns = ['Pre-DACA', 'Post-DACA']
print(outcome_table.round(4))

# Calculate simple DiD estimate
pre_treat = df[(df['treated']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['treated']==1) & (df['post']==1)]['fulltime'].mean()
pre_control = df[(df['treated']==0) & (df['post']==0)]['fulltime'].mean()
post_control = df[(df['treated']==0) & (df['post']==1)]['fulltime'].mean()

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   Simple DiD estimate: {did_simple:.4f}")
print(f"   Treatment change: {post_treat - pre_treat:.4f}")
print(f"   Control change: {post_control - pre_control:.4f}")

# =============================================================================
# COVARIATE SUMMARY
# =============================================================================
print("\n8. Summary statistics by treatment group:")

covariates = ['AGE', 'SEX', 'MARST', 'EDUC', 'NCHILD', 'FAMSIZE']
for var in covariates:
    if var in df.columns:
        treat_mean = df[df['treated']==1][var].mean()
        control_mean = df[df['treated']==0][var].mean()
        print(f"   {var}: Treatment={treat_mean:.2f}, Control={control_mean:.2f}")

# Create female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Create married indicator
df['married'] = (df['MARST'] == 1).astype(int)

# Create education categories
df['educ_less_hs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = ((df['EDUC'] >= 6) & (df['EDUC'] <= 7)).astype(int)
df['educ_some_college'] = ((df['EDUC'] >= 8) & (df['EDUC'] <= 9)).astype(int)
df['educ_college'] = (df['EDUC'] >= 10).astype(int)

# =============================================================================
# REGRESSION ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("REGRESSION ANALYSIS")
print("=" * 70)

# Model 1: Basic DiD
print("\n9. Model 1: Basic Difference-in-Differences")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with year fixed effects
print("\n10. Model 2: DiD with Year Fixed Effects")
df['year_factor'] = df['YEAR'].astype(str)
model2 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df).fit(cov_type='HC1')
print("\n   Key coefficients:")
print(f"   treated_post: {model2.params['treated_post']:.4f} (SE: {model2.bse['treated_post']:.4f})")
print(f"   t-stat: {model2.tvalues['treated_post']:.3f}, p-value: {model2.pvalues['treated_post']:.4f}")

# Model 3: DiD with demographic controls
print("\n11. Model 3: DiD with Year FE and Demographic Controls")
model3 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post + female + married + NCHILD + educ_hs + educ_some_college + educ_college', data=df).fit(cov_type='HC1')
print("\n   Key coefficients:")
print(f"   treated_post: {model3.params['treated_post']:.4f} (SE: {model3.bse['treated_post']:.4f})")
print(f"   t-stat: {model3.tvalues['treated_post']:.3f}, p-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: DiD with state fixed effects
print("\n12. Model 4: DiD with Year and State Fixed Effects")
df['state_factor'] = df['STATEFIP'].astype(str)
model4 = smf.ols('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post', data=df).fit(cov_type='HC1')
print("\n   Key coefficients:")
print(f"   treated_post: {model4.params['treated_post']:.4f} (SE: {model4.bse['treated_post']:.4f})")
print(f"   t-stat: {model4.tvalues['treated_post']:.3f}, p-value: {model4.pvalues['treated_post']:.4f}")

# Model 5: Full specification
print("\n13. Model 5: Full Specification (Year FE + State FE + Demographics)")
model5 = smf.ols('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + female + married + NCHILD + educ_hs + educ_some_college + educ_college', data=df).fit(cov_type='HC1')
print("\n   Key coefficients:")
print(f"   treated_post: {model5.params['treated_post']:.4f} (SE: {model5.bse['treated_post']:.4f})")
print(f"   t-stat: {model5.tvalues['treated_post']:.3f}, p-value: {model5.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# =============================================================================
# WEIGHTED ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("WEIGHTED ANALYSIS")
print("=" * 70)

print("\n14. Model 6: Weighted DiD (PERWT)")
# Using WLS with person weights
import statsmodels.api as sm

# Prepare data for weighted regression
X = df[['treated', 'post', 'treated_post']].copy()
X = sm.add_constant(X)
y = df['fulltime']
weights = df['PERWT']

model6 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print(f"   treated_post: {model6.params['treated_post']:.4f} (SE: {model6.bse['treated_post']:.4f})")
print(f"   t-stat: {model6.tvalues['treated_post']:.3f}, p-value: {model6.pvalues['treated_post']:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 70)
print("ROBUSTNESS CHECKS")
print("=" * 70)

# Robustness 1: Alternative outcome - any employment
print("\n15. Robustness Check 1: Employment (any hours)")
model_emp = smf.ols('employed ~ treated + C(year_factor) + treated_post', data=df).fit(cov_type='HC1')
print(f"   treated_post: {model_emp.params['treated_post']:.4f} (SE: {model_emp.bse['treated_post']:.4f})")
print(f"   p-value: {model_emp.pvalues['treated_post']:.4f}")

# Robustness 2: Different age bandwidths
print("\n16. Robustness Check 2: Narrower age bandwidth (27-29 vs 32-34)")
df['treat_narrow'] = ((df['BIRTHYR'] >= 1983) & (df['BIRTHYR'] <= 1985)).astype(int)
df['control_narrow'] = ((df['BIRTHYR'] >= 1978) & (df['BIRTHYR'] <= 1980)).astype(int)
df_narrow = df[(df['treat_narrow'] == 1) | (df['control_narrow'] == 1)].copy()
df_narrow['treated_narrow'] = df_narrow['treat_narrow']
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.ols('fulltime ~ treated_narrow + C(year_factor) + treated_post_narrow', data=df_narrow).fit(cov_type='HC1')
print(f"   treated_post: {model_narrow.params['treated_post_narrow']:.4f} (SE: {model_narrow.bse['treated_post_narrow']:.4f})")
print(f"   p-value: {model_narrow.pvalues['treated_post_narrow']:.4f}")
print(f"   N: {len(df_narrow):,}")

# Robustness 3: Separate by gender
print("\n17. Robustness Check 3: By Gender")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_male).fit(cov_type='HC1')
model_female = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_female).fit(cov_type='HC1')

print(f"   Males: treated_post = {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f}), N = {len(df_male):,}")
print(f"   Females: treated_post = {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f}), N = {len(df_female):,}")

# Robustness 4: Pre-trend test (placebo test using pre-period only)
print("\n18. Robustness Check 4: Pre-trend Test")
df_pre = df[df['post'] == 0].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat_post'] = df_pre['treated'] * df_pre['placebo_post']

model_pretrend = smf.ols('fulltime ~ treated + placebo_post + placebo_treat_post', data=df_pre).fit(cov_type='HC1')
print(f"   Placebo DiD (2009-2011 vs 2006-2008): {model_pretrend.params['placebo_treat_post']:.4f}")
print(f"   SE: {model_pretrend.bse['placebo_treat_post']:.4f}, p-value: {model_pretrend.pvalues['placebo_treat_post']:.4f}")

# =============================================================================
# EVENT STUDY
# =============================================================================
print("\n" + "=" * 70)
print("EVENT STUDY ANALYSIS")
print("=" * 70)

print("\n19. Year-by-year treatment effects (relative to 2011):")

# Create year dummies interacted with treatment
years = sorted(df['YEAR'].unique())
base_year = 2011

event_study_results = []
for year in years:
    if year != base_year:
        df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'treated_year_{year}'] = df['treated'] * df[f'year_{year}']

# Run event study regression
formula = 'fulltime ~ treated + ' + ' + '.join([f'year_{y}' for y in years if y != base_year])
formula += ' + ' + ' + '.join([f'treated_year_{y}' for y in years if y != base_year])

model_event = smf.ols(formula, data=df).fit(cov_type='HC1')

print(f"\n   {'Year':<8} {'Coefficient':<12} {'SE':<10} {'95% CI':<25}")
print("   " + "-" * 55)
for year in sorted(years):
    if year != base_year:
        coef_name = f'treated_year_{year}'
        coef = model_event.params[coef_name]
        se = model_event.bse[coef_name]
        ci_low, ci_high = model_event.conf_int().loc[coef_name]
        print(f"   {year:<8} {coef:<12.4f} {se:<10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# FINAL RESULTS SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print("\n20. Preferred Specification Results (Model 5: Full specification)")
print(f"\n   DACA Effect on Full-Time Employment:")
print(f"   Point Estimate: {model5.params['treated_post']:.4f}")
print(f"   Standard Error: {model5.bse['treated_post']:.4f}")
print(f"   95% Confidence Interval: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   t-statistic: {model5.tvalues['treated_post']:.3f}")
print(f"   p-value: {model5.pvalues['treated_post']:.4f}")
print(f"\n   Sample Size: {int(model5.nobs):,}")
print(f"   R-squared: {model5.rsquared:.4f}")

# Save key results
results_dict = {
    'preferred_estimate': model5.params['treated_post'],
    'se': model5.bse['treated_post'],
    'ci_low': model5.conf_int().loc['treated_post', 0],
    'ci_high': model5.conf_int().loc['treated_post', 1],
    'pvalue': model5.pvalues['treated_post'],
    'n': int(model5.nobs),
    'r2': model5.rsquared
}

# =============================================================================
# SAVE RESULTS FOR LATEX
# =============================================================================
print("\n21. Saving results for LaTeX report...")

# Create results dataframe
results_df = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) Year FE', '(3) Year FE + Demographics',
              '(4) Year FE + State FE', '(5) Full Specification', '(6) Weighted'],
    'Estimate': [model1.params['treated_post'], model2.params['treated_post'],
                 model3.params['treated_post'], model4.params['treated_post'],
                 model5.params['treated_post'], model6.params['treated_post']],
    'SE': [model1.bse['treated_post'], model2.bse['treated_post'],
           model3.bse['treated_post'], model4.bse['treated_post'],
           model5.bse['treated_post'], model6.bse['treated_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared,
           model4.rsquared, model5.rsquared, model6.rsquared]
})

results_df.to_csv('regression_results.csv', index=False)
print("   Saved regression_results.csv")

# Save event study results
event_results = []
for year in sorted(years):
    if year != base_year:
        coef_name = f'treated_year_{year}'
        event_results.append({
            'Year': year,
            'Coefficient': model_event.params[coef_name],
            'SE': model_event.bse[coef_name],
            'CI_low': model_event.conf_int().loc[coef_name, 0],
            'CI_high': model_event.conf_int().loc[coef_name, 1]
        })

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("   Saved event_study_results.csv")

# Save descriptive statistics
desc_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'female': ['mean'],
    'married': ['mean'],
    'AGE': ['mean'],
    'NCHILD': ['mean']
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("   Saved descriptive_stats.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

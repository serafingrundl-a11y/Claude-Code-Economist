"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY: ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n1. Loading data...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years available: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. DEFINE SAMPLE RESTRICTIONS
# ============================================================================
print("\n2. Applying sample restrictions...")

# Save initial count
initial_n = len(df)

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN == 1)
# HISPAN = 1 is "Mexican"
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After restricting to Hispanic-Mexican (HISPAN=1): {len(df_sample):,}")

# Restriction 2: Born in Mexico (BPL = 200)
# BPL = 200 is Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"   After restricting to Mexican-born (BPL=200): {len(df_sample):,}")

# Restriction 3: Not a US citizen (CITIZEN = 3)
# CITIZEN = 3 is "Not a citizen"
# Per instructions: "Assume that anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes"
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After restricting to non-citizens (CITIZEN=3): {len(df_sample):,}")

# Restriction 4: Working-age population (16-64) for employment analysis
# We need ages where people can reasonably be employed
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"   After restricting to ages 16-64: {len(df_sample):,}")

# ============================================================================
# 3. DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n3. Defining DACA eligibility...")

# DACA Eligibility Requirements (as of June 15, 2012):
# 1. Arrived unlawfully in the US before their 16th birthday
# 2. Had not yet had their 31st birthday as of June 15, 2012
# 3. Lived continuously in the US since June 15, 2007
# 4. Were present in the US on June 15, 2012 and did not have lawful status

# Calculate age as of June 15, 2012
# Birth year is available; use BIRTHQTR to refine if needed
# For simplicity, we'll use BIRTHYR to calculate approximate age on June 15, 2012
df_sample['age_june2012'] = 2012 - df_sample['BIRTHYR']

# Condition 1: Arrived before 16th birthday
# We need to compute age at arrival using YRIMMIG (year of immigration)
# Age at arrival = YRIMMIG - BIRTHYR
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Condition 2: Under 31 as of June 15, 2012 (born on or after June 15, 1981)
# Approximately: born in 1981 or later, accounting for birth quarter
# More precisely: BIRTHYR >= 1982, or (BIRTHYR == 1981 and BIRTHQTR >= 2)
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
df_sample['under_31_june2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 2))
)

# Condition 3: Lived continuously in US since June 15, 2007
# YRIMMIG should be 2007 or earlier
df_sample['in_us_since_2007'] = df_sample['YRIMMIG'] <= 2007

# Condition 4: Arrived before 16th birthday
df_sample['arrived_before_16'] = df_sample['age_at_arrival'] < 16

# Define DACA eligible: must meet all conditions
# Note: We cannot directly observe continuous residence or presence on June 15, 2012
# but use immigration year as proxy for continuous residence
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['under_31_june2012'] &
    df_sample['in_us_since_2007']
)

# Convert to integer
df_sample['daca_eligible'] = df_sample['daca_eligible'].astype(int)

print(f"   DACA eligible: {df_sample['daca_eligible'].sum():,} ({100*df_sample['daca_eligible'].mean():.1f}%)")
print(f"   Not DACA eligible: {(1-df_sample['daca_eligible']).sum():,}")

# ============================================================================
# 4. DEFINE TREATMENT PERIOD
# ============================================================================
print("\n4. Defining treatment period...")

# DACA implemented June 15, 2012
# Pre-period: 2006-2011 (or 2008-2011 to have cleaner comparison)
# Post-period: 2013-2016 (outcome years per instructions)
# Note: 2012 is ambiguous because DACA was implemented mid-year

# Create treatment period variable
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Drop 2012 observations due to ambiguity
df_sample_analysis = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   Dropping 2012 (mid-implementation year)")
print(f"   Sample size after dropping 2012: {len(df_sample_analysis):,}")

# ============================================================================
# 5. DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n5. Defining outcome variable...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A or not working)
df_sample_analysis['fulltime'] = (df_sample_analysis['UHRSWORK'] >= 35).astype(int)

print(f"   Full-time employed (35+ hrs/week): {df_sample_analysis['fulltime'].sum():,} ({100*df_sample_analysis['fulltime'].mean():.1f}%)")

# ============================================================================
# 6. CREATE CONTROL VARIABLES
# ============================================================================
print("\n6. Creating control variables...")

# Age and age squared
df_sample_analysis['age_sq'] = df_sample_analysis['AGE'] ** 2

# Gender (female = 1)
df_sample_analysis['female'] = (df_sample_analysis['SEX'] == 2).astype(int)

# Marital status (married = 1)
df_sample_analysis['married'] = (df_sample_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample_analysis['educ_less_hs'] = (df_sample_analysis['EDUC'] < 6).astype(int)
df_sample_analysis['educ_hs'] = (df_sample_analysis['EDUC'] == 6).astype(int)
df_sample_analysis['educ_some_college'] = (df_sample_analysis['EDUC'].isin([7, 8, 9])).astype(int)
df_sample_analysis['educ_college_plus'] = (df_sample_analysis['EDUC'] >= 10).astype(int)

# Years in US (centered)
df_sample_analysis['yrsusa'] = df_sample_analysis['YRSUSA1']
df_sample_analysis['yrsusa_sq'] = df_sample_analysis['yrsusa'] ** 2

# State fixed effects will be included via STATEFIP

print(f"   Female: {100*df_sample_analysis['female'].mean():.1f}%")
print(f"   Married: {100*df_sample_analysis['married'].mean():.1f}%")
print(f"   Mean age: {df_sample_analysis['AGE'].mean():.1f}")
print(f"   Mean years in US: {df_sample_analysis['yrsusa'].mean():.1f}")

# ============================================================================
# 7. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("7. DESCRIPTIVE STATISTICS")
print("="*80)

# Summary by eligibility and period
print("\n   A. Sample size by eligibility and period:")
crosstab = pd.crosstab(df_sample_analysis['daca_eligible'], df_sample_analysis['post'],
                        margins=True, margins_name='Total')
crosstab.index = ['Non-eligible', 'DACA-eligible', 'Total']
crosstab.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)', 'Total']
print(crosstab)

print("\n   B. Full-time employment rates by group and period:")
employment_table = df_sample_analysis.groupby(['daca_eligible', 'post'])['fulltime'].agg(['mean', 'std', 'count'])
employment_table = employment_table.round(4)
print(employment_table)

# Calculate simple difference-in-differences
pre_treat = df_sample_analysis[(df_sample_analysis['daca_eligible']==1) & (df_sample_analysis['post']==0)]['fulltime'].mean()
post_treat = df_sample_analysis[(df_sample_analysis['daca_eligible']==1) & (df_sample_analysis['post']==1)]['fulltime'].mean()
pre_control = df_sample_analysis[(df_sample_analysis['daca_eligible']==0) & (df_sample_analysis['post']==0)]['fulltime'].mean()
post_control = df_sample_analysis[(df_sample_analysis['daca_eligible']==0) & (df_sample_analysis['post']==1)]['fulltime'].mean()

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   C. Simple Difference-in-Differences:")
print(f"      DACA-eligible Pre:  {pre_treat:.4f}")
print(f"      DACA-eligible Post: {post_treat:.4f}")
print(f"      Non-eligible Pre:   {pre_control:.4f}")
print(f"      Non-eligible Post:  {post_control:.4f}")
print(f"      DiD Estimate:       {simple_did:.4f}")

# ============================================================================
# 8. DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df_sample_analysis['did'] = df_sample_analysis['daca_eligible'] * df_sample_analysis['post']

# Model 1: Basic DiD (no controls)
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + did', data=df_sample_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})
print(f"   DiD coefficient: {model1.params['did']:.4f} (SE: {model1.bse['did']:.4f})")
print(f"   95% CI: [{model1.conf_int().loc['did', 0]:.4f}, {model1.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model1.pvalues['did']:.4f}")
print(f"   N = {int(model1.nobs):,}")

# Model 2: DiD with demographic controls
print("\n   Model 2: DiD with demographic controls")
model2 = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married',
                 data=df_sample_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})
print(f"   DiD coefficient: {model2.params['did']:.4f} (SE: {model2.bse['did']:.4f})")
print(f"   95% CI: [{model2.conf_int().loc['did', 0]:.4f}, {model2.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model2.pvalues['did']:.4f}")
print(f"   N = {int(model2.nobs):,}")

# Model 3: DiD with demographic and education controls
print("\n   Model 3: DiD with demographic + education controls")
model3 = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus',
                 data=df_sample_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})
print(f"   DiD coefficient: {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})")
print(f"   95% CI: [{model3.conf_int().loc['did', 0]:.4f}, {model3.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['did']:.4f}")
print(f"   N = {int(model3.nobs):,}")

# Model 4: DiD with all controls including year and state fixed effects
print("\n   Model 4: Full model with year and state fixed effects")
df_sample_analysis['YEAR_cat'] = pd.Categorical(df_sample_analysis['YEAR'])
df_sample_analysis['STATE_cat'] = pd.Categorical(df_sample_analysis['STATEFIP'])

model4 = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_sample_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})
print(f"   DiD coefficient: {model4.params['did']:.4f} (SE: {model4.bse['did']:.4f})")
print(f"   95% CI: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['did']:.4f}")
print(f"   N = {int(model4.nobs):,}")
print(f"   R-squared: {model4.rsquared:.4f}")

# ============================================================================
# 9. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("9. ROBUSTNESS CHECKS")
print("="*80)

# 9A: Restricting to prime working age (25-54)
print("\n   A. Restricting to prime working age (25-54):")
df_prime = df_sample_analysis[(df_sample_analysis['AGE'] >= 25) & (df_sample_analysis['AGE'] <= 54)]
model_prime = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_prime).fit(cov_type='cluster', cov_kwds={'groups': df_prime['STATEFIP']})
print(f"   DiD coefficient: {model_prime.params['did']:.4f} (SE: {model_prime.bse['did']:.4f})")
print(f"   p-value: {model_prime.pvalues['did']:.4f}")
print(f"   N = {int(model_prime.nobs):,}")

# 9B: Separate analysis by gender
print("\n   B. Analysis by gender:")
df_male = df_sample_analysis[df_sample_analysis['female'] == 0]
df_female = df_sample_analysis[df_sample_analysis['female'] == 1]

model_male = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
model_female = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})

print(f"   Males:   DiD = {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f}), N = {int(model_male.nobs):,}")
print(f"   Females: DiD = {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f}), N = {int(model_female.nobs):,}")

# 9C: Alternative control group - arrived after 16
print("\n   C. Using only arrived-after-16 as control (alternative control group):")
df_alt = df_sample_analysis[(df_sample_analysis['daca_eligible'] == 1) |
                             ((df_sample_analysis['arrived_before_16'] == False) &
                              (df_sample_analysis['in_us_since_2007'] == True))].copy()
df_alt['did'] = df_alt['daca_eligible'] * df_alt['post']
model_alt = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)',
                 data=df_alt).fit(cov_type='cluster', cov_kwds={'groups': df_alt['STATEFIP']})
print(f"   DiD coefficient: {model_alt.params['did']:.4f} (SE: {model_alt.bse['did']:.4f})")
print(f"   p-value: {model_alt.pvalues['did']:.4f}")
print(f"   N = {int(model_alt.nobs):,}")

# 9D: Event study / Dynamic effects
print("\n   D. Event study (year-by-year effects):")
df_sample_analysis['year_x_elig'] = df_sample_analysis.apply(
    lambda row: f"y{row['YEAR']}_elig" if row['daca_eligible'] == 1 else "control", axis=1)

# Create year dummies interacted with eligibility
for year in sorted(df_sample_analysis['YEAR'].unique()):
    df_sample_analysis[f'elig_x_{year}'] = ((df_sample_analysis['YEAR'] == year) &
                                            (df_sample_analysis['daca_eligible'] == 1)).astype(int)

# Use 2011 as reference year
year_vars = [f'elig_x_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
formula_event = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + ' + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(YEAR) + C(STATEFIP)'

model_event = smf.ols(formula_event, data=df_sample_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})

print("   Year-specific effects (relative to 2011):")
for var in year_vars:
    year = var.split('_')[-1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"      {year}: {coef:7.4f} ({se:.4f}){sig}")

# ============================================================================
# 10. WEIGHTED ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("10. WEIGHTED ANALYSIS (using PERWT)")
print("="*80)

import statsmodels.api as sm

# Prepare data for weighted regression
X = df_sample_analysis[['daca_eligible', 'did', 'AGE', 'age_sq', 'female', 'married',
                         'educ_hs', 'educ_some_college', 'educ_college_plus']].copy()
# Add year dummies (excluding first)
years = sorted(df_sample_analysis['YEAR'].unique())
for y in years[1:]:
    X[f'year_{y}'] = (df_sample_analysis['YEAR'] == y).astype(int)

# Add state dummies (excluding first)
states = sorted(df_sample_analysis['STATEFIP'].unique())
for s in states[1:]:
    X[f'state_{s}'] = (df_sample_analysis['STATEFIP'] == s).astype(int)

X = sm.add_constant(X)
y = df_sample_analysis['fulltime']
weights = df_sample_analysis['PERWT']

model_weighted = sm.WLS(y, X, weights=weights).fit(cov_type='cluster', cov_kwds={'groups': df_sample_analysis['STATEFIP']})
print(f"   DiD coefficient (weighted): {model_weighted.params['did']:.4f} (SE: {model_weighted.bse['did']:.4f})")
print(f"   95% CI: [{model_weighted.conf_int().loc['did', 0]:.4f}, {model_weighted.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model_weighted.pvalues['did']:.4f}")

# ============================================================================
# 11. SUMMARY OF RESULTS
# ============================================================================
print("\n" + "="*80)
print("11. SUMMARY OF RESULTS")
print("="*80)

print("\n   PREFERRED ESTIMATE (Model 4 - Full model with year and state FE):")
print(f"   -----------------------------------------------------------------------")
print(f"   Effect of DACA eligibility on full-time employment: {model4.params['did']:.4f}")
print(f"   Standard Error (clustered by state): {model4.bse['did']:.4f}")
print(f"   95% Confidence Interval: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['did']:.4f}")
print(f"   Sample Size: {int(model4.nobs):,}")
print(f"   -----------------------------------------------------------------------")

# ============================================================================
# 12. SAVE RESULTS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("12. SAVING RESULTS")
print("="*80)

# Save descriptive statistics
desc_stats = df_sample_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_college_plus': 'mean',
    'yrsusa': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')

# Save regression results summary
results_summary = pd.DataFrame({
    'Model': ['Model 1 (Basic)', 'Model 2 (Demographics)', 'Model 3 (+ Education)',
              'Model 4 (Full)', 'Weighted', 'Prime Age (25-54)', 'Males Only', 'Females Only'],
    'DiD_Coefficient': [model1.params['did'], model2.params['did'], model3.params['did'],
                        model4.params['did'], model_weighted.params['did'], model_prime.params['did'],
                        model_male.params['did'], model_female.params['did']],
    'Std_Error': [model1.bse['did'], model2.bse['did'], model3.bse['did'],
                  model4.bse['did'], model_weighted.bse['did'], model_prime.bse['did'],
                  model_male.bse['did'], model_female.bse['did']],
    'p_value': [model1.pvalues['did'], model2.pvalues['did'], model3.pvalues['did'],
                model4.pvalues['did'], model_weighted.pvalues['did'], model_prime.pvalues['did'],
                model_male.pvalues['did'], model_female.pvalues['did']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model_weighted.nobs), int(model_prime.nobs),
          int(model_male.nobs), int(model_female.nobs)]
})
results_summary.to_csv('regression_results.csv', index=False)

# Save event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params[f'elig_x_{y}'] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]],
    'Std_Error': [model_event.bse[f'elig_x_{y}'] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]],
    'p_value': [model_event.pvalues[f'elig_x_{y}'] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
})
event_results.to_csv('event_study_results.csv', index=False)

# Save sample characteristics
print("\n   Saved: descriptive_stats.csv")
print("   Saved: regression_results.csv")
print("   Saved: event_study_results.csv")

# ============================================================================
# 13. ADDITIONAL SUMMARY STATISTICS FOR REPORT
# ============================================================================
print("\n" + "="*80)
print("13. ADDITIONAL STATISTICS FOR REPORT")
print("="*80)

print("\n   Sample composition:")
print(f"   Total observations in analysis: {len(df_sample_analysis):,}")
print(f"   Unique years: {sorted(df_sample_analysis['YEAR'].unique())}")
print(f"   Unique states: {df_sample_analysis['STATEFIP'].nunique()}")

print("\n   Demographics by eligibility status:")
for elig in [0, 1]:
    subset = df_sample_analysis[df_sample_analysis['daca_eligible'] == elig]
    status = "DACA-eligible" if elig == 1 else "Non-eligible"
    print(f"\n   {status}:")
    print(f"      N = {len(subset):,}")
    print(f"      Mean age: {subset['AGE'].mean():.1f}")
    print(f"      Female: {100*subset['female'].mean():.1f}%")
    print(f"      Married: {100*subset['married'].mean():.1f}%")
    print(f"      College+: {100*subset['educ_college_plus'].mean():.1f}%")
    print(f"      Mean years in US: {subset['yrsusa'].mean():.1f}")
    print(f"      Full-time employment rate: {100*subset['fulltime'].mean():.1f}%")

print("\n   Employment trends by year:")
yearly_emp = df_sample_analysis.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
yearly_emp.columns = ['Non-eligible', 'DACA-eligible']
print(yearly_emp.round(4))

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Store key results for LaTeX
key_results = {
    'did_estimate': model4.params['did'],
    'did_se': model4.bse['did'],
    'did_pval': model4.pvalues['did'],
    'did_ci_low': model4.conf_int().loc['did', 0],
    'did_ci_high': model4.conf_int().loc['did', 1],
    'sample_size': int(model4.nobs),
    'n_eligible': int(df_sample_analysis['daca_eligible'].sum()),
    'n_control': int((1-df_sample_analysis['daca_eligible']).sum()),
    'pre_treat_rate': pre_treat,
    'post_treat_rate': post_treat,
    'pre_control_rate': pre_control,
    'post_control_rate': post_control,
}

# Save for LaTeX
import json
with open('key_results.json', 'w') as f:
    json.dump(key_results, f, indent=2)

print("\n   Saved: key_results.json")

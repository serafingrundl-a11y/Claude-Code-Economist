"""
DACA Employment Effects Replication Study
=========================================
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Identification: Difference-in-differences
- Treatment: Ages 26-30 at DACA (June 15, 2012)
- Control: Ages 31-35 at DACA
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 70)
print("DACA Employment Effects Replication Study")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1] Loading ACS data...")

# Define columns needed for analysis
cols_needed = [
    'YEAR', 'PERWT', 'STATEFIP',
    'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD',
    'CITIZEN', 'YRIMMIG',
    'EDUC', 'EDUCD', 'MARST',
    'EMPSTAT', 'EMPSTATD', 'LABFORCE',
    'UHRSWORK'
]

# Load data
df = pd.read_csv('data/data.csv', usecols=cols_needed, low_memory=False)
print(f"Total records loaded: {len(df):,}")

# ============================================================================
# STEP 2: Sample Selection
# ============================================================================
print("\n[2] Constructing analysis sample...")

# Filter 1: Hispanic-Mexican ethnicity
# HISPAN == 1 is Mexican, or HISPAND 100-107 for detailed Mexican categories
df_mex = df[(df['HISPAN'] == 1) | ((df['HISPAND'] >= 100) & (df['HISPAND'] <= 107))].copy()
print(f"After Hispanic-Mexican filter: {len(df_mex):,}")

# Filter 2: Born in Mexico (BPL == 200)
df_mex = df_mex[df_mex['BPL'] == 200]
print(f"After Mexico birthplace filter: {len(df_mex):,}")

# Filter 3: Non-citizens (CITIZEN == 3)
# This is our proxy for undocumented status per instructions
df_mex = df_mex[df_mex['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_mex):,}")

# ============================================================================
# STEP 3: Calculate Age at DACA Date and Define Groups
# ============================================================================
print("\n[3] Defining treatment and control groups...")

# DACA date: June 15, 2012
# We need to calculate age as of June 15, 2012

# Age calculation:
# - If born in Q1 (Jan-Mar) or Q2 (Apr-Jun): age = 2012 - birthyear by June 15
# - If born in Q3 (Jul-Sep) or Q4 (Oct-Dec): age = 2012 - birthyear - 1 by June 15

# More precisely: by June 15, 2012, someone has had their birthday if:
# - Q1: Jan-Mar -> birthday passed
# - Q2: Apr-Jun -> approximately half had birthday (use conservative approach)
# - Q3, Q4: birthday not yet reached by June 15

# For simplicity, calculate approximate age:
# age_at_daca = 2012 - BIRTHYR, adjusted for quarter
# Q1, Q2: already had birthday
# Q3, Q4: not had birthday yet

df_mex['age_at_daca'] = 2012 - df_mex['BIRTHYR']
# Adjust for those born after June (Q3, Q4) - they hadn't had birthday yet
df_mex.loc[df_mex['BIRTHQTR'].isin([3, 4]), 'age_at_daca'] = df_mex['age_at_daca'] - 1

print(f"\nAge at DACA distribution:")
print(df_mex['age_at_daca'].value_counts().sort_index())

# Treatment group: Ages 26-30 at DACA (born roughly 1982-1986)
# Control group: Ages 31-35 at DACA (born roughly 1977-1981)
df_mex['treat'] = ((df_mex['age_at_daca'] >= 26) & (df_mex['age_at_daca'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_at_daca'] >= 31) & (df_mex['age_at_daca'] <= 35)).astype(int)

# Keep only treatment and control groups
df_sample = df_mex[(df_mex['treat'] == 1) | (df_mex['control'] == 1)].copy()
print(f"\nAfter selecting ages 26-35 at DACA: {len(df_sample):,}")

# ============================================================================
# STEP 4: Apply Additional DACA Eligibility Criteria
# ============================================================================
print("\n[4] Applying DACA eligibility criteria...")

# Criterion: Arrived before age 16
# Immigration year must be such that they were under 16 when they arrived
# age_at_arrival = YRIMMIG - BIRTHYR
# Require age_at_arrival < 16

df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_arrival'] < 16]
print(f"After arrival before age 16: {len(df_sample):,}")

# Criterion: Continuous presence since June 15, 2007
# YRIMMIG <= 2007
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After continuous presence (arrived by 2007): {len(df_sample):,}")

# ============================================================================
# STEP 5: Define Time Periods
# ============================================================================
print("\n[5] Defining time periods...")

# Exclude 2012 (cannot distinguish pre/post within year)
df_sample = df_sample[df_sample['YEAR'] != 2012]
print(f"After excluding 2012: {len(df_sample):,}")

# Pre-treatment: 2006-2011
# Post-treatment: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"\nSample by year:")
print(df_sample.groupby('YEAR').size())

# ============================================================================
# STEP 6: Create Outcome Variable
# ============================================================================
print("\n[6] Creating outcome variable...")

# Full-time employment: UHRSWORK >= 35
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"\nOutcome variable summary:")
print(f"Full-time employment rate: {df_sample['fulltime'].mean():.3f}")
print(f"Employment rate: {df_sample['employed'].mean():.3f}")

# ============================================================================
# STEP 7: Descriptive Statistics
# ============================================================================
print("\n[7] Descriptive Statistics...")

# Create descriptive table by treatment group and time period
def calc_stats(data, weight_col='PERWT'):
    """Calculate weighted statistics"""
    stats_dict = {}

    # Weighted mean for full-time
    weights = data[weight_col]
    stats_dict['fulltime'] = np.average(data['fulltime'], weights=weights)
    stats_dict['employed'] = np.average(data['employed'], weights=weights)
    stats_dict['age'] = np.average(data['AGE'], weights=weights)
    stats_dict['female'] = np.average(data['SEX'] == 2, weights=weights)
    stats_dict['married'] = np.average(data['MARST'].isin([1, 2]), weights=weights)
    stats_dict['N'] = len(data)
    stats_dict['N_weighted'] = weights.sum()

    return pd.Series(stats_dict)

# Stats by group
print("\n--- Treatment Group (Ages 26-30 at DACA) ---")
treat_pre = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 0)]
treat_post = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 1)]

print("\nPre-DACA (2006-2011):")
print(calc_stats(treat_pre))
print("\nPost-DACA (2013-2016):")
print(calc_stats(treat_post))

print("\n--- Control Group (Ages 31-35 at DACA) ---")
ctrl_pre = df_sample[(df_sample['control'] == 1) & (df_sample['post'] == 0)]
ctrl_post = df_sample[(df_sample['control'] == 1) & (df_sample['post'] == 1)]

print("\nPre-DACA (2006-2011):")
print(calc_stats(ctrl_pre))
print("\nPost-DACA (2013-2016):")
print(calc_stats(ctrl_post))

# ============================================================================
# STEP 8: Difference-in-Differences Analysis
# ============================================================================
print("\n" + "=" * 70)
print("[8] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Create interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_somecol'] = (df_sample['EDUC'] >= 7).astype(int)  # Some college+

model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + AGE',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
df_sample['year_cat'] = df_sample['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + female + married + educ_hs + AGE',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with year and state fixed effects ---")
model4 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + educ_hs',
                  data=df_sample,
                  weights=df_sample['PERWT']).fit(cov_type='HC1')
print("\nDiD coefficient (treat_post):")
print(f"Coefficient: {model4.params['treat_post']:.4f}")
print(f"Std Error: {model4.bse['treat_post']:.4f}")
print(f"t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# STEP 9: Simple DiD Calculation (Manual)
# ============================================================================
print("\n" + "=" * 70)
print("[9] MANUAL DID CALCULATION")
print("=" * 70)

# Calculate weighted means
def weighted_mean(data, var, weight='PERWT'):
    return np.average(data[var], weights=data[weight])

mean_treat_post = weighted_mean(treat_post, 'fulltime')
mean_treat_pre = weighted_mean(treat_pre, 'fulltime')
mean_ctrl_post = weighted_mean(ctrl_post, 'fulltime')
mean_ctrl_pre = weighted_mean(ctrl_pre, 'fulltime')

print(f"\nTreatment group (26-30):")
print(f"  Pre-DACA:  {mean_treat_pre:.4f}")
print(f"  Post-DACA: {mean_treat_post:.4f}")
print(f"  Change:    {mean_treat_post - mean_treat_pre:.4f}")

print(f"\nControl group (31-35):")
print(f"  Pre-DACA:  {mean_ctrl_pre:.4f}")
print(f"  Post-DACA: {mean_ctrl_post:.4f}")
print(f"  Change:    {mean_ctrl_post - mean_ctrl_pre:.4f}")

did_estimate = (mean_treat_post - mean_treat_pre) - (mean_ctrl_post - mean_ctrl_pre)
print(f"\nDiD Estimate: {did_estimate:.4f}")

# ============================================================================
# STEP 10: Robustness Checks
# ============================================================================
print("\n" + "=" * 70)
print("[10] ROBUSTNESS CHECKS")
print("=" * 70)

# Robustness 1: Alternative outcome - any employment
print("\n--- Robustness 1: Employment (any) as outcome ---")
model_emp = smf.wls('employed ~ treat + C(YEAR) + treat_post + female + married + educ_hs + AGE',
                     data=df_sample,
                     weights=df_sample['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# Robustness 2: By gender
print("\n--- Robustness 2: Effects by gender ---")
model_male = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + AGE',
                      data=df_sample[df_sample['female'] == 0],
                      weights=df_sample[df_sample['female'] == 0]['PERWT']).fit(cov_type='HC1')
print(f"Males - DiD coefficient: {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")

model_female = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + married + educ_hs + AGE',
                        data=df_sample[df_sample['female'] == 1],
                        weights=df_sample[df_sample['female'] == 1]['PERWT']).fit(cov_type='HC1')
print(f"Females - DiD coefficient: {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# Robustness 3: Placebo test with narrower age bands
print("\n--- Robustness 3: Narrower age bands (27-29 vs 32-34) ---")
df_narrow = df_mex[(df_mex['age_at_daca'] >= 27) & (df_mex['age_at_daca'] <= 34)].copy()
df_narrow['age_at_arrival'] = df_narrow['YRIMMIG'] - df_narrow['BIRTHYR']
df_narrow = df_narrow[df_narrow['age_at_arrival'] < 16]
df_narrow = df_narrow[df_narrow['YRIMMIG'] <= 2007]
df_narrow = df_narrow[df_narrow['YEAR'] != 2012]
df_narrow['treat_narrow'] = ((df_narrow['age_at_daca'] >= 27) & (df_narrow['age_at_daca'] <= 29)).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'].isin([1, 2])).astype(int)
df_narrow['educ_hs'] = (df_narrow['EDUC'] >= 6).astype(int)

model_narrow = smf.wls('fulltime ~ treat_narrow + C(YEAR) + treat_post_narrow + female + married + educ_hs + AGE',
                        data=df_narrow,
                        weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f})")

# ============================================================================
# STEP 11: Event Study
# ============================================================================
print("\n" + "=" * 70)
print("[11] EVENT STUDY ANALYSIS")
print("=" * 70)

# Create year dummies interacted with treatment
df_sample['year_2006'] = (df_sample['YEAR'] == 2006).astype(int)
df_sample['year_2007'] = (df_sample['YEAR'] == 2007).astype(int)
df_sample['year_2008'] = (df_sample['YEAR'] == 2008).astype(int)
df_sample['year_2009'] = (df_sample['YEAR'] == 2009).astype(int)
df_sample['year_2010'] = (df_sample['YEAR'] == 2010).astype(int)
df_sample['year_2011'] = (df_sample['YEAR'] == 2011).astype(int)  # Reference year
df_sample['year_2013'] = (df_sample['YEAR'] == 2013).astype(int)
df_sample['year_2014'] = (df_sample['YEAR'] == 2014).astype(int)
df_sample['year_2015'] = (df_sample['YEAR'] == 2015).astype(int)
df_sample['year_2016'] = (df_sample['YEAR'] == 2016).astype(int)

# Interactions (excluding 2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_sample[f'treat_x_{year}'] = df_sample['treat'] * df_sample[f'year_{year}']

event_formula = 'fulltime ~ treat + C(YEAR) + female + married + educ_hs + AGE + '
event_formula += ' + '.join([f'treat_x_{year}' for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])

model_event = smf.wls(event_formula,
                       data=df_sample,
                       weights=df_sample['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (reference year: 2011):")
print("-" * 50)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{year}']
    se = model_event.bse[f'treat_x_{year}']
    pval = model_event.pvalues[f'treat_x_{year}']
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"Year {year}: {coef:7.4f} ({se:.4f}) {sig}")

# ============================================================================
# STEP 12: Save Results
# ============================================================================
print("\n" + "=" * 70)
print("[12] SAVING RESULTS")
print("=" * 70)

# Create summary results
results = {
    'sample_size': len(df_sample),
    'treat_n': (df_sample['treat'] == 1).sum(),
    'control_n': (df_sample['control'] == 1).sum(),
    'did_estimate_simple': did_estimate,
    'did_estimate_model1': model1.params['treat_post'],
    'did_se_model1': model1.bse['treat_post'],
    'did_estimate_model4': model4.params['treat_post'],
    'did_se_model4': model4.bse['treat_post'],
    'did_pvalue_model4': model4.pvalues['treat_post']
}

print("\n=== FINAL RESULTS SUMMARY ===")
print(f"Total sample size: {results['sample_size']:,}")
print(f"Treatment group (26-30): {results['treat_n']:,}")
print(f"Control group (31-35): {results['control_n']:,}")
print(f"\nPreferred DiD Estimate (Model 4 with FE):")
print(f"  Coefficient: {results['did_estimate_model4']:.4f}")
print(f"  Std Error: {results['did_se_model4']:.4f}")
print(f"  p-value: {results['did_pvalue_model4']:.4f}")

# Save detailed results to CSV
results_df = pd.DataFrame({
    'Group': ['Treatment Pre', 'Treatment Post', 'Control Pre', 'Control Post'],
    'Mean_Fulltime': [mean_treat_pre, mean_treat_post, mean_ctrl_pre, mean_ctrl_post],
    'N': [len(treat_pre), len(treat_post), len(ctrl_pre), len(ctrl_post)]
})
results_df.to_csv('results_summary.csv', index=False)

# Save model coefficients
coef_df = pd.DataFrame({
    'Model': ['Model 1 (Basic)', 'Model 2 (Controls)', 'Model 3 (Year FE)', 'Model 4 (Year+State FE)'],
    'DiD_Coef': [model1.params['treat_post'], model2.params['treat_post'],
                 model3.params['treat_post'], model4.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'],
           model3.bse['treat_post'], model4.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post']]
})
coef_df.to_csv('model_coefficients.csv', index=False)

# Event study coefficients
event_df = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [model_event.params.get(f'treat_x_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'SE': [model_event.bse.get(f'treat_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [model_event.bse.get(f'treat_x_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
event_df.to_csv('event_study.csv', index=False)

print("\nResults saved to CSV files.")
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

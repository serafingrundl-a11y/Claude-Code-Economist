"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the US.

Author: Replication 22
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

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA WITH FILTERING (memory efficient)
# =============================================================================
print("\n1. Loading data with pre-filtering...")

# Only read columns we need
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
            'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter
chunk_size = 1000000
chunks = []
total_rows = 0
filtered_rows = 0

print("   Reading and filtering data in chunks...")
for chunk in pd.read_csv('data/data.csv', usecols=use_cols, chunksize=chunk_size):
    total_rows += len(chunk)
    # Filter as we go
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3) & # Non-citizen
        (chunk['AGE'] >= 16) & (chunk['AGE'] <= 64) &  # Working age
        (chunk['YEAR'] != 2012) &  # Exclude 2012
        (chunk['YRIMMIG'] > 0)    # Valid immigration year
    ].copy()
    filtered_rows += len(chunk_filtered)
    chunks.append(chunk_filtered)
    print(f"   Processed {total_rows:,} rows, kept {filtered_rows:,}...")

df_sample = pd.concat(chunks, ignore_index=True)
del chunks
print(f"\n   Total original rows: {total_rows:,}")
print(f"   Final sample size: {len(df_sample):,}")
print(f"   Years in sample: {sorted(df_sample['YEAR'].unique())}")

# =============================================================================
# 2. CREATE VARIABLES
# =============================================================================
print("\n2. Creating analysis variables...")

# 2a. Age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# 2b. DACA eligibility criteria
# Must have:
# - Arrived before age 16
# - Born 1981 or later (not yet 31 as of June 15, 2012)
# - Immigrated by 2007 (continuously present since June 15, 2007)

# Age at immigration < 16
arrived_before_16 = (df_sample['age_at_immig'] < 16)

# Birth year criteria: not yet 31 as of June 15, 2012
# Those born in Q1-Q2 of 1981 would have turned 31 by June 15, 2012
# Those born in Q3-Q4 of 1981 would not yet be 31 on June 15, 2012
under_31_in_2012 = ((df_sample['BIRTHYR'] >= 1982) |
                    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'].isin([3, 4]))))

# Immigrated by 2007 (present since June 15, 2007)
immig_by_2007 = (df_sample['YRIMMIG'] <= 2007)

# DACA eligible: all three criteria
df_sample['daca_eligible'] = (arrived_before_16 & under_31_in_2012 & immig_by_2007).astype(np.int8)

print(f"   DACA eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")

# 2c. Post-DACA indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(np.int8)
print(f"   Post-DACA observations: {df_sample['post'].sum():,}")

# 2d. Outcome: Full-time employment (35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(np.int8)
print(f"   Full-time employed: {df_sample['fulltime'].sum():,} ({df_sample['fulltime'].mean()*100:.1f}%)")

# 2e. Employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(np.int8)
print(f"   Employed: {df_sample['employed'].sum():,} ({df_sample['employed'].mean()*100:.1f}%)")

# 2f. DiD interaction term
df_sample['daca_post'] = (df_sample['daca_eligible'] * df_sample['post']).astype(np.int8)

# 2g. Control variables
df_sample['age_sq'] = (df_sample['AGE'] ** 2).astype(np.int16)
df_sample['female'] = (df_sample['SEX'] == 2).astype(np.int8)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(np.int8)

# Education categories
df_sample['less_than_hs'] = (df_sample['EDUC'] < 6).astype(np.int8)
df_sample['hs_grad'] = (df_sample['EDUC'] == 6).astype(np.int8)
df_sample['some_college'] = (df_sample['EDUC'].isin([7, 8, 9])).astype(np.int8)
df_sample['college_plus'] = (df_sample['EDUC'] >= 10).astype(np.int8)

print(f"\n   Final sample size: {len(df_sample):,}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n3. Descriptive Statistics...")

# Summary by treatment status and period
print("\n   --- Mean Full-time Employment by Group and Period ---")
ft_summary = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'PERWT': 'sum'
}).round(4)
print(ft_summary)

# Calculate raw DiD
pre_treat = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==0)]['fulltime'].mean()
post_treat = df_sample[(df_sample['daca_eligible']==1) & (df_sample['post']==1)]['fulltime'].mean()
pre_control = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==0)]['fulltime'].mean()
post_control = df_sample[(df_sample['daca_eligible']==0) & (df_sample['post']==1)]['fulltime'].mean()

raw_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n   Raw DiD estimate: {raw_did:.4f}")
print(f"   Treatment pre: {pre_treat:.4f}, post: {post_treat:.4f}, diff: {post_treat-pre_treat:.4f}")
print(f"   Control pre: {pre_control:.4f}, post: {post_control:.4f}, diff: {post_control-pre_control:.4f}")

# Weighted means
print("\n   --- Weighted Mean Full-time Employment by Group and Period ---")
for eligible in [0, 1]:
    for post_val in [0, 1]:
        mask = (df_sample['daca_eligible'] == eligible) & (df_sample['post'] == post_val)
        weighted_mean = np.average(df_sample.loc[mask, 'fulltime'],
                                    weights=df_sample.loc[mask, 'PERWT'])
        group_label = 'Eligible' if eligible else 'Ineligible'
        period_label = 'Post' if post_val else 'Pre'
        print(f"   {group_label}, {period_label}: {weighted_mean:.4f}")

# =============================================================================
# 4. REGRESSION ANALYSIS
# =============================================================================
print("\n4. Regression Analysis...")

# Create state dummies for faster computation
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True, dtype=np.int8)
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True, dtype=np.int8)

# Model 1: Basic DiD (no controls)
print("\n   --- Model 1: Basic DiD ---")
X1 = df_sample[['daca_eligible', 'post', 'daca_post']]
X1 = sm.add_constant(X1)
model1 = sm.OLS(df_sample['fulltime'], X1).fit(cov_type='HC1')
print(f"   DiD coefficient: {model1.params['daca_post']:.4f} (SE: {model1.bse['daca_post']:.4f}, p: {model1.pvalues['daca_post']:.4f})")

# Model 2: With demographic controls
print("\n   --- Model 2: DiD with Demographic Controls ---")
X2 = df_sample[['daca_eligible', 'post', 'daca_post', 'AGE', 'age_sq', 'female', 'married']]
X2 = sm.add_constant(X2)
model2 = sm.OLS(df_sample['fulltime'], X2).fit(cov_type='HC1')
print(f"   DiD coefficient: {model2.params['daca_post']:.4f} (SE: {model2.bse['daca_post']:.4f}, p: {model2.pvalues['daca_post']:.4f})")

# Model 3: With education controls
print("\n   --- Model 3: DiD with Education Controls ---")
X3 = df_sample[['daca_eligible', 'post', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
                'hs_grad', 'some_college', 'college_plus']]
X3 = sm.add_constant(X3)
model3 = sm.OLS(df_sample['fulltime'], X3).fit(cov_type='HC1')
print(f"   DiD coefficient: {model3.params['daca_post']:.4f} (SE: {model3.bse['daca_post']:.4f}, p: {model3.pvalues['daca_post']:.4f})")

# Model 4: With state fixed effects
print("\n   --- Model 4: DiD with State Fixed Effects ---")
X4_base = df_sample[['daca_eligible', 'post', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
                     'hs_grad', 'some_college', 'college_plus']]
X4 = pd.concat([X4_base, state_dummies], axis=1)
X4 = sm.add_constant(X4)
model4 = sm.OLS(df_sample['fulltime'], X4).fit(cov_type='HC1')
print(f"   DiD coefficient: {model4.params['daca_post']:.4f} (SE: {model4.bse['daca_post']:.4f}, p: {model4.pvalues['daca_post']:.4f})")
print(f"   R-squared: {model4.rsquared:.4f}, N: {int(model4.nobs):,}")

# Model 5: With year fixed effects (PREFERRED)
print("\n   --- Model 5: DiD with State and Year Fixed Effects (PREFERRED) ---")
X5_base = df_sample[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
                     'hs_grad', 'some_college', 'college_plus']]
X5 = pd.concat([X5_base, state_dummies, year_dummies], axis=1)
X5 = sm.add_constant(X5)
model5 = sm.OLS(df_sample['fulltime'], X5).fit(cov_type='HC1')
print(f"   DiD coefficient: {model5.params['daca_post']:.4f} (SE: {model5.bse['daca_post']:.4f}, p: {model5.pvalues['daca_post']:.4f})")
print(f"   95% CI: [{model5.conf_int().loc['daca_post', 0]:.4f}, {model5.conf_int().loc['daca_post', 1]:.4f}]")
print(f"   R-squared: {model5.rsquared:.4f}, N: {int(model5.nobs):,}")

# =============================================================================
# 5. PREFERRED SPECIFICATION SUMMARY
# =============================================================================
print("\n" + "="*60)
print("   PREFERRED ESTIMATE (DiD with state and year FE)")
print("="*60)
print(f"   Effect size: {model5.params['daca_post']:.6f}")
print(f"   Standard error: {model5.bse['daca_post']:.6f}")
print(f"   95% CI: [{model5.conf_int().loc['daca_post', 0]:.6f}, {model5.conf_int().loc['daca_post', 1]:.6f}]")
print(f"   t-statistic: {model5.tvalues['daca_post']:.4f}")
print(f"   p-value: {model5.pvalues['daca_post']:.6f}")
print(f"   Sample size: {int(model5.nobs):,}")
print(f"   R-squared: {model5.rsquared:.4f}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n5. Robustness Checks...")

# 6a. Placebo test: Pre-period only (2006-2011), fake treatment at 2010
print("\n   --- Robustness 1: Placebo Test (Pre-period, fake treatment 2010) ---")
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2010).astype(np.int8)
df_pre['fake_daca_post'] = (df_pre['daca_eligible'] * df_pre['fake_post']).astype(np.int8)

state_dummies_pre = pd.get_dummies(df_pre['STATEFIP'], prefix='state', drop_first=True, dtype=np.int8)
year_dummies_pre = pd.get_dummies(df_pre['YEAR'], prefix='year', drop_first=True, dtype=np.int8)

Xp_base = df_pre[['daca_eligible', 'fake_post', 'fake_daca_post', 'AGE', 'age_sq', 'female', 'married',
                  'hs_grad', 'some_college', 'college_plus']]
Xp = pd.concat([Xp_base.reset_index(drop=True), state_dummies_pre.reset_index(drop=True),
                year_dummies_pre.reset_index(drop=True)], axis=1)
Xp = sm.add_constant(Xp)
placebo = sm.OLS(df_pre['fulltime'].reset_index(drop=True), Xp).fit(cov_type='HC1')
print(f"   Placebo DiD coefficient: {placebo.params['fake_daca_post']:.4f}")
print(f"   Standard error: {placebo.bse['fake_daca_post']:.4f}")
print(f"   p-value: {placebo.pvalues['fake_daca_post']:.4f}")

del df_pre, state_dummies_pre, year_dummies_pre, Xp_base, Xp

# 6b. Alternative outcome: Employment
print("\n   --- Robustness 2: Alternative outcome - Employment ---")
X_emp = pd.concat([df_sample[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
                              'hs_grad', 'some_college', 'college_plus']],
                   state_dummies, year_dummies], axis=1)
X_emp = sm.add_constant(X_emp)
model_emp = sm.OLS(df_sample['employed'], X_emp).fit(cov_type='HC1')
print(f"   DiD coefficient (employment): {model_emp.params['daca_post']:.4f}")
print(f"   Standard error: {model_emp.bse['daca_post']:.4f}")
print(f"   p-value: {model_emp.pvalues['daca_post']:.4f}")

# 6c. Narrower age bandwidth
print("\n   --- Robustness 3: Narrower age bandwidth (born 1976-1986) ---")
df_narrow = df_sample[(df_sample['BIRTHYR'] >= 1976) & (df_sample['BIRTHYR'] <= 1986)].copy()
state_dummies_n = pd.get_dummies(df_narrow['STATEFIP'], prefix='state', drop_first=True, dtype=np.int8)
year_dummies_n = pd.get_dummies(df_narrow['YEAR'], prefix='year', drop_first=True, dtype=np.int8)

Xn_base = df_narrow[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'female', 'married',
                     'hs_grad', 'some_college', 'college_plus']]
Xn = pd.concat([Xn_base.reset_index(drop=True), state_dummies_n.reset_index(drop=True),
                year_dummies_n.reset_index(drop=True)], axis=1)
Xn = sm.add_constant(Xn)
model_narrow = sm.OLS(df_narrow['fulltime'].reset_index(drop=True), Xn).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_narrow.params['daca_post']:.4f}")
print(f"   Standard error: {model_narrow.bse['daca_post']:.4f}")
print(f"   p-value: {model_narrow.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_narrow.nobs):,}")

del df_narrow, state_dummies_n, year_dummies_n, Xn_base, Xn

# 6d. Male subsample
print("\n   --- Robustness 4: Male subsample ---")
df_male = df_sample[df_sample['female'] == 0].copy()
state_dummies_m = pd.get_dummies(df_male['STATEFIP'], prefix='state', drop_first=True, dtype=np.int8)
year_dummies_m = pd.get_dummies(df_male['YEAR'], prefix='year', drop_first=True, dtype=np.int8)

Xm_base = df_male[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'married',
                   'hs_grad', 'some_college', 'college_plus']]
Xm = pd.concat([Xm_base.reset_index(drop=True), state_dummies_m.reset_index(drop=True),
                year_dummies_m.reset_index(drop=True)], axis=1)
Xm = sm.add_constant(Xm)
model_male = sm.OLS(df_male['fulltime'].reset_index(drop=True), Xm).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_male.params['daca_post']:.4f}")
print(f"   Standard error: {model_male.bse['daca_post']:.4f}")
print(f"   p-value: {model_male.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_male.nobs):,}")

del df_male, state_dummies_m, year_dummies_m, Xm_base, Xm

# 6e. Female subsample
print("\n   --- Robustness 5: Female subsample ---")
df_female = df_sample[df_sample['female'] == 1].copy()
state_dummies_f = pd.get_dummies(df_female['STATEFIP'], prefix='state', drop_first=True, dtype=np.int8)
year_dummies_f = pd.get_dummies(df_female['YEAR'], prefix='year', drop_first=True, dtype=np.int8)

Xf_base = df_female[['daca_eligible', 'daca_post', 'AGE', 'age_sq', 'married',
                     'hs_grad', 'some_college', 'college_plus']]
Xf = pd.concat([Xf_base.reset_index(drop=True), state_dummies_f.reset_index(drop=True),
                year_dummies_f.reset_index(drop=True)], axis=1)
Xf = sm.add_constant(Xf)
model_female = sm.OLS(df_female['fulltime'].reset_index(drop=True), Xf).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_female.params['daca_post']:.4f}")
print(f"   Standard error: {model_female.bse['daca_post']:.4f}")
print(f"   p-value: {model_female.pvalues['daca_post']:.4f}")
print(f"   N: {int(model_female.nobs):,}")

del df_female, state_dummies_f, year_dummies_f, Xf_base, Xf

# =============================================================================
# 7. EVENT STUDY
# =============================================================================
print("\n6. Event Study (Dynamic Effects)...")

# Create year-specific interactions
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for yr in years:
    df_sample[f'daca_x_{yr}'] = ((df_sample['daca_eligible'] == 1) & (df_sample['YEAR'] == yr)).astype(np.int8)

# Run event study regression (omit 2011 as reference year)
year_cols = [f'daca_x_{yr}' for yr in years if yr != 2011]
Xe_base = df_sample[['daca_eligible'] + year_cols + ['AGE', 'age_sq', 'female', 'married',
                                                      'hs_grad', 'some_college', 'college_plus']]
Xe = pd.concat([Xe_base, state_dummies, year_dummies], axis=1)
Xe = sm.add_constant(Xe)
event_study = sm.OLS(df_sample['fulltime'], Xe).fit(cov_type='HC1')

print("\n   Event Study Coefficients (relative to 2011):")
event_results = []
for yr in years:
    if yr != 2011:
        coef = event_study.params[f'daca_x_{yr}']
        se = event_study.bse[f'daca_x_{yr}']
        pval = event_study.pvalues[f'daca_x_{yr}']
        ci_low = event_study.conf_int().loc[f'daca_x_{yr}', 0]
        ci_high = event_study.conf_int().loc[f'daca_x_{yr}', 1]
        print(f"   {yr}: {coef:.4f} (SE: {se:.4f}, p: {pval:.4f})")
        event_results.append({
            'Year': yr,
            'Coefficient': coef,
            'Std_Error': se,
            'CI_Lower': ci_low,
            'CI_Upper': ci_high,
            'p_value': pval
        })
    else:
        event_results.append({
            'Year': yr,
            'Coefficient': 0,
            'Std_Error': 0,
            'CI_Lower': 0,
            'CI_Upper': 0,
            'p_value': 1
        })

# =============================================================================
# 8. SAVE RESULTS
# =============================================================================
print("\n7. Saving results...")

# Create summary table
results_dict = {
    'Model': ['(1) Basic', '(2) Demographics', '(3) Education', '(4) State FE', '(5) State+Year FE'],
    'DiD_Coefficient': [model1.params['daca_post'], model2.params['daca_post'],
                        model3.params['daca_post'], model4.params['daca_post'],
                        model5.params['daca_post']],
    'Std_Error': [model1.bse['daca_post'], model2.bse['daca_post'],
                  model3.bse['daca_post'], model4.bse['daca_post'],
                  model5.bse['daca_post']],
    't_stat': [model1.tvalues['daca_post'], model2.tvalues['daca_post'],
               model3.tvalues['daca_post'], model4.tvalues['daca_post'],
               model5.tvalues['daca_post']],
    'p_value': [model1.pvalues['daca_post'], model2.pvalues['daca_post'],
                model3.pvalues['daca_post'], model4.pvalues['daca_post'],
                model5.pvalues['daca_post']],
    'R_squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs)]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_main.csv', index=False)
print("   Saved main results to results_main.csv")

# Save robustness results
robustness_dict = {
    'Test': ['Placebo (2010)', 'Employment outcome', 'Narrow bandwidth', 'Males only', 'Females only'],
    'DiD_Coefficient': [placebo.params['fake_daca_post'], model_emp.params['daca_post'],
                        model_narrow.params['daca_post'], model_male.params['daca_post'],
                        model_female.params['daca_post']],
    'Std_Error': [placebo.bse['fake_daca_post'], model_emp.bse['daca_post'],
                  model_narrow.bse['daca_post'], model_male.bse['daca_post'],
                  model_female.bse['daca_post']],
    'p_value': [placebo.pvalues['fake_daca_post'], model_emp.pvalues['daca_post'],
                model_narrow.pvalues['daca_post'], model_male.pvalues['daca_post'],
                model_female.pvalues['daca_post']],
    'N': [int(placebo.nobs), int(model_emp.nobs), int(model_narrow.nobs),
          int(model_male.nobs), int(model_female.nobs)]
}

robustness_df = pd.DataFrame(robustness_dict)
robustness_df.to_csv('results_robustness.csv', index=False)
print("   Saved robustness results to results_robustness.csv")

# Save event study results
event_df = pd.DataFrame(event_results).sort_values('Year')
event_df.to_csv('results_event_study.csv', index=False)
print("   Saved event study results to results_event_study.csv")

# Save descriptive statistics
desc_stats = df_sample.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'female': ['mean'],
    'married': ['mean'],
    'less_than_hs': ['mean'],
    'hs_grad': ['mean'],
    'some_college': ['mean'],
    'college_plus': ['mean'],
    'PERWT': ['sum']
}).round(4)
desc_stats.to_csv('results_descriptives.csv')
print("   Saved descriptive statistics to results_descriptives.csv")

# Additional statistics for report
print("\n   --- Additional Statistics ---")
print(f"   Mean age DACA eligible: {df_sample[df_sample['daca_eligible']==1]['AGE'].mean():.1f}")
print(f"   Mean age DACA ineligible: {df_sample[df_sample['daca_eligible']==0]['AGE'].mean():.1f}")
print(f"   Proportion female DACA eligible: {df_sample[df_sample['daca_eligible']==1]['female'].mean():.3f}")
print(f"   Proportion female DACA ineligible: {df_sample[df_sample['daca_eligible']==0]['female'].mean():.3f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Print final summary for report
print(f"\n*** FINAL RESULTS SUMMARY ***")
print(f"Preferred estimate (Model 5):")
print(f"  Effect of DACA eligibility on full-time employment: {model5.params['daca_post']:.4f}")
print(f"  Standard error: {model5.bse['daca_post']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['daca_post', 0]:.4f}, {model5.conf_int().loc['daca_post', 1]:.4f}]")
print(f"  Sample size: {int(model5.nobs):,}")
print(f"  Interpretation: DACA eligibility {'increased' if model5.params['daca_post'] > 0 else 'decreased'} ")
print(f"                  full-time employment by {abs(model5.params['daca_post']*100):.2f} percentage points")

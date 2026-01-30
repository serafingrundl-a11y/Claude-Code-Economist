#!/usr/bin/env python3
"""
DACA Replication Study - Analysis Script
Replication 48

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Design: Difference-in-Differences
- Treatment: Ages 26-30 on June 15, 2012 (born 1982-1986)
- Control: Ages 31-35 on June 15, 2012 (born 1977-1981)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("="*70)

# =============================================================================
# 1. LOAD AND PREPARE DATA (CHUNKED TO MANAGE MEMORY)
# =============================================================================
print("\n1. LOADING DATA (chunked processing)...")

# Define columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
               'UHRSWORK', 'MARST']

# Load data in chunks and filter
chunks = []
chunk_size = 500000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    # Filter for DACA-eligible population immediately
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3) &  # Not a citizen
        (chunk['BIRTHYR'] >= 1977) & (chunk['BIRTHYR'] <= 1986) &  # Relevant age range
        (chunk['YEAR'] != 2012)  # Exclude 2012
    ].copy()

    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

print(f"   Processed all chunks, concatenating...")

df_sample = pd.concat(chunks, ignore_index=True)
print(f"   Initial filtered observations: {len(df_sample):,}")

# Additional filters
# Must have arrived before age 16
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[(df_sample['age_at_immig'] < 16) & (df_sample['age_at_immig'] >= 0)].copy()
print(f"   After requiring arrival before age 16: {len(df_sample):,}")

# Arrived by 2007 (continuous residence since June 15, 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"   After YRIMMIG<=2007 (residence since 2007): {len(df_sample):,}")

# Final sample summary
print(f"\n   Years in data: {sorted(df_sample['YEAR'].unique())}")

# =============================================================================
# 2. VARIABLE CONSTRUCTION
# =============================================================================
print("\n2. CONSTRUCTING VARIABLES...")

# Treatment indicator
df_sample['treat'] = (df_sample['BIRTHYR'] >= 1982).astype(int)
print(f"   Treatment group (born 1982-1986): {df_sample['treat'].sum():,}")
print(f"   Control group (born 1977-1981): {(1-df_sample['treat']).sum():,}")

# Post-treatment indicator
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"   Post-period (2013-2016): {df_sample['post'].sum():,}")
print(f"   Pre-period (2006-2011): {(1-df_sample['post']).sum():,}")

# Interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# Full-time employment outcome (35+ hours per week)
# UHRSWORK = 0 means not working; we want 35+ hours for full-time
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Any employment indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 7).astype(int)  # Some college or more

# Sex indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Marital status
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Age in survey year
df_sample['age'] = df_sample['YEAR'] - df_sample['BIRTHYR']

print(f"\n   Full-time employment rate: {df_sample['fulltime'].mean()*100:.1f}%")
print(f"   Employment rate: {df_sample['employed'].mean()*100:.1f}%")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*70)
print("3. DESCRIPTIVE STATISTICS")
print("="*70)

# Summary by treatment and period
print("\n3.1 Full-time employment rates by group and period:")
summary = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'PERWT': 'sum'
}).round(4)
print(summary)

# Weighted means
print("\n3.2 Weighted full-time employment rates:")
for t in [0, 1]:
    for p in [0, 1]:
        subset = df_sample[(df_sample['treat'] == t) & (df_sample['post'] == p)]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        group = "Treatment" if t == 1 else "Control"
        period = "Post" if p == 1 else "Pre"
        print(f"   {group}, {period}: {weighted_mean*100:.2f}%")

# Raw DiD calculation
pre_treat = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 0)]
post_treat = df_sample[(df_sample['treat'] == 1) & (df_sample['post'] == 1)]
pre_control = df_sample[(df_sample['treat'] == 0) & (df_sample['post'] == 0)]
post_control = df_sample[(df_sample['treat'] == 0) & (df_sample['post'] == 1)]

# Weighted means
w_pre_treat = np.average(pre_treat['fulltime'], weights=pre_treat['PERWT'])
w_post_treat = np.average(post_treat['fulltime'], weights=post_treat['PERWT'])
w_pre_control = np.average(pre_control['fulltime'], weights=pre_control['PERWT'])
w_post_control = np.average(post_control['fulltime'], weights=post_control['PERWT'])

raw_did = (w_post_treat - w_pre_treat) - (w_post_control - w_pre_control)
print(f"\n3.3 Raw (weighted) DiD estimate: {raw_did*100:.3f} percentage points")
print(f"   Treatment change: {(w_post_treat - w_pre_treat)*100:.3f} pp")
print(f"   Control change: {(w_post_control - w_pre_control)*100:.3f} pp")

# =============================================================================
# 4. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "="*70)
print("4. MAIN REGRESSION RESULTS")
print("="*70)

# Model 1: Basic DiD
print("\n4.1 Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_sample).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(model1.summary().tables[1])

# Model 2: Basic DiD with weights
print("\n4.2 Model 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls
print("\n4.3 Model 3: DiD with demographic controls (weighted)")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + educ_college',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n4.4 Model 4: DiD with year fixed effects (weighted)")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treat + treat_post + C(year_factor)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
# Print just the key coefficient
print(f"   treat_post coefficient: {model4.params['treat_post']:.5f}")
print(f"   Std error: {model4.bse['treat_post']:.5f}")
print(f"   t-stat: {model4.tvalues['treat_post']:.3f}")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with year and state fixed effects
print("\n4.5 Model 5: DiD with year and state fixed effects (weighted)")
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treat + treat_post + C(year_factor) + C(state_factor)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='HC1')  # Can't cluster by state when state FE included
print(f"   treat_post coefficient: {model5.params['treat_post']:.5f}")
print(f"   Std error: {model5.bse['treat_post']:.5f}")
print(f"   t-stat: {model5.tvalues['treat_post']:.3f}")
print(f"   p-value: {model5.pvalues['treat_post']:.4f}")

# Model 6: Full specification
print("\n4.6 Model 6: Full specification (year FE + state FE + demographics)")
model6 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + educ_college + C(year_factor) + C(state_factor)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='HC1')
print(f"   treat_post coefficient: {model6.params['treat_post']:.5f}")
print(f"   Std error: {model6.bse['treat_post']:.5f}")
print(f"   t-stat: {model6.tvalues['treat_post']:.3f}")
print(f"   p-value: {model6.pvalues['treat_post']:.4f}")

# =============================================================================
# 5. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("5. ROBUSTNESS CHECKS")
print("="*70)

# 5.1 Placebo test using pre-period only (2006-2008 vs 2009-2011)
print("\n5.1 Placebo test (pre-period only: 2009-2011 vs 2006-2008)")
df_pre = df_sample[df_sample['post'] == 0].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat_post'] = df_pre['treat'] * df_pre['placebo_post']

placebo_model = smf.wls('fulltime ~ treat + placebo_post + placebo_treat_post',
                         data=df_pre, weights=df_pre['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print(f"   Placebo DiD coefficient: {placebo_model.params['placebo_treat_post']:.5f}")
print(f"   Std error: {placebo_model.bse['placebo_treat_post']:.5f}")
print(f"   p-value: {placebo_model.pvalues['placebo_treat_post']:.4f}")

# 5.2 Alternative outcome: Any employment
print("\n5.2 Alternative outcome: Any employment")
emp_model = smf.wls('employed ~ treat + post + treat_post',
                     data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"   treat_post coefficient: {emp_model.params['treat_post']:.5f}")
print(f"   Std error: {emp_model.bse['treat_post']:.5f}")
print(f"   p-value: {emp_model.pvalues['treat_post']:.4f}")

# 5.3 By sex
print("\n5.3 Heterogeneity by sex:")
for sex, label in [(0, 'Male'), (1, 'Female')]:
    df_sex = df_sample[df_sample['female'] == sex]
    sex_model = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_sex, weights=df_sex['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_sex['STATEFIP']})
    print(f"   {label}: coef={sex_model.params['treat_post']:.5f}, "
          f"se={sex_model.bse['treat_post']:.5f}, p={sex_model.pvalues['treat_post']:.4f}")

# 5.4 Event study - yearly effects
print("\n5.4 Event study - yearly treatment effects:")
df_sample['treat_2006'] = df_sample['treat'] * (df_sample['YEAR'] == 2006).astype(int)
df_sample['treat_2007'] = df_sample['treat'] * (df_sample['YEAR'] == 2007).astype(int)
df_sample['treat_2008'] = df_sample['treat'] * (df_sample['YEAR'] == 2008).astype(int)
df_sample['treat_2009'] = df_sample['treat'] * (df_sample['YEAR'] == 2009).astype(int)
df_sample['treat_2010'] = df_sample['treat'] * (df_sample['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_sample['treat_2013'] = df_sample['treat'] * (df_sample['YEAR'] == 2013).astype(int)
df_sample['treat_2014'] = df_sample['treat'] * (df_sample['YEAR'] == 2014).astype(int)
df_sample['treat_2015'] = df_sample['treat'] * (df_sample['YEAR'] == 2015).astype(int)
df_sample['treat_2016'] = df_sample['treat'] * (df_sample['YEAR'] == 2016).astype(int)

event_model = smf.wls('fulltime ~ treat + C(year_factor) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016',
                       data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = event_model.params[f'treat_{year}']
    se = event_model.bse[f'treat_{year}']
    print(f"   Year {year}: coef={coef:.5f}, se={se:.5f}")

# =============================================================================
# 6. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*70)
print("6. SUMMARY OF RESULTS")
print("="*70)

# Preferred specification: Model 2 (basic weighted DiD with state-clustered SE)
preferred_coef = model2.params['treat_post']
preferred_se = model2.bse['treat_post']
preferred_ci_low = preferred_coef - 1.96 * preferred_se
preferred_ci_high = preferred_coef + 1.96 * preferred_se
preferred_pval = model2.pvalues['treat_post']
n_obs = len(df_sample)

print(f"\nPREFERRED ESTIMATE (Model 2: Basic weighted DiD)")
print(f"   Effect size: {preferred_coef:.5f} ({preferred_coef*100:.3f} percentage points)")
print(f"   Standard error: {preferred_se:.5f}")
print(f"   95% CI: [{preferred_ci_low:.5f}, {preferred_ci_high:.5f}]")
print(f"           [{preferred_ci_low*100:.3f} pp, {preferred_ci_high*100:.3f} pp]")
print(f"   t-statistic: {model2.tvalues['treat_post']:.3f}")
print(f"   p-value: {preferred_pval:.4f}")
print(f"   Sample size: {n_obs:,}")

print(f"\nINTERPRETATION:")
if preferred_pval < 0.05:
    direction = "increased" if preferred_coef > 0 else "decreased"
    print(f"   DACA eligibility {direction} the probability of full-time employment")
    print(f"   by {abs(preferred_coef)*100:.2f} percentage points (p < 0.05).")
else:
    print(f"   The effect is not statistically significant at the 5% level.")
    print(f"   We cannot reject the null hypothesis of no effect.")

# =============================================================================
# 7. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "="*70)
print("7. SAVING RESULTS")
print("="*70)

results_dict = {
    'preferred_effect': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci_low,
    'preferred_ci_high': preferred_ci_high,
    'preferred_pval': preferred_pval,
    'sample_size': n_obs,
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model3_coef': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
    'model4_coef': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'model5_coef': model5.params['treat_post'],
    'model5_se': model5.bse['treat_post'],
    'model6_coef': model6.params['treat_post'],
    'model6_se': model6.bse['treat_post'],
    'placebo_coef': placebo_model.params['placebo_treat_post'],
    'placebo_se': placebo_model.bse['placebo_treat_post'],
    'employment_coef': emp_model.params['treat_post'],
    'employment_se': emp_model.bse['treat_post'],
    'raw_did': raw_did,
    'w_pre_treat': w_pre_treat,
    'w_post_treat': w_post_treat,
    'w_pre_control': w_pre_control,
    'w_post_control': w_post_control,
    'n_treatment': df_sample['treat'].sum(),
    'n_control': (1-df_sample['treat']).sum(),
    'n_pre': (1-df_sample['post']).sum(),
    'n_post': df_sample['post'].sum(),
}

# Save to CSV
results_df = pd.DataFrame([results_dict])
results_df.to_csv('results_48.csv', index=False)
print("   Results saved to results_48.csv")

# Create summary tables for the report
print("\n   Creating summary tables...")

# Table 1: Descriptive statistics
desc_stats = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'age': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)
desc_stats.to_csv('descriptive_stats_48.csv')

# Table 2: Main regression results
reg_table = pd.DataFrame({
    'Model': ['(1) Basic', '(2) Weighted', '(3) Demographics', '(4) Year FE', '(5) Year+State FE', '(6) Full'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                   model3.params['treat_post'], model4.params['treat_post'],
                   model5.params['treat_post'], model6.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                 model3.bse['treat_post'], model4.bse['treat_post'],
                 model5.bse['treat_post'], model6.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
               model3.pvalues['treat_post'], model4.pvalues['treat_post'],
               model5.pvalues['treat_post'], model6.pvalues['treat_post']]
})
reg_table.to_csv('regression_results_48.csv', index=False)

# Event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [event_model.params.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                   [event_model.params.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'Std_Error': [event_model.bse.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
                 [event_model.bse.get(f'treat_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
event_results.to_csv('event_study_48.csv', index=False)

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

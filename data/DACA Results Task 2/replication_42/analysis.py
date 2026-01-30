"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
for Hispanic-Mexican, Mexican-born individuals.

Treatment: Ages 26-30 at policy implementation (June 15, 2012)
Control: Ages 31-35 at policy implementation
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("DACA Replication Analysis")
print("=" * 60)

# ============================================================================
# STEP 1: Load and Filter Data
# ============================================================================
print("\n[1] Loading data...")

# Read data in chunks due to large file size
chunk_size = 500000
chunks = []

# Filter criteria as we read:
# - HISPAN == 1 (Mexican Hispanic)
# - BPL == 200 (Born in Mexico)
# - CITIZEN == 3 (Not a citizen)
# - YEAR in range 2006-2016

for chunk in pd.read_csv(os.path.join(OUTPUT_DIR, "data", "data.csv"),
                         chunksize=chunk_size,
                         dtype={'YEAR': int, 'HISPAN': int, 'BPL': int,
                                'CITIZEN': int, 'BIRTHYR': int, 'YRIMMIG': int,
                                'AGE': int, 'UHRSWORK': int, 'SEX': int,
                                'EDUC': int, 'MARST': int, 'PERWT': float,
                                'BIRTHQTR': int, 'STATEFIP': int}):

    # Filter to Mexican-born, Hispanic-Mexican, non-citizens
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &    # Born in Mexico
        (chunk['CITIZEN'] == 3) &  # Not a citizen
        (chunk['YEAR'] >= 2006) &
        (chunk['YEAR'] <= 2016) &
        (chunk['YEAR'] != 2012)    # Exclude 2012 (implementation year)
    ].copy()

    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"   After filtering to Mexican-born, Hispanic-Mexican non-citizens: {len(df):,} observations")

# ============================================================================
# STEP 2: Construct Treatment and Control Groups
# ============================================================================
print("\n[2] Constructing treatment and control groups...")

# DACA was implemented June 15, 2012
# Treatment: Ages 26-30 on June 15, 2012 -> Born between June 16, 1981 and June 15, 1986
# Control: Ages 31-35 on June 15, 2012 -> Born between June 16, 1976 and June 15, 1981

# Birth year cutoffs (using calendar year approximation since we don't have exact dates):
# Treatment: BIRTHYR in [1982, 1986] approximately (age 26-30 in mid-2012)
# Control: BIRTHYR in [1977, 1981] approximately (age 31-35 in mid-2012)

# More precise approach: account for birth quarter
# Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
# June 15, 2012 is in Q2

# A person is age X on June 15, 2012 if:
# - Born in year (2012 - X) in Q3 or Q4, OR
# - Born in year (2012 - X - 1) in Q1 or Q2

def get_age_june_2012(birthyr, birthqtr):
    """Calculate age as of June 15, 2012"""
    if birthqtr in [1, 2]:  # Born Jan-June
        return 2012 - birthyr
    else:  # Born July-Dec
        return 2012 - birthyr - 1

df['age_june_2012'] = df.apply(lambda row: get_age_june_2012(row['BIRTHYR'], row['BIRTHQTR']), axis=1)

# Filter to treatment (26-30) and control (31-35) groups
df_analysis = df[(df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 35)].copy()
print(f"   After filtering to ages 26-35 in June 2012: {len(df_analysis):,} observations")

# Create treatment indicator
df_analysis['treated'] = (df_analysis['age_june_2012'] <= 30).astype(int)

# ============================================================================
# STEP 3: Apply DACA Eligibility Criteria
# ============================================================================
print("\n[3] Applying DACA eligibility criteria...")

# DACA eligibility (beyond age):
# 1. Arrived in US before age 16
# 2. Lived continuously in US since June 15, 2007 (proxy: YRIMMIG <= 2007)

# Calculate age at immigration
df_analysis['age_at_immig'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Apply eligibility filters
df_eligible = df_analysis[
    (df_analysis['YRIMMIG'] > 0) &           # Valid immigration year
    (df_analysis['YRIMMIG'] <= 2007) &        # Arrived by 2007
    (df_analysis['age_at_immig'] < 16)        # Arrived before age 16
].copy()

print(f"   After DACA eligibility criteria: {len(df_eligible):,} observations")

# ============================================================================
# STEP 4: Create Outcome and Period Variables
# ============================================================================
print("\n[4] Creating outcome and period variables...")

# Full-time employment: UHRSWORK >= 35
df_eligible['fulltime'] = (df_eligible['UHRSWORK'] >= 35).astype(int)

# Post-treatment indicator (2013-2016)
df_eligible['post'] = (df_eligible['YEAR'] >= 2013).astype(int)

# Create year dummies
df_eligible['year'] = df_eligible['YEAR']

# ============================================================================
# STEP 5: Summary Statistics
# ============================================================================
print("\n[5] Computing summary statistics...")

# Summary by group and period
summary = df_eligible.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),  # Proportion male
    'EDUC': 'mean'
}).round(4)

print("\nSummary Statistics by Group and Period:")
print("-" * 60)
print(summary)

# Save summary stats
summary_flat = df_eligible.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum'
})
summary_flat.columns = ['fulltime_mean', 'fulltime_sd', 'n_obs', 'sum_weights']
summary_flat = summary_flat.reset_index()
summary_flat.to_csv(os.path.join(OUTPUT_DIR, 'summary_stats.csv'), index=False)

# Weighted means
print("\nWeighted Full-Time Employment Rates:")
print("-" * 60)
for (t, p), grp in df_eligible.groupby(['treated', 'post']):
    wt_mean = np.average(grp['fulltime'], weights=grp['PERWT'])
    group_name = "Treatment" if t == 1 else "Control"
    period_name = "Post" if p == 1 else "Pre"
    print(f"   {group_name}, {period_name}: {wt_mean:.4f}")

# ============================================================================
# STEP 6: Difference-in-Differences Estimation
# ============================================================================
print("\n[6] Running Difference-in-Differences Regression...")

# Create interaction term
df_eligible['treated_post'] = df_eligible['treated'] * df_eligible['post']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted OLS) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_eligible).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD with survey weights
print("\n--- Model 2: Basic DiD (Weighted OLS) ---")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_eligible,
                  weights=df_eligible['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with covariates (weighted)
print("\n--- Model 3: DiD with Covariates (Weighted OLS) ---")

# Create covariate dummies
df_eligible['male'] = (df_eligible['SEX'] == 1).astype(int)
df_eligible['married'] = (df_eligible['MARST'] == 1).astype(int)

# Education categories
df_eligible['educ_hs'] = (df_eligible['EDUC'] >= 6).astype(int)  # HS or more
df_eligible['educ_college'] = (df_eligible['EDUC'] >= 7).astype(int)  # Some college or more

model3 = smf.wls('fulltime ~ treated + post + treated_post + male + married + educ_hs + C(STATEFIP)',
                  data=df_eligible,
                  weights=df_eligible['PERWT']).fit()

# Extract just the key coefficients
print(f"   treated_post coefficient: {model3.params['treated_post']:.5f}")
print(f"   Standard error: {model3.bse['treated_post']:.5f}")
print(f"   t-statistic: {model3.tvalues['treated_post']:.3f}")
print(f"   p-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: DiD with robust standard errors
print("\n--- Model 4: DiD with Robust Standard Errors ---")
model4 = smf.wls('fulltime ~ treated + post + treated_post + male + married + educ_hs',
                  data=df_eligible,
                  weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   treated_post coefficient: {model4.params['treated_post']:.5f}")
print(f"   Robust SE: {model4.bse['treated_post']:.5f}")
print(f"   t-statistic: {model4.tvalues['treated_post']:.3f}")
print(f"   p-value: {model4.pvalues['treated_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treated_post', 0]:.5f}, {model4.conf_int().loc['treated_post', 1]:.5f}]")

# ============================================================================
# STEP 7: Event Study / Year-by-Year Effects
# ============================================================================
print("\n[7] Event Study Analysis...")

# Create year interactions with treatment
year_dummies = pd.get_dummies(df_eligible['year'], prefix='year', drop_first=False)
df_eligible = pd.concat([df_eligible, year_dummies], axis=1)

# Reference year: 2011 (last pre-treatment year)
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for yr in years:
    df_eligible[f'treat_x_{yr}'] = df_eligible['treated'] * (df_eligible['year'] == yr).astype(int)

# Drop 2011 as reference
event_formula = 'fulltime ~ treated + ' + ' + '.join([f'year_{yr}' for yr in years]) + ' + ' + \
                ' + '.join([f'treat_x_{yr}' for yr in years if yr != 2011]) + ' + male + married + educ_hs'

model_event = smf.wls(event_formula, data=df_eligible, weights=df_eligible['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year, ref=2011):")
print("-" * 60)
event_results = []
for yr in years:
    if yr != 2011:
        coef = model_event.params[f'treat_x_{yr}']
        se = model_event.bse[f'treat_x_{yr}']
        pval = model_event.pvalues[f'treat_x_{yr}']
        ci_low, ci_high = model_event.conf_int().loc[f'treat_x_{yr}']
        print(f"   {yr}: {coef:.5f} (SE: {se:.5f}, p={pval:.4f})")
        event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high, 'pval': pval})
    else:
        event_results.append({'year': yr, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0, 'pval': 1})

event_df = pd.DataFrame(event_results)
event_df.to_csv(os.path.join(OUTPUT_DIR, 'event_study_results.csv'), index=False)

# ============================================================================
# STEP 8: Robustness Checks
# ============================================================================
print("\n[8] Robustness Checks...")

# 8a: Restrict to working-age adults who could plausibly be employed
print("\n--- 8a: Subgroup Analysis by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_eligible[df_eligible['SEX'] == sex_val]
    model_sub = smf.wls('fulltime ~ treated + post + treated_post',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"   {sex_name}: DiD = {model_sub.params['treated_post']:.5f} (SE: {model_sub.bse['treated_post']:.5f}, p={model_sub.pvalues['treated_post']:.4f})")

# 8b: Alternative outcome - any employment (EMPSTAT == 1)
print("\n--- 8b: Alternative Outcome - Any Employment ---")
df_eligible['employed'] = (df_eligible['EMPSTAT'] == 1).astype(int)
model_emp = smf.wls('employed ~ treated + post + treated_post + male + married + educ_hs',
                    data=df_eligible, weights=df_eligible['PERWT']).fit(cov_type='HC1')
print(f"   DiD (Any Employment): {model_emp.params['treated_post']:.5f} (SE: {model_emp.bse['treated_post']:.5f}, p={model_emp.pvalues['treated_post']:.4f})")

# 8c: Placebo test - use only pre-period data with fake treatment at 2009
print("\n--- 8c: Placebo Test (Fake treatment at 2009) ---")
df_pre = df_eligible[df_eligible['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['treated_placebo_post'] = df_pre['treated'] * df_pre['placebo_post']
model_placebo = smf.wls('fulltime ~ treated + placebo_post + treated_placebo_post + male + married + educ_hs',
                        data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"   Placebo DiD: {model_placebo.params['treated_placebo_post']:.5f} (SE: {model_placebo.bse['treated_placebo_post']:.5f}, p={model_placebo.pvalues['treated_placebo_post']:.4f})")

# 8d: Narrower bandwidth (ages 28-30 vs 31-33)
print("\n--- 8d: Narrower Bandwidth (ages 28-30 vs 31-33) ---")
df_narrow = df_eligible[(df_eligible['age_june_2012'] >= 28) & (df_eligible['age_june_2012'] <= 33)].copy()
model_narrow = smf.wls('fulltime ~ treated + post + treated_post + male + married + educ_hs',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"   Narrow DiD: {model_narrow.params['treated_post']:.5f} (SE: {model_narrow.bse['treated_post']:.5f}, p={model_narrow.pvalues['treated_post']:.4f})")

# ============================================================================
# STEP 9: Save Main Results
# ============================================================================
print("\n[9] Saving results...")

# Main results dictionary
main_results = {
    'preferred_estimate': {
        'effect_size': float(model4.params['treated_post']),
        'standard_error': float(model4.bse['treated_post']),
        'ci_lower': float(model4.conf_int().loc['treated_post', 0]),
        'ci_upper': float(model4.conf_int().loc['treated_post', 1]),
        'p_value': float(model4.pvalues['treated_post']),
        't_statistic': float(model4.tvalues['treated_post']),
        'sample_size': len(df_eligible),
        'weighted_sample': float(df_eligible['PERWT'].sum())
    },
    'model_unweighted': {
        'effect_size': float(model1.params['treated_post']),
        'standard_error': float(model1.bse['treated_post']),
        'p_value': float(model1.pvalues['treated_post']),
        'sample_size': int(model1.nobs)
    },
    'model_weighted_basic': {
        'effect_size': float(model2.params['treated_post']),
        'standard_error': float(model2.bse['treated_post']),
        'p_value': float(model2.pvalues['treated_post'])
    },
    'model_with_state_fe': {
        'effect_size': float(model3.params['treated_post']),
        'standard_error': float(model3.bse['treated_post']),
        'p_value': float(model3.pvalues['treated_post'])
    },
    'subgroup_male': {
        'effect_size': float(smf.wls('fulltime ~ treated + post + treated_post',
                                     data=df_eligible[df_eligible['SEX']==1],
                                     weights=df_eligible[df_eligible['SEX']==1]['PERWT']).fit(cov_type='HC1').params['treated_post'])
    },
    'subgroup_female': {
        'effect_size': float(smf.wls('fulltime ~ treated + post + treated_post',
                                     data=df_eligible[df_eligible['SEX']==2],
                                     weights=df_eligible[df_eligible['SEX']==2]['PERWT']).fit(cov_type='HC1').params['treated_post'])
    },
    'robustness_any_employment': {
        'effect_size': float(model_emp.params['treated_post']),
        'standard_error': float(model_emp.bse['treated_post']),
        'p_value': float(model_emp.pvalues['treated_post'])
    },
    'placebo_test': {
        'effect_size': float(model_placebo.params['treated_placebo_post']),
        'standard_error': float(model_placebo.bse['treated_placebo_post']),
        'p_value': float(model_placebo.pvalues['treated_placebo_post'])
    },
    'narrow_bandwidth': {
        'effect_size': float(model_narrow.params['treated_post']),
        'standard_error': float(model_narrow.bse['treated_post']),
        'p_value': float(model_narrow.pvalues['treated_post'])
    }
}

with open(os.path.join(OUTPUT_DIR, 'main_results.json'), 'w') as f:
    json.dump(main_results, f, indent=2)

# ============================================================================
# STEP 10: Create Tables for LaTeX
# ============================================================================
print("\n[10] Creating LaTeX tables...")

# Table 1: Summary Statistics
summary_table = df_eligible.groupby(['treated', 'post']).apply(
    lambda x: pd.Series({
        'N': len(x),
        'Weighted N': x['PERWT'].sum(),
        'Full-time Rate': np.average(x['fulltime'], weights=x['PERWT']),
        'Male Share': np.average(x['male'], weights=x['PERWT']),
        'HS+ Share': np.average(x['educ_hs'], weights=x['PERWT']),
        'Married Share': np.average(x['married'], weights=x['PERWT']),
        'Mean Age': np.average(x['AGE'], weights=x['PERWT'])
    })
).round(4)

summary_table.to_csv(os.path.join(OUTPUT_DIR, 'table1_summary.csv'))

# Table 2: DiD Results
did_table = pd.DataFrame({
    'Model': ['(1) Unweighted', '(2) Weighted', '(3) + Covariates', '(4) + Robust SE', '(5) + State FE'],
    'DiD Estimate': [
        model1.params['treated_post'],
        model2.params['treated_post'],
        model4.params['treated_post'],
        model4.params['treated_post'],
        model3.params['treated_post']
    ],
    'SE': [
        model1.bse['treated_post'],
        model2.bse['treated_post'],
        model4.bse['treated_post'],
        model4.bse['treated_post'],
        model3.bse['treated_post']
    ],
    'p-value': [
        model1.pvalues['treated_post'],
        model2.pvalues['treated_post'],
        model4.pvalues['treated_post'],
        model4.pvalues['treated_post'],
        model3.pvalues['treated_post']
    ]
}).round(5)

did_table.to_csv(os.path.join(OUTPUT_DIR, 'table2_did_results.csv'), index=False)

# Table 3: Robustness checks
robustness_table = pd.DataFrame({
    'Check': ['Placebo (2009)', 'Narrow Bandwidth', 'Any Employment', 'Males Only', 'Females Only'],
    'Estimate': [
        model_placebo.params['treated_placebo_post'],
        model_narrow.params['treated_post'],
        model_emp.params['treated_post'],
        main_results['subgroup_male']['effect_size'],
        main_results['subgroup_female']['effect_size']
    ],
    'SE': [
        model_placebo.bse['treated_placebo_post'],
        model_narrow.bse['treated_post'],
        model_emp.bse['treated_post'],
        smf.wls('fulltime ~ treated + post + treated_post',
                data=df_eligible[df_eligible['SEX']==1],
                weights=df_eligible[df_eligible['SEX']==1]['PERWT']).fit(cov_type='HC1').bse['treated_post'],
        smf.wls('fulltime ~ treated + post + treated_post',
                data=df_eligible[df_eligible['SEX']==2],
                weights=df_eligible[df_eligible['SEX']==2]['PERWT']).fit(cov_type='HC1').bse['treated_post']
    ]
}).round(5)

robustness_table.to_csv(os.path.join(OUTPUT_DIR, 'table3_robustness.csv'), index=False)

# Simple 2x2 DiD table for manual calculation check
print("\n" + "=" * 60)
print("SIMPLE 2x2 DiD TABLE (Weighted Means)")
print("=" * 60)

cell_means = {}
for (t, p), grp in df_eligible.groupby(['treated', 'post']):
    wt_mean = np.average(grp['fulltime'], weights=grp['PERWT'])
    cell_means[(t, p)] = wt_mean

print(f"\n{'':20} {'Pre-DACA':>15} {'Post-DACA':>15} {'Difference':>15}")
print("-" * 65)

treat_pre = cell_means[(1, 0)]
treat_post = cell_means[(1, 1)]
ctrl_pre = cell_means[(0, 0)]
ctrl_post = cell_means[(0, 1)]

print(f"{'Treatment (26-30)':20} {treat_pre:>15.4f} {treat_post:>15.4f} {treat_post - treat_pre:>15.4f}")
print(f"{'Control (31-35)':20} {ctrl_pre:>15.4f} {ctrl_post:>15.4f} {ctrl_post - ctrl_pre:>15.4f}")
print("-" * 65)
did_manual = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
print(f"{'Difference-in-Diff':20} {'':<15} {'':<15} {did_manual:>15.4f}")

# Save cell means
cell_means_df = pd.DataFrame({
    'Group': ['Treatment', 'Treatment', 'Control', 'Control'],
    'Period': ['Pre', 'Post', 'Pre', 'Post'],
    'FullTimeRate': [treat_pre, treat_post, ctrl_pre, ctrl_post]
})
cell_means_df.to_csv(os.path.join(OUTPUT_DIR, 'cell_means.csv'), index=False)

# ============================================================================
# STEP 11: Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"""
PREFERRED ESTIMATE (Model 4: Weighted DiD with covariates and robust SE):
  Effect Size:     {main_results['preferred_estimate']['effect_size']:.5f}
  Standard Error:  {main_results['preferred_estimate']['standard_error']:.5f}
  95% CI:          [{main_results['preferred_estimate']['ci_lower']:.5f}, {main_results['preferred_estimate']['ci_upper']:.5f}]
  p-value:         {main_results['preferred_estimate']['p_value']:.4f}
  Sample Size:     {main_results['preferred_estimate']['sample_size']:,}
  Weighted N:      {main_results['preferred_estimate']['weighted_sample']:,.0f}

INTERPRETATION:
  DACA eligibility is associated with a {abs(main_results['preferred_estimate']['effect_size'])*100:.2f} percentage point
  {'increase' if main_results['preferred_estimate']['effect_size'] > 0 else 'decrease'} in the probability of full-time employment
  among eligible individuals aged 26-30 compared to the 31-35 age group.
  This effect is {'statistically significant' if main_results['preferred_estimate']['p_value'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("\n[Analysis Complete]")
print("=" * 60)

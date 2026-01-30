"""
DACA Impact on Full-Time Employment - Replication Analysis
Replication ID: 77

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment?

Treatment: Ages 26-30 as of June 15, 2012
Control: Ages 31-35 as of June 15, 2012
Method: Difference-in-differences
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
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("="*80)

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("\n[STEP 1] Loading data...")

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Sample Selection - DACA Eligible Population
# =============================================================================
print("\n[STEP 2] Applying sample restrictions...")

# Initial count
n_total = len(df)

# Restriction 1: Hispanic-Mexican (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
n_hispanic_mexican = len(df_sample)
print(f"  After Hispanic-Mexican restriction: {n_hispanic_mexican:,} ({n_hispanic_mexican/n_total*100:.1f}%)")

# Restriction 2: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
n_born_mexico = len(df_sample)
print(f"  After born in Mexico restriction: {n_born_mexico:,} ({n_born_mexico/n_hispanic_mexican*100:.1f}%)")

# Restriction 3: Non-citizen (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
n_noncitizen = len(df_sample)
print(f"  After non-citizen restriction: {n_noncitizen:,} ({n_noncitizen/n_born_mexico*100:.1f}%)")

# Restriction 4: Arrived before age 16
# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immig'] < 16].copy()
n_arrived_young = len(df_sample)
print(f"  After arrived before age 16: {n_arrived_young:,} ({n_arrived_young/n_noncitizen*100:.1f}%)")

# Restriction 5: Calculate age as of June 15, 2012
# For simplicity, use BIRTHYR to determine age in 2012
df_sample['age_at_daca'] = 2012 - df_sample['BIRTHYR']

# Treatment group: Born 1982-1986 (ages 26-30 in 2012)
# Control group: Born 1977-1981 (ages 31-35 in 2012)
df_sample = df_sample[(df_sample['BIRTHYR'] >= 1977) & (df_sample['BIRTHYR'] <= 1986)].copy()
n_age_range = len(df_sample)
print(f"  After age range restriction (26-35 in 2012): {n_age_range:,} ({n_age_range/n_arrived_young*100:.1f}%)")

# Restriction 6: Exclude 2012 (implementation year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
n_excl_2012 = len(df_sample)
print(f"  After excluding 2012: {n_excl_2012:,}")

print(f"\n  Final analytic sample: {len(df_sample):,} person-year observations")

# =============================================================================
# STEP 3: Create Analysis Variables
# =============================================================================
print("\n[STEP 3] Creating analysis variables...")

# Treatment indicator: Born 1982-1986 (ages 26-30 in 2012)
df_sample['treated'] = (df_sample['BIRTHYR'] >= 1982).astype(int)

# Post-period indicator: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Interaction term
df_sample['treated_post'] = df_sample['treated'] * df_sample['post']

# Outcome: Full-time employment (usually works 35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator (EMPSTAT == 1)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Full-time conditional on employed
df_sample['fulltime_if_emp'] = np.where(df_sample['employed'] == 1,
                                         df_sample['fulltime'], np.nan)

print(f"  Treatment group (ages 26-30): {df_sample['treated'].sum():,} observations")
print(f"  Control group (ages 31-35): {(1-df_sample['treated']).sum():,} observations")
print(f"  Pre-period (2006-2011): {(1-df_sample['post']).sum():,} observations")
print(f"  Post-period (2013-2016): {df_sample['post'].sum():,} observations")

# =============================================================================
# STEP 4: Descriptive Statistics
# =============================================================================
print("\n[STEP 4] Descriptive statistics...")

# Create covariates
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['less_than_hs'] = (df_sample['EDUCD'] < 62).astype(int)
df_sample['hs_diploma'] = (df_sample['EDUCD'] >= 62).astype(int) & (df_sample['EDUCD'] < 65)
df_sample['some_college'] = (df_sample['EDUCD'] >= 65).astype(int) & (df_sample['EDUCD'] < 101)
df_sample['college_plus'] = (df_sample['EDUCD'] >= 101).astype(int)

# Summary by treatment and period
print("\n  Mean full-time employment by group and period:")
summary = df_sample.groupby(['treated', 'post']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'fulltime_n': len(x),
        'employed_mean': np.average(x['employed'], weights=x['PERWT']),
    })
).round(4)
print(summary)

# Simple DiD calculation
pre_treat = summary.loc[(1, 0), 'fulltime_mean']
post_treat = summary.loc[(1, 1), 'fulltime_mean']
pre_control = summary.loc[(0, 0), 'fulltime_mean']
post_control = summary.loc[(0, 1), 'fulltime_mean']

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n  Simple DiD estimate: {simple_did:.4f}")
print(f"    Treatment group change: {post_treat - pre_treat:.4f}")
print(f"    Control group change: {post_control - pre_control:.4f}")

# =============================================================================
# STEP 5: Main Regression Analysis
# =============================================================================
print("\n[STEP 5] Difference-in-differences regression...")

# Model 1: Basic DiD (unweighted)
print("\n  Model 1: Basic DiD (no weights, no covariates)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_sample).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model1.params['treated_post']:.4f}")
print(f"    Standard error: {model1.bse['treated_post']:.4f}")
print(f"    95% CI: [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model1.pvalues['treated_post']:.4f}")
print(f"    N: {int(model1.nobs):,}")

# Model 2: Weighted DiD
print("\n  Model 2: Weighted DiD (with PERWT)")
model2 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model2.params['treated_post']:.4f}")
print(f"    Standard error: {model2.bse['treated_post']:.4f}")
print(f"    95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model2.pvalues['treated_post']:.4f}")

# Model 3: With covariates
print("\n  Model 3: Weighted DiD with covariates")
df_sample['age_in_year'] = df_sample['YEAR'] - df_sample['BIRTHYR']
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age_in_year + C(EDUC)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model3.params['treated_post']:.4f}")
print(f"    Standard error: {model3.bse['treated_post']:.4f}")
print(f"    95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: With state and year fixed effects
print("\n  Model 4: Weighted DiD with state and year FE")
model4 = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model4.params['treated_post']:.4f}")
print(f"    Standard error: {model4.bse['treated_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model4.pvalues['treated_post']:.4f}")

# Model 5: Full model with covariates and FE
print("\n  Model 5: Full specification (covariates + state/year FE)")
model5 = smf.wls('fulltime ~ treated + treated_post + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                  data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model5.params['treated_post']:.4f}")
print(f"    Standard error: {model5.bse['treated_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model5.pvalues['treated_post']:.4f}")

# =============================================================================
# STEP 6: Robustness Checks
# =============================================================================
print("\n[STEP 6] Robustness checks...")

# 6.1 Employment outcome
print("\n  6.1 Effect on employment (EMPSTAT == 1)")
model_emp = smf.wls('employed ~ treated + treated_post + C(YEAR) + C(STATEFIP)',
                     data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model_emp.params['treated_post']:.4f}")
print(f"    Standard error: {model_emp.bse['treated_post']:.4f}")
print(f"    p-value: {model_emp.pvalues['treated_post']:.4f}")

# 6.2 Effect on hours (continuous)
print("\n  6.2 Effect on usual hours worked (continuous)")
model_hrs = smf.wls('UHRSWORK ~ treated + treated_post + C(YEAR) + C(STATEFIP)',
                     data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"    DiD coefficient: {model_hrs.params['treated_post']:.4f}")
print(f"    Standard error: {model_hrs.bse['treated_post']:.4f}")
print(f"    p-value: {model_hrs.pvalues['treated_post']:.4f}")

# 6.3 By gender
print("\n  6.3 Heterogeneity by gender")
df_male = df_sample[df_sample['female'] == 0]
df_female = df_sample[df_sample['female'] == 1]

model_male = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP)',
                      data=df_male, weights=df_male['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"    Male DiD coefficient: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")

model_female = smf.wls('fulltime ~ treated + treated_post + C(YEAR) + C(STATEFIP)',
                        data=df_female, weights=df_female['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"    Female DiD coefficient: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# 6.4 Pre-trends test
print("\n  6.4 Pre-trends test (event study)")
df_pre = df_sample[df_sample['YEAR'] <= 2011].copy()
df_pre['year_2007'] = (df_pre['YEAR'] == 2007).astype(int)
df_pre['year_2008'] = (df_pre['YEAR'] == 2008).astype(int)
df_pre['year_2009'] = (df_pre['YEAR'] == 2009).astype(int)
df_pre['year_2010'] = (df_pre['YEAR'] == 2010).astype(int)
df_pre['year_2011'] = (df_pre['YEAR'] == 2011).astype(int)

df_pre['treat_2007'] = df_pre['treated'] * df_pre['year_2007']
df_pre['treat_2008'] = df_pre['treated'] * df_pre['year_2008']
df_pre['treat_2009'] = df_pre['treated'] * df_pre['year_2009']
df_pre['treat_2010'] = df_pre['treated'] * df_pre['year_2010']
df_pre['treat_2011'] = df_pre['treated'] * df_pre['year_2011']

model_pretrend = smf.wls('fulltime ~ treated + C(YEAR) + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2011 + C(STATEFIP)',
                          data=df_pre, weights=df_pre['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})
print("    Pre-trend coefficients (relative to 2006):")
for year in ['2007', '2008', '2009', '2010', '2011']:
    coef = model_pretrend.params[f'treat_{year}']
    se = model_pretrend.bse[f'treat_{year}']
    pval = model_pretrend.pvalues[f'treat_{year}']
    print(f"      {year}: {coef:.4f} (SE: {se:.4f}, p: {pval:.3f})")

# 6.5 Event study (full)
print("\n  6.5 Event study (full specification)")
df_sample['year_2007'] = (df_sample['YEAR'] == 2007).astype(int)
df_sample['year_2008'] = (df_sample['YEAR'] == 2008).astype(int)
df_sample['year_2009'] = (df_sample['YEAR'] == 2009).astype(int)
df_sample['year_2010'] = (df_sample['YEAR'] == 2010).astype(int)
df_sample['year_2011'] = (df_sample['YEAR'] == 2011).astype(int)
df_sample['year_2013'] = (df_sample['YEAR'] == 2013).astype(int)
df_sample['year_2014'] = (df_sample['YEAR'] == 2014).astype(int)
df_sample['year_2015'] = (df_sample['YEAR'] == 2015).astype(int)
df_sample['year_2016'] = (df_sample['YEAR'] == 2016).astype(int)

for yr in ['2007', '2008', '2009', '2010', '2011', '2013', '2014', '2015', '2016']:
    df_sample[f'treat_{yr}'] = df_sample['treated'] * df_sample[f'year_{yr}']

model_event = smf.wls('fulltime ~ treated + C(YEAR) + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2011 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + C(STATEFIP)',
                       data=df_sample, weights=df_sample['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print("    Event study coefficients (relative to 2006):")
for year in ['2007', '2008', '2009', '2010', '2011', '2013', '2014', '2015', '2016']:
    coef = model_event.params[f'treat_{year}']
    se = model_event.bse[f'treat_{year}']
    pval = model_event.pvalues[f'treat_{year}']
    sig = '*' if pval < 0.05 else ''
    print(f"      {year}: {coef:7.4f} (SE: {se:.4f}, p: {pval:.3f}) {sig}")

# =============================================================================
# STEP 7: Summary Tables for Report
# =============================================================================
print("\n[STEP 7] Creating summary tables for report...")

# Table 1: Sample characteristics
print("\n  TABLE 1: Sample Characteristics")
print("-" * 60)

# Overall sample stats
overall_stats = {
    'N (observations)': len(df_sample),
    'N (weighted)': df_sample['PERWT'].sum(),
    'Full-time employment rate': np.average(df_sample['fulltime'], weights=df_sample['PERWT']),
    'Employment rate': np.average(df_sample['employed'], weights=df_sample['PERWT']),
    'Female (%)': np.average(df_sample['female'], weights=df_sample['PERWT']) * 100,
    'Married (%)': np.average(df_sample['married'], weights=df_sample['PERWT']) * 100,
    'Mean age in year': np.average(df_sample['age_in_year'], weights=df_sample['PERWT']),
}
for k, v in overall_stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v:,}")

# Table 2: DiD Summary
print("\n  TABLE 2: Difference-in-Differences Summary")
print("-" * 60)
print("                        Pre-DACA    Post-DACA    Difference")
print(f"  Treatment (26-30):    {pre_treat:.4f}      {post_treat:.4f}       {post_treat-pre_treat:+.4f}")
print(f"  Control (31-35):      {pre_control:.4f}      {post_control:.4f}       {post_control-pre_control:+.4f}")
print(f"  DiD Estimate:                                  {simple_did:+.4f}")

# Table 3: Main results
print("\n  TABLE 3: Main Regression Results")
print("-" * 60)
print("  Model                          Coef      SE       95% CI              p-value")
results_table = [
    ('Basic DiD', model1.params['treated_post'], model1.bse['treated_post'],
     model1.conf_int().loc['treated_post', 0], model1.conf_int().loc['treated_post', 1],
     model1.pvalues['treated_post']),
    ('Weighted', model2.params['treated_post'], model2.bse['treated_post'],
     model2.conf_int().loc['treated_post', 0], model2.conf_int().loc['treated_post', 1],
     model2.pvalues['treated_post']),
    ('+ Covariates', model3.params['treated_post'], model3.bse['treated_post'],
     model3.conf_int().loc['treated_post', 0], model3.conf_int().loc['treated_post', 1],
     model3.pvalues['treated_post']),
    ('+ State/Year FE', model4.params['treated_post'], model4.bse['treated_post'],
     model4.conf_int().loc['treated_post', 0], model4.conf_int().loc['treated_post', 1],
     model4.pvalues['treated_post']),
    ('Full Model', model5.params['treated_post'], model5.bse['treated_post'],
     model5.conf_int().loc['treated_post', 0], model5.conf_int().loc['treated_post', 1],
     model5.pvalues['treated_post']),
]
for name, coef, se, ci_lo, ci_hi, pval in results_table:
    print(f"  {name:25s} {coef:7.4f}  {se:7.4f}  [{ci_lo:7.4f}, {ci_hi:7.4f}]  {pval:.4f}")

# =============================================================================
# STEP 8: Save Results
# =============================================================================
print("\n[STEP 8] Saving results...")

# Save key results to CSV
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'Weighted', 'With Covariates', 'State/Year FE', 'Full Model'],
    'Coefficient': [model1.params['treated_post'], model2.params['treated_post'],
                   model3.params['treated_post'], model4.params['treated_post'],
                   model5.params['treated_post']],
    'SE': [model1.bse['treated_post'], model2.bse['treated_post'],
           model3.bse['treated_post'], model4.bse['treated_post'],
           model5.bse['treated_post']],
    'CI_lower': [model1.conf_int().loc['treated_post', 0], model2.conf_int().loc['treated_post', 0],
                 model3.conf_int().loc['treated_post', 0], model4.conf_int().loc['treated_post', 0],
                 model5.conf_int().loc['treated_post', 0]],
    'CI_upper': [model1.conf_int().loc['treated_post', 1], model2.conf_int().loc['treated_post', 1],
                 model3.conf_int().loc['treated_post', 1], model4.conf_int().loc['treated_post', 1],
                 model5.conf_int().loc['treated_post', 1]],
    'p_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
})
results_df.to_csv('results_main.csv', index=False)

# Save event study results
event_results = []
for year in ['2007', '2008', '2009', '2010', '2011', '2013', '2014', '2015', '2016']:
    event_results.append({
        'Year': int(year),
        'Coefficient': model_event.params[f'treat_{year}'],
        'SE': model_event.bse[f'treat_{year}'],
        'CI_lower': model_event.conf_int().loc[f'treat_{year}', 0],
        'CI_upper': model_event.conf_int().loc[f'treat_{year}', 1],
        'p_value': model_event.pvalues[f'treat_{year}']
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('results_event_study.csv', index=False)

# Save descriptive stats
desc_stats = df_sample.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'UHRSWORK': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('results_descriptive.csv')

print("  Results saved to:")
print("    - results_main.csv")
print("    - results_event_study.csv")
print("    - results_descriptive.csv")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"\nPreferred Estimate (Full Model with State/Year FE):")
print(f"  Effect size: {model4.params['treated_post']:.4f}")
print(f"  Standard error: {model4.bse['treated_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  p-value: {model4.pvalues['treated_post']:.4f}")
print(f"  Sample size: {int(model4.nobs):,}")

print(f"\nInterpretation:")
coef = model4.params['treated_post']
if coef > 0:
    print(f"  DACA eligibility is associated with a {coef*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(coef)*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model4.pvalues['treated_post'] < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)

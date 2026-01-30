"""
DACA Replication Study - Final Analysis Script
Participant ID: 94

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.

Memory-efficient implementation using chunked reading.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
import json
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY - FINAL ANALYSIS")
print("=" * 70)

# ============================================================================
# 1. LOAD AND FILTER DATA (CHUNKED)
# ============================================================================
print("\n1. Loading and filtering data (chunked for memory efficiency)...")

# Only keep columns we need
cols_needed = ['YEAR', 'SERIAL', 'PERWT', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR',
               'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUCD', 'MARST',
               'UHRSWORK']

chunks = []
total_rows = 0
filtered_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=500000):
    total_rows += len(chunk)

    # Filter to target population
    # Hispanic-Mexican, born in Mexico, non-citizen
    filtered = chunk[(chunk['HISPAN'] == 1) &
                     (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3) &
                     (chunk['YRIMMIG'] > 0) &
                     (chunk['AGE'] >= 16) &
                     (chunk['AGE'] <= 40)].copy()

    if len(filtered) > 0:
        chunks.append(filtered)
        filtered_rows += len(filtered)

df_mex = pd.concat(chunks, ignore_index=True)
print(f"   Total rows in data: {total_rows:,}")
print(f"   After filtering to target population: {len(df_mex):,}")

# ============================================================================
# 2. CONSTRUCT VARIABLES
# ============================================================================
print("\n2. Constructing analysis variables...")

# Age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Filter out impossible values
df_mex = df_mex[(df_mex['age_at_arrival'] >= 0) & (df_mex['age_at_arrival'] <= 40)].copy()

# DACA eligibility criteria
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)
df_mex['under_31_june2012'] = ((df_mex['BIRTHYR'] >= 1982) |
                               ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))).astype(int)
df_mex['in_us_since_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# Full DACA eligibility
df_mex['daca_eligible'] = ((df_mex['arrived_before_16'] == 1) &
                            (df_mex['under_31_june2012'] == 1) &
                            (df_mex['in_us_since_2007'] == 1)).astype(int)

print(f"   Total observations: {len(df_mex):,}")
print(f"   DACA eligible: {df_mex['daca_eligible'].sum():,} ({df_mex['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# 3. CREATE ANALYSIS SAMPLES
# ============================================================================
print("\n3. Creating analysis samples...")

# SAMPLE A: Narrow bandwidth around arrival age 16 cutoff
# Treatment: arrived at ages 10-15 (just under cutoff)
# Control: arrived at ages 16-21 (just over cutoff)
# Both must meet age criteria (under 31 in 2012, in US since 2007)

df_narrow = df_mex[(df_mex['under_31_june2012'] == 1) &
                   (df_mex['in_us_since_2007'] == 1) &
                   (df_mex['age_at_arrival'] >= 10) &
                   (df_mex['age_at_arrival'] <= 21)].copy()

df_narrow['treated'] = (df_narrow['age_at_arrival'] < 16).astype(int)
df_narrow = df_narrow[df_narrow['YEAR'] != 2012].copy()  # Exclude transition year
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']

# Control variables
df_narrow['age'] = df_narrow['AGE']
df_narrow['age_sq'] = df_narrow['age'] ** 2
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'].isin([1, 2])).astype(int)
df_narrow['state'] = df_narrow['STATEFIP'].astype(str)
df_narrow['year_str'] = df_narrow['YEAR'].astype(str)
df_narrow['educ_hs'] = (df_narrow['EDUCD'] >= 60).astype(int)
df_narrow['educ_college'] = (df_narrow['EDUCD'] >= 100).astype(int)

print(f"\nNarrow Bandwidth Sample (arrival age 10-21):")
print(f"   Treatment (arrived 10-15): {(df_narrow['treated']==1).sum():,}")
print(f"   Control (arrived 16-21): {(df_narrow['treated']==0).sum():,}")
print(f"   Pre-period (2006-2011): {(df_narrow['post']==0).sum():,}")
print(f"   Post-period (2013-2016): {(df_narrow['post']==1).sum():,}")

# ============================================================================
# 4. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 70)
print("4. SUMMARY STATISTICS")
print("=" * 70)

# Pre-period summary by treatment status
print("\nPre-Period (2006-2011) Characteristics:")
pre = df_narrow[df_narrow['post'] == 0]

summary_stats = []
for t, label in [(1, 'Treatment (arr 10-15)'), (0, 'Control (arr 16-21)')]:
    s = pre[pre['treated'] == t]
    stats = {
        'Group': label,
        'N': len(s),
        'Age': s['age'].mean(),
        'Female (%)': s['female'].mean() * 100,
        'Married (%)': s['married'].mean() * 100,
        'HS+ (%)': s['educ_hs'].mean() * 100,
        'Full-time (%)': s['fulltime'].mean() * 100,
        'Arr age': s['age_at_arrival'].mean()
    }
    summary_stats.append(stats)
    print(f"\n{label}:")
    print(f"   N = {stats['N']:,}")
    print(f"   Mean age = {stats['Age']:.1f}")
    print(f"   Female = {stats['Female (%)']:.1f}%")
    print(f"   Married = {stats['Married (%)']:.1f}%")
    print(f"   HS education+ = {stats['HS+ (%)']:.1f}%")
    print(f"   Full-time employed = {stats['Full-time (%)']:.1f}%")
    print(f"   Mean arrival age = {stats['Arr age']:.1f}")

# ============================================================================
# 5. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("5. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Model 1: Basic DiD
print("\nModel 1: Basic DiD (no controls)")
m1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_narrow).fit(
    cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
print(f"   DiD Estimate: {m1.params['treated_post']:.4f} (SE = {m1.bse['treated_post']:.4f})")
print(f"   p-value: {m1.pvalues['treated_post']:.4f}, N = {m1.nobs:,.0f}")

# Model 2: With demographic controls
print("\nModel 2: DiD with demographic controls")
m2 = smf.ols('fulltime ~ treated + post + treated_post + age + age_sq + female + married + educ_hs + educ_college',
             data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
print(f"   DiD Estimate: {m2.params['treated_post']:.4f} (SE = {m2.bse['treated_post']:.4f})")
print(f"   p-value: {m2.pvalues['treated_post']:.4f}, N = {m2.nobs:,.0f}")

# Model 3: With state FE
print("\nModel 3: DiD with demographics + state FE")
m3 = smf.ols('fulltime ~ treated + post + treated_post + age + age_sq + female + married + educ_hs + educ_college + C(state)',
             data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
print(f"   DiD Estimate: {m3.params['treated_post']:.4f} (SE = {m3.bse['treated_post']:.4f})")
print(f"   p-value: {m3.pvalues['treated_post']:.4f}, N = {m3.nobs:,.0f}")

# Model 4: Full specification with year FE (PREFERRED)
print("\nModel 4: Full specification with year FE (PREFERRED)")
m4 = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + educ_hs + educ_college + C(state) + C(year_str)',
             data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})
print(f"   DiD Estimate: {m4.params['treated_post']:.4f} (SE = {m4.bse['treated_post']:.4f})")
print(f"   95% CI: [{m4.conf_int().loc['treated_post', 0]:.4f}, {m4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"   p-value: {m4.pvalues['treated_post']:.4f}")
print(f"   N = {m4.nobs:,.0f}, R² = {m4.rsquared:.4f}")

# Store preferred model
preferred_model = m4
preferred_effect = m4.params['treated_post']
preferred_se = m4.bse['treated_post']
preferred_ci = m4.conf_int().loc['treated_post']
preferred_pval = m4.pvalues['treated_post']
preferred_n = int(m4.nobs)

# ============================================================================
# 6. EVENT STUDY
# ============================================================================
print("\n" + "=" * 70)
print("6. EVENT STUDY (Reference: 2011)")
print("=" * 70)

# Create year-by-treatment interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_narrow[f'treat_yr_{yr}'] = (df_narrow['treated'] * (df_narrow['YEAR'] == yr)).astype(int)

event_vars = ' + '.join([f'treat_yr_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
event_formula = f'fulltime ~ treated + age + age_sq + female + married + educ_hs + educ_college + C(state) + C(year_str) + {event_vars}'
m_event = smf.ols(event_formula, data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['SERIAL']})

print("\n   Year    Coef      SE       p-val")
print("   " + "-" * 40)
event_results = []
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    c = m_event.params[f'treat_yr_{yr}']
    s = m_event.bse[f'treat_yr_{yr}']
    p = m_event.pvalues[f'treat_yr_{yr}']
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))
    print(f"   {yr}    {c:7.4f}   {s:6.4f}   {p:.4f}  {sig}")
    event_results.append({'year': yr, 'coef': float(c), 'se': float(s), 'pval': float(p)})

# ============================================================================
# 7. ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 70)
print("7. ROBUSTNESS CHECKS")
print("=" * 70)

# 7.1 Different bandwidths
print("\n7.1 Different Arrival Age Bandwidths:")
bandwidth_results = []
for bw in [(13, 18), (12, 19), (10, 21), (8, 23), (5, 26)]:
    low, high = bw
    df_bw = df_mex[(df_mex['under_31_june2012'] == 1) &
                   (df_mex['in_us_since_2007'] == 1) &
                   (df_mex['age_at_arrival'] >= low) &
                   (df_mex['age_at_arrival'] <= high) &
                   (df_mex['YEAR'] != 2012)].copy()
    df_bw['treated'] = (df_bw['age_at_arrival'] < 16).astype(int)
    df_bw['post'] = (df_bw['YEAR'] >= 2013).astype(int)
    df_bw['fulltime'] = (df_bw['UHRSWORK'] >= 35).astype(int)
    df_bw['treated_post'] = df_bw['treated'] * df_bw['post']
    df_bw['age'] = df_bw['AGE']
    df_bw['age_sq'] = df_bw['age'] ** 2
    df_bw['female'] = (df_bw['SEX'] == 2).astype(int)
    df_bw['married'] = (df_bw['MARST'].isin([1, 2])).astype(int)
    df_bw['state'] = df_bw['STATEFIP'].astype(str)
    df_bw['year_str'] = df_bw['YEAR'].astype(str)
    df_bw['educ_hs'] = (df_bw['EDUCD'] >= 60).astype(int)
    df_bw['educ_college'] = (df_bw['EDUCD'] >= 100).astype(int)

    m_bw = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + educ_hs + educ_college + C(state) + C(year_str)',
                   data=df_bw).fit(cov_type='cluster', cov_kwds={'groups': df_bw['SERIAL']})
    print(f"   [{low:2d}, {high:2d}]: Effect = {m_bw.params['treated_post']:.4f} (SE = {m_bw.bse['treated_post']:.4f}), N = {m_bw.nobs:,.0f}")
    bandwidth_results.append({'low': low, 'high': high, 'effect': float(m_bw.params['treated_post']),
                             'se': float(m_bw.bse['treated_post']), 'n': int(m_bw.nobs)})

# 7.2 By gender
print("\n7.2 By Gender:")
gender_results = []
for sex, label in [(1, 'Male'), (2, 'Female')]:
    subset = df_narrow[df_narrow['SEX'] == sex]
    m_g = smf.ols('fulltime ~ treated + treated_post + age + age_sq + married + educ_hs + educ_college + C(state) + C(year_str)',
                  data=subset).fit(cov_type='cluster', cov_kwds={'groups': subset['SERIAL']})
    print(f"   {label}: Effect = {m_g.params['treated_post']:.4f} (SE = {m_g.bse['treated_post']:.4f}), N = {m_g.nobs:,.0f}")
    gender_results.append({'gender': label, 'effect': float(m_g.params['treated_post']),
                          'se': float(m_g.bse['treated_post']), 'n': int(m_g.nobs)})

# 7.3 Placebo test with pre-period
print("\n7.3 Placebo Test (Pre-period: 2006-2008 vs 2009-2011):")
pre_only = df_narrow[df_narrow['post'] == 0].copy()
pre_only['placebo_post'] = (pre_only['YEAR'] >= 2009).astype(int)
pre_only['treated_placebo'] = pre_only['treated'] * pre_only['placebo_post']
m_placebo = smf.ols('fulltime ~ treated + placebo_post + treated_placebo + age + age_sq + female + married + educ_hs + educ_college + C(state)',
                    data=pre_only).fit(cov_type='cluster', cov_kwds={'groups': pre_only['SERIAL']})
print(f"   Placebo DiD: {m_placebo.params['treated_placebo']:.4f} (SE = {m_placebo.bse['treated_placebo']:.4f})")
print(f"   p-value: {m_placebo.pvalues['treated_placebo']:.4f}")
placebo_result = {'effect': float(m_placebo.params['treated_placebo']),
                  'se': float(m_placebo.bse['treated_placebo']),
                  'pval': float(m_placebo.pvalues['treated_placebo'])}

# 7.4 Alternative control: cohort just too old for DACA
print("\n7.4 Alternative Control: Slightly Older Cohort (born 1977-1981):")
df_alt = df_mex[(df_mex['arrived_before_16'] == 1) &
                (df_mex['in_us_since_2007'] == 1) &
                (df_mex['BIRTHYR'].between(1977, 1995)) &
                (df_mex['YEAR'] != 2012)].copy()
df_alt['treated'] = (df_alt['BIRTHYR'] >= 1982).astype(int)  # DACA eligible vs too old
df_alt['post'] = (df_alt['YEAR'] >= 2013).astype(int)
df_alt['fulltime'] = (df_alt['UHRSWORK'] >= 35).astype(int)
df_alt['treated_post'] = df_alt['treated'] * df_alt['post']
df_alt['age'] = df_alt['AGE']
df_alt['age_sq'] = df_alt['age'] ** 2
df_alt['female'] = (df_alt['SEX'] == 2).astype(int)
df_alt['married'] = (df_alt['MARST'].isin([1, 2])).astype(int)
df_alt['state'] = df_alt['STATEFIP'].astype(str)
df_alt['year_str'] = df_alt['YEAR'].astype(str)
df_alt['educ_hs'] = (df_alt['EDUCD'] >= 60).astype(int)
df_alt['educ_college'] = (df_alt['EDUCD'] >= 100).astype(int)

m_alt = smf.ols('fulltime ~ treated + treated_post + age + age_sq + female + married + educ_hs + educ_college + C(state) + C(year_str)',
                data=df_alt).fit(cov_type='cluster', cov_kwds={'groups': df_alt['SERIAL']})
print(f"   Cohort DiD: {m_alt.params['treated_post']:.4f} (SE = {m_alt.bse['treated_post']:.4f})")
print(f"   N = {m_alt.nobs:,.0f}")
alt_control_result = {'effect': float(m_alt.params['treated_post']),
                      'se': float(m_alt.bse['treated_post']),
                      'n': int(m_alt.nobs)}

# ============================================================================
# 8. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("8. FINAL SUMMARY")
print("=" * 70)

print(f"""
PREFERRED SPECIFICATION: Narrow Bandwidth DiD

Identification Strategy:
- Compare Mexican-born non-citizens who arrived at ages 10-15 (DACA eligible)
  to those who arrived at ages 16-21 (just above the age-at-arrival cutoff)
- Both groups meet other DACA criteria (under 31 in 2012, in US since 2007)
- Difference-in-differences: before (2006-2011) vs after (2013-2016) DACA

Main Result:
- Effect Size: {preferred_effect:.4f} ({preferred_effect*100:.2f} percentage points)
- Standard Error: {preferred_se:.4f}
- 95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]
- p-value: {preferred_pval:.6f}
- Sample Size: {preferred_n:,}

Interpretation:
DACA eligibility is associated with a {preferred_effect*100:.2f} percentage point
{'increase' if preferred_effect > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if preferred_pval < 0.05 else 'not statistically significant'}
at the 5% level.
""")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("Saving results...")

results = {
    'preferred_estimate': {
        'specification': 'Narrow Bandwidth DiD (arrival age 10-21)',
        'effect_size': float(preferred_effect),
        'std_error': float(preferred_se),
        'ci_lower': float(preferred_ci[0]),
        'ci_upper': float(preferred_ci[1]),
        'p_value': float(preferred_pval),
        'n_obs': preferred_n,
        'r_squared': float(m4.rsquared)
    },
    'all_models': {
        'model1_basic': {'effect': float(m1.params['treated_post']), 'se': float(m1.bse['treated_post']), 'n': int(m1.nobs)},
        'model2_demographics': {'effect': float(m2.params['treated_post']), 'se': float(m2.bse['treated_post']), 'n': int(m2.nobs)},
        'model3_state_fe': {'effect': float(m3.params['treated_post']), 'se': float(m3.bse['treated_post']), 'n': int(m3.nobs)},
        'model4_full': {'effect': float(m4.params['treated_post']), 'se': float(m4.bse['treated_post']), 'n': int(m4.nobs)}
    },
    'event_study': event_results,
    'robustness': {
        'bandwidths': bandwidth_results,
        'by_gender': gender_results,
        'placebo': placebo_result,
        'alt_control': alt_control_result
    },
    'summary_stats': summary_stats
}

with open('results_final.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results_final.json")
print("\nAnalysis complete.")

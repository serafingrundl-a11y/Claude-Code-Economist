#!/usr/bin/env python3
"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment probability?

DACA Eligibility Criteria:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status
5. Not a citizen (we treat non-naturalized, non-citizen as undocumented for DACA purposes)

Identification Strategy: Difference-in-Differences
- Treatment group: DACA-eligible non-citizens
- Control group: Non-eligible non-citizens (similar background but don't meet age/arrival criteria)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded as transition year)

Author: Replication Analysis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
import gc

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
data_path = "data/data.csv"

# Only read columns we need to save memory
needed_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK']

print("Reading data in chunks (filtering as we go)...")

# Process in chunks to avoid memory issues
chunks = []
chunk_size = 1000000  # 1 million rows at a time

for i, chunk in enumerate(pd.read_csv(data_path, usecols=needed_cols, chunksize=chunk_size)):
    # Filter immediately to Hispanic-Mexican, Mexico-born
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)
    print(f"  Processed chunk {i+1}, kept {len(chunk_filtered):,} rows", flush=True)
    del chunk
    gc.collect()

print("Combining chunks...")
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"Total observations after initial filter: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Save basic stats
initial_n = len(df)

print("\n2. ADDITIONAL SAMPLE RESTRICTIONS...")

# Step 1: Restrict to working-age population (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"After restricting to ages 16-64: {len(df):,}")

# Step 2: Exclude 2012 (transition year - DACA announced mid-year)
df_analysis = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

del df
gc.collect()

# Save the sample
sample_n = len(df_analysis)

print("\n3. CREATING VARIABLES...")

# Create post-DACA indicator
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"Post-DACA years (2013-2016): {df_analysis[df_analysis['post']==1]['YEAR'].unique()}")
print(f"Pre-DACA years (2006-2011): {df_analysis[df_analysis['post']==0]['YEAR'].unique()}")

# Create outcome: full-time employment (UHRSWORK >= 35)
# UHRSWORK = Usual hours worked per week
# Full-time defined as 35+ hours
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Determine DACA eligibility
# Key criteria for DACA eligibility:
# 1. Non-citizen (CITIZEN == 3: Not a citizen)
# 2. Arrived before 16th birthday (need to calculate age at arrival)
# 3. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 4. In US since June 15, 2007 (immigrated 2007 or earlier)
# 5. Present in US on June 15, 2012 (we assume this for the ACS sample)

# Calculate age at immigration
# Age at immigration = YRIMMIG - BIRTHYR
df_analysis['age_at_immig'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']

# Handle cases where YRIMMIG is 0 (N/A - US born) or missing
df_analysis.loc[df_analysis['YRIMMIG'] == 0, 'age_at_immig'] = np.nan

# Non-citizen indicator
# CITIZEN: 0=N/A, 1=Born abroad of American parents, 2=Naturalized citizen, 3=Not a citizen
df_analysis['noncitizen'] = (df_analysis['CITIZEN'] == 3).astype(int)

# DACA eligibility criteria
# 1. Arrived before age 16
df_analysis['arrived_before_16'] = (df_analysis['age_at_immig'] < 16) & (df_analysis['age_at_immig'] >= 0)

# 2. Under 31 as of June 15, 2012 (born June 15, 1981 or later)
# Since we only have birth year and quarter, we use a conservative approach:
# Born 1982 or later definitely qualifies
# Born 1981: depends on birth quarter
#   BIRTHQTR 1 (Jan-Mar): would be 31 by June 2012 - NOT eligible
#   BIRTHQTR 2 (Apr-Jun): might be 31 by June 2012 - borderline
#   BIRTHQTR 3 (Jul-Sep): would be 30 by June 2012 - eligible
#   BIRTHQTR 4 (Oct-Dec): would be 30 by June 2012 - eligible
df_analysis['under_31_june2012'] = (
    (df_analysis['BIRTHYR'] >= 1982) |
    ((df_analysis['BIRTHYR'] == 1981) & (df_analysis['BIRTHQTR'].isin([3, 4])))
)

# 3. In US since June 15, 2007 (immigrated 2007 or earlier)
df_analysis['in_us_since_2007'] = (df_analysis['YRIMMIG'] <= 2007) & (df_analysis['YRIMMIG'] > 0)

# DACA eligible = All criteria met AND non-citizen
df_analysis['daca_eligible'] = (
    df_analysis['noncitizen'] &
    df_analysis['arrived_before_16'] &
    df_analysis['under_31_june2012'] &
    df_analysis['in_us_since_2007']
).astype(int)

print(f"\nNon-citizens in sample: {df_analysis['noncitizen'].sum():,} ({100*df_analysis['noncitizen'].mean():.1f}%)")
print(f"DACA eligible: {df_analysis['daca_eligible'].sum():,} ({100*df_analysis['daca_eligible'].mean():.1f}%)")

# Create treatment indicator: DACA-eligible non-citizens
df_analysis['treated'] = df_analysis['daca_eligible']

# Create DiD interaction
df_analysis['did'] = df_analysis['treated'] * df_analysis['post']

print("\n4. SAMPLE SUMMARY STATISTICS...")

# Summary by treatment status
print("\n--- Sample sizes by treatment and period ---")
summary_table = pd.crosstab(df_analysis['treated'], df_analysis['post'],
                            values=df_analysis['PERWT'], aggfunc='sum', margins=True)
summary_table.columns = ['Pre-DACA', 'Post-DACA', 'Total']
summary_table.index = ['Non-eligible', 'DACA-eligible', 'Total']
print(summary_table.round(0))

# Unweighted counts
print("\n--- Unweighted sample sizes ---")
unweighted = pd.crosstab(df_analysis['treated'], df_analysis['post'], margins=True)
unweighted.columns = ['Pre-DACA', 'Post-DACA', 'Total']
unweighted.index = ['Non-eligible', 'DACA-eligible', 'Total']
print(unweighted)

# Outcome means by group and period
print("\n--- Full-time employment rates by group and period ---")
outcome_means = df_analysis.groupby(['treated', 'post']).apply(
    lambda x: pd.Series({
        'mean_fulltime': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).round(4)
print(outcome_means)

# Calculate raw DiD estimate
pre_treated = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]
post_treated = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]
pre_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]
post_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]

mean_pre_t = np.average(pre_treated['fulltime'], weights=pre_treated['PERWT'])
mean_post_t = np.average(post_treated['fulltime'], weights=post_treated['PERWT'])
mean_pre_c = np.average(pre_control['fulltime'], weights=pre_control['PERWT'])
mean_post_c = np.average(post_control['fulltime'], weights=post_control['PERWT'])

raw_did = (mean_post_t - mean_pre_t) - (mean_post_c - mean_pre_c)

print(f"\n--- Raw Difference-in-Differences ---")
print(f"Treated Pre:  {mean_pre_t:.4f}")
print(f"Treated Post: {mean_post_t:.4f}")
print(f"Control Pre:  {mean_pre_c:.4f}")
print(f"Control Post: {mean_post_c:.4f}")
print(f"DiD Estimate: {raw_did:.4f} ({raw_did*100:.2f} percentage points)")

print("\n5. REGRESSION ANALYSIS...")

# Prepare control variables
# Age and age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_somecoll'] = (df_analysis['EDUC'].isin([7, 8, 9])).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

# Gender
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status (married vs not)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.wls('fulltime ~ treated + post + did + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with year fixed effects ---")
# Create year dummies (omit one year as reference)
year_dummies = pd.get_dummies(df_analysis['YEAR'], prefix='year', drop_first=True)
df_analysis = pd.concat([df_analysis, year_dummies], axis=1)

year_cols = [c for c in df_analysis.columns if c.startswith('year_')]
year_formula = ' + '.join(year_cols)

model3 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {year_formula}',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})")
print(f"t-statistic: {model3.tvalues['did']:.3f}, p-value: {model3.pvalues['did']:.4f}")

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with state fixed effects ---")
state_dummies = pd.get_dummies(df_analysis['STATEFIP'], prefix='state', drop_first=True)
df_analysis = pd.concat([df_analysis, state_dummies], axis=1)

state_cols = [c for c in df_analysis.columns if c.startswith('state_')]
state_formula = ' + '.join(state_cols)

model4 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {year_formula} + {state_formula}',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model4.params['did']:.4f} (SE: {model4.bse['did']:.4f})")
print(f"t-statistic: {model4.tvalues['did']:.3f}, p-value: {model4.pvalues['did']:.4f}")

# Store results
main_result = {
    'coefficient': model4.params['did'],
    'se': model4.bse['did'],
    'tstat': model4.tvalues['did'],
    'pvalue': model4.pvalues['did'],
    'n': model4.nobs,
    'ci_lower': model4.params['did'] - 1.96 * model4.bse['did'],
    'ci_upper': model4.params['did'] + 1.96 * model4.bse['did']
}

print("\n" + "="*80)
print("MAIN RESULT (Model 4 - Full specification)")
print("="*80)
print(f"Effect of DACA eligibility on full-time employment probability:")
print(f"  Coefficient: {main_result['coefficient']:.4f}")
print(f"  Standard Error: {main_result['se']:.4f}")
print(f"  95% CI: [{main_result['ci_lower']:.4f}, {main_result['ci_upper']:.4f}]")
print(f"  t-statistic: {main_result['tstat']:.3f}")
print(f"  p-value: {main_result['pvalue']:.4f}")
print(f"  Sample size: {int(main_result['n']):,}")

print("\n6. ROBUSTNESS CHECKS...")

# Robustness 1: Restrict to non-citizens only
print("\n--- Robustness 1: Non-citizens only ---")
df_noncit = df_analysis[df_analysis['noncitizen'] == 1].copy()
print(f"Sample size (non-citizens only): {len(df_noncit):,}")

model_r1 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {year_formula}',
                    data=df_noncit,
                    weights=df_noncit['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r1.params['did']:.4f} (SE: {model_r1.bse['did']:.4f})")

# Robustness 2: Different age range (18-35) - focus on young workers
print("\n--- Robustness 2: Young workers (18-35) ---")
df_young = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 35)].copy()
print(f"Sample size (ages 18-35): {len(df_young):,}")

model_r2 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {year_formula}',
                    data=df_young,
                    weights=df_young['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r2.params['did']:.4f} (SE: {model_r2.bse['did']:.4f})")

# Robustness 3: Males only
print("\n--- Robustness 3: Males only ---")
df_male = df_analysis[df_analysis['female'] == 0].copy()
print(f"Sample size (males): {len(df_male):,}")

model_r3 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + married + educ_hs + educ_somecoll + educ_college + {year_formula}',
                    data=df_male,
                    weights=df_male['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r3.params['did']:.4f} (SE: {model_r3.bse['did']:.4f})")

# Robustness 4: Females only
print("\n--- Robustness 4: Females only ---")
df_female = df_analysis[df_analysis['female'] == 1].copy()
print(f"Sample size (females): {len(df_female):,}")

model_r4 = smf.wls(f'fulltime ~ treated + did + AGE + age_sq + married + educ_hs + educ_somecoll + educ_college + {year_formula}',
                    data=df_female,
                    weights=df_female['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r4.params['did']:.4f} (SE: {model_r4.bse['did']:.4f})")

# Robustness 5: Placebo test (pseudo-treatment in 2009)
print("\n--- Robustness 5: Placebo test (pseudo-treatment in 2009) ---")
df_placebo = df_analysis[df_analysis['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011])].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['did_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

# Create year dummies for placebo
placebo_year_dummies = pd.get_dummies(df_placebo['YEAR'], prefix='pyear', drop_first=True)
df_placebo = pd.concat([df_placebo, placebo_year_dummies], axis=1)
placebo_year_cols = [c for c in df_placebo.columns if c.startswith('pyear_')]
placebo_year_formula = ' + '.join(placebo_year_cols)

model_placebo = smf.wls(f'fulltime ~ treated + did_placebo + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {placebo_year_formula}',
                         data=df_placebo,
                         weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {model_placebo.params['did_placebo']:.4f} (SE: {model_placebo.bse['did_placebo']:.4f})")
print(f"p-value: {model_placebo.pvalues['did_placebo']:.4f}")

print("\n7. EVENT STUDY ANALYSIS...")

# Event study: Year-by-year effects
# Create interaction terms for each year (reference: 2011)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for y in years:
    df_analysis[f'treat_x_{y}'] = (df_analysis['treated'] * (df_analysis['YEAR'] == y)).astype(int)

treat_year_cols = [f'treat_x_{y}' for y in years]
treat_year_formula = ' + '.join(treat_year_cols)

model_event = smf.wls(f'fulltime ~ treated + AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + {year_formula} + {treat_year_formula}',
                       data=df_analysis,
                       weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("Year-by-year treatment effects (reference: 2011):")
for y in years:
    coef = model_event.params[f'treat_x_{y}']
    se = model_event.bse[f'treat_x_{y}']
    print(f"  {y}: {coef:.4f} (SE: {se:.4f})")

# Store event study results
event_study_results = pd.DataFrame({
    'year': years,
    'coef': [model_event.params[f'treat_x_{y}'] for y in years],
    'se': [model_event.bse[f'treat_x_{y}'] for y in years]
})
event_study_results['ci_lower'] = event_study_results['coef'] - 1.96 * event_study_results['se']
event_study_results['ci_upper'] = event_study_results['coef'] + 1.96 * event_study_results['se']

print("\n8. SUMMARY STATISTICS TABLE...")

# Create detailed summary statistics
def weighted_stats(df, var, weight):
    """Calculate weighted mean and std"""
    mean = np.average(df[var], weights=df[weight])
    variance = np.average((df[var] - mean)**2, weights=df[weight])
    std = np.sqrt(variance)
    return mean, std

# Summary stats by treatment group
vars_to_summarize = ['fulltime', 'AGE', 'female', 'married', 'educ_lesshs', 'educ_hs', 'educ_somecoll', 'educ_college']
var_labels = ['Full-time employed', 'Age', 'Female', 'Married', 'Less than HS', 'High school', 'Some college', 'College+']

summary_stats = []
for var, label in zip(vars_to_summarize, var_labels):
    # Treatment group
    t_mean, t_std = weighted_stats(df_analysis[df_analysis['treated']==1], var, 'PERWT')
    # Control group
    c_mean, c_std = weighted_stats(df_analysis[df_analysis['treated']==0], var, 'PERWT')
    summary_stats.append({
        'Variable': label,
        'Treated Mean': t_mean,
        'Treated SD': t_std,
        'Control Mean': c_mean,
        'Control SD': c_std
    })

summary_df = pd.DataFrame(summary_stats)
print(summary_df.to_string(index=False))

print("\n9. SAVING RESULTS...")

# Save all results to a pickle file for report generation
results = {
    'main_result': main_result,
    'model1': {'coef': model1.params['did'], 'se': model1.bse['did'], 'pval': model1.pvalues['did'], 'n': model1.nobs},
    'model2': {'coef': model2.params['did'], 'se': model2.bse['did'], 'pval': model2.pvalues['did'], 'n': model2.nobs},
    'model3': {'coef': model3.params['did'], 'se': model3.bse['did'], 'pval': model3.pvalues['did'], 'n': model3.nobs},
    'model4': {'coef': model4.params['did'], 'se': model4.bse['did'], 'pval': model4.pvalues['did'], 'n': model4.nobs},
    'robustness': {
        'noncit_only': {'coef': model_r1.params['did'], 'se': model_r1.bse['did'], 'n': model_r1.nobs},
        'young_workers': {'coef': model_r2.params['did'], 'se': model_r2.bse['did'], 'n': model_r2.nobs},
        'males': {'coef': model_r3.params['did'], 'se': model_r3.bse['did'], 'n': model_r3.nobs},
        'females': {'coef': model_r4.params['did'], 'se': model_r4.bse['did'], 'n': model_r4.nobs},
        'placebo': {'coef': model_placebo.params['did_placebo'], 'se': model_placebo.bse['did_placebo'],
                    'pval': model_placebo.pvalues['did_placebo'], 'n': model_placebo.nobs}
    },
    'event_study': event_study_results,
    'summary_stats': summary_df,
    'raw_did': raw_did,
    'sample_sizes': {
        'initial': initial_n,
        'analysis': sample_n,
        'treated_pre': len(pre_treated),
        'treated_post': len(post_treated),
        'control_pre': len(pre_control),
        'control_post': len(post_control)
    },
    'outcome_means': {
        'treated_pre': mean_pre_t,
        'treated_post': mean_post_t,
        'control_pre': mean_pre_c,
        'control_post': mean_post_c
    }
}

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results saved to results.pkl")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPreferred Estimate (Model 4 with year and state FE):")
print(f"  Effect: {main_result['coefficient']:.4f} ({main_result['coefficient']*100:.2f} percentage points)")
print(f"  SE: {main_result['se']:.4f}")
print(f"  95% CI: [{main_result['ci_lower']:.4f}, {main_result['ci_upper']:.4f}]")
print(f"  p-value: {main_result['pvalue']:.4f}")
print(f"  N: {int(main_result['n']):,}")

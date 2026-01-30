"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US

Author: [Anonymous for replication]
Date: 2026-01-25
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
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("="*80)

# =============================================================================
# STEP 1: Load Data with Memory Optimization
# =============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

# Only load columns we need
cols_needed = ['YEAR', 'STATEFIP', 'AGE', 'SEX', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'MARST',
               'UHRSWORK', 'EMPSTAT', 'PERWT']

dtype_dict = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'AGE': 'int8',
    'SEX': 'int8',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'MARST': 'int8',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'PERWT': 'float32'
}

print("Loading ACS data (filtering during load to save memory)...")

# Load data in chunks and filter as we go
chunks = []
chunk_size = 1000000

for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtype_dict,
                          chunksize=chunk_size, low_memory=False):
    # Apply filters immediately to save memory
    # Filter 1: Hispanic-Mexican
    chunk = chunk[chunk['HISPAN'] == 1]
    # Filter 2: Born in Mexico
    chunk = chunk[chunk['BPL'] == 200]
    # Filter 3: Non-citizen
    chunk = chunk[chunk['CITIZEN'] == 3]
    # Filter 4: Working age
    chunk = chunk[(chunk['AGE'] >= 16) & (chunk['AGE'] <= 64)]
    # Filter 5: Exclude 2012
    chunk = chunk[chunk['YEAR'] != 2012]
    # Filter 6: Valid immigration year
    chunk = chunk[chunk['YRIMMIG'] > 0]

    if len(chunk) > 0:
        chunks.append(chunk)
    print(f"  Processed chunk, current filtered records: {sum(len(c) for c in chunks):,}")

df_mexican = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory
print(f"\nTotal observations after filtering: {len(df_mexican):,}")
print(f"Years in data: {sorted(df_mexican['YEAR'].unique())}")

# =============================================================================
# STEP 2: Sample Summary
# =============================================================================
print("\n" + "="*80)
print("STEP 2: Sample Summary")
print("="*80)

print(f"Sample characteristics after restrictions:")
print(f"  - Hispanic-Mexican, Mexican-born, Non-citizen")
print(f"  - Working age (16-64)")
print(f"  - Excluding 2012")
print(f"  - Valid immigration year")
print(f"  Total N: {len(df_mexican):,}")

# =============================================================================
# STEP 3: Create Key Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 3: Creating Key Variables")
print("="*80)

# Age at immigration (approximation)
df_mexican['age_at_immig'] = df_mexican['YRIMMIG'] - df_mexican['BIRTHYR']

# Post-DACA indicator (2013-2016)
df_mexican['post'] = (df_mexican['YEAR'] >= 2013).astype('int8')

# DACA Eligibility criteria:
# 1. Arrived before age 16: age_at_immig < 16
# 2. Not yet 31 as of June 15, 2012: BIRTHYR >= 1982 (born after June 15, 1981)
# 3. Present since June 15, 2007: YRIMMIG <= 2007
# 4. Non-citizen: already restricted above

df_mexican['eligible'] = (
    (df_mexican['age_at_immig'] < 16) &  # Arrived before 16th birthday
    (df_mexican['BIRTHYR'] >= 1982) &     # Under 31 as of June 2012
    (df_mexican['YRIMMIG'] <= 2007)       # In US since at least 2007
).astype('int8')

print(f"DACA eligible observations: {df_mexican['eligible'].sum():,} ({100*df_mexican['eligible'].mean():.1f}%)")
print(f"Non-eligible observations: {(1-df_mexican['eligible']).sum():,} ({100*(1-df_mexican['eligible'].mean()):.1f}%)")

# Outcome: Full-time employment (UHRSWORK >= 35)
df_mexican['fulltime'] = (df_mexican['UHRSWORK'] >= 35).astype('int8')

# Interaction term for DiD
df_mexican['eligible_post'] = df_mexican['eligible'] * df_mexican['post']

print(f"\nOutcome variable distribution:")
print(f"Full-time employed: {df_mexican['fulltime'].sum():,} ({100*df_mexican['fulltime'].mean():.1f}%)")

# =============================================================================
# STEP 4: Create Control Variables
# =============================================================================
print("\n" + "="*80)
print("STEP 4: Creating Control Variables")
print("="*80)

# Age squared
df_mexican['age_sq'] = df_mexican['AGE'].astype('int16') ** 2

# Female indicator
df_mexican['female'] = (df_mexican['SEX'] == 2).astype('int8')

# Education categories
df_mexican['educ_cat'] = pd.cut(df_mexican['EDUC'],
                                 bins=[-1, 2, 6, 10, 99],
                                 labels=['Less than HS', 'HS/Some College', 'College', 'Grad+'])

# Marital status: married vs not
df_mexican['married'] = (df_mexican['MARST'].isin([1, 2])).astype('int8')

print("Control variables created:")
print(f"  - Female: {df_mexican['female'].mean()*100:.1f}%")
print(f"  - Married: {df_mexican['married'].mean()*100:.1f}%")
print(f"  - Mean age: {df_mexican['AGE'].mean():.1f}")
print(f"  - Education distribution:")
print(df_mexican['educ_cat'].value_counts(normalize=True).sort_index())

# =============================================================================
# STEP 5: Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 5: Summary Statistics")
print("="*80)

# Summary by eligibility status
print("\n--- Summary by DACA Eligibility Status ---")
summary_vars = ['fulltime', 'AGE', 'female', 'married', 'EDUC', 'UHRSWORK']

summary_stats = df_mexican.groupby('eligible')[summary_vars].agg(['mean', 'std', 'count'])
print(summary_stats)

# Summary by period
print("\n--- Summary by Period (Pre vs Post DACA) ---")
period_stats = df_mexican.groupby('post')[summary_vars].mean()
print(period_stats)

# 2x2 table: Eligible x Post
print("\n--- Mean Full-Time Employment (2x2 Table) ---")
table_2x2 = df_mexican.groupby(['eligible', 'post'])['fulltime'].mean().unstack()
table_2x2.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
table_2x2.index = ['Non-Eligible', 'Eligible']
print(table_2x2)

# Calculate raw DiD estimate
pre_diff = table_2x2.loc['Eligible', 'Pre-DACA (2006-2011)'] - table_2x2.loc['Non-Eligible', 'Pre-DACA (2006-2011)']
post_diff = table_2x2.loc['Eligible', 'Post-DACA (2013-2016)'] - table_2x2.loc['Non-Eligible', 'Post-DACA (2013-2016)']
raw_did = post_diff - pre_diff
print(f"\nRaw DiD estimate: {raw_did:.4f}")

# Sample sizes by group and period
print("\n--- Sample Sizes ---")
sample_sizes = df_mexican.groupby(['eligible', 'post']).size().unstack()
sample_sizes.columns = ['Pre-DACA', 'Post-DACA']
sample_sizes.index = ['Non-Eligible', 'Eligible']
print(sample_sizes)

# =============================================================================
# STEP 6: Main DiD Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 6: Main Difference-in-Differences Analysis")
print("="*80)

# Convert necessary columns to float for regression
df_mexican['AGE'] = df_mexican['AGE'].astype('float32')
df_mexican['age_sq'] = df_mexican['age_sq'].astype('float32')
df_mexican['fulltime'] = df_mexican['fulltime'].astype('float32')
df_mexican['eligible'] = df_mexican['eligible'].astype('float32')
df_mexican['post'] = df_mexican['post'].astype('float32')
df_mexican['eligible_post'] = df_mexican['eligible_post'].astype('float32')
df_mexican['female'] = df_mexican['female'].astype('float32')
df_mexican['married'] = df_mexican['married'].astype('float32')
df_mexican['PERWT'] = df_mexican['PERWT'].astype('float64')

# Model 1: Basic DiD (no controls, no weights)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('fulltime ~ eligible + post + eligible_post', data=df_mexican).fit()
print(model1.summary().tables[1])

# Model 2: DiD with individual controls
print("\n--- Model 2: DiD with Individual Controls ---")
model2 = smf.ols('fulltime ~ eligible + post + eligible_post + AGE + age_sq + female + married + C(EDUC)',
                  data=df_mexican).fit()
print(model2.summary().tables[1])

# Model 3: DiD with state and year fixed effects
print("\n--- Model 3: DiD with State and Year Fixed Effects ---")
model3 = smf.ols('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                  data=df_mexican).fit()
# Print only key coefficients
print("Key coefficients:")
print(f"eligible:       {model3.params['eligible']:.4f} (SE: {model3.bse['eligible']:.4f})")
print(f"eligible_post:  {model3.params['eligible_post']:.4f} (SE: {model3.bse['eligible_post']:.4f})")

# Model 4: Full model with weights (PREFERRED SPECIFICATION)
print("\n--- Model 4: Full Model with Weights (PREFERRED) ---")
model4 = smf.wls('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                  data=df_mexican, weights=df_mexican['PERWT']).fit()
print("Key coefficients:")
print(f"eligible:       {model4.params['eligible']:.4f} (SE: {model4.bse['eligible']:.4f})")
print(f"eligible_post:  {model4.params['eligible_post']:.4f} (SE: {model4.bse['eligible_post']:.4f})")
print(f"                t-stat: {model4.tvalues['eligible_post']:.3f}, p-value: {model4.pvalues['eligible_post']:.4f}")

# Calculate 95% CI for preferred estimate
ci_low = model4.params['eligible_post'] - 1.96 * model4.bse['eligible_post']
ci_high = model4.params['eligible_post'] + 1.96 * model4.bse['eligible_post']
print(f"                95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# =============================================================================
# STEP 7: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 7: Robustness Checks")
print("="*80)

# Robustness 1: Alternative eligibility definition (relaxing birth year to 1981)
print("\n--- Robustness 1: Alternative Eligibility (BIRTHYR >= 1981) ---")
df_mexican['eligible_alt'] = (
    (df_mexican['age_at_immig'] < 16) &
    (df_mexican['BIRTHYR'] >= 1981) &  # More inclusive
    (df_mexican['YRIMMIG'] <= 2007)
).astype('float32')
df_mexican['eligible_alt_post'] = df_mexican['eligible_alt'] * df_mexican['post']

model_rob1 = smf.wls('fulltime ~ eligible_alt + eligible_alt_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                      data=df_mexican, weights=df_mexican['PERWT']).fit()
print(f"eligible_alt_post: {model_rob1.params['eligible_alt_post']:.4f} (SE: {model_rob1.bse['eligible_alt_post']:.4f})")

# Robustness 2: Restrict to younger sample (age 18-35)
print("\n--- Robustness 2: Younger Sample (Age 18-35) ---")
df_young = df_mexican[(df_mexican['AGE'] >= 18) & (df_mexican['AGE'] <= 35)].copy()
model_rob2 = smf.wls('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                      data=df_young, weights=df_young['PERWT']).fit()
print(f"eligible_post: {model_rob2.params['eligible_post']:.4f} (SE: {model_rob2.bse['eligible_post']:.4f})")
print(f"Sample size: {len(df_young):,}")

# Robustness 3: Men only
print("\n--- Robustness 3: Men Only ---")
df_men = df_mexican[df_mexican['female'] == 0].copy()
model_rob3 = smf.wls('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + C(EDUC)',
                      data=df_men, weights=df_men['PERWT']).fit()
print(f"eligible_post: {model_rob3.params['eligible_post']:.4f} (SE: {model_rob3.bse['eligible_post']:.4f})")
print(f"Sample size: {len(df_men):,}")

# Robustness 4: Women only
print("\n--- Robustness 4: Women Only ---")
df_women = df_mexican[df_mexican['female'] == 1].copy()
model_rob4 = smf.wls('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + married + C(EDUC)',
                      data=df_women, weights=df_women['PERWT']).fit()
print(f"eligible_post: {model_rob4.params['eligible_post']:.4f} (SE: {model_rob4.bse['eligible_post']:.4f})")
print(f"Sample size: {len(df_women):,}")

# Robustness 5: Linear probability model with robust standard errors
print("\n--- Robustness 5: Robust Standard Errors ---")
model_rob5 = smf.wls('fulltime ~ eligible + eligible_post + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                      data=df_mexican, weights=df_mexican['PERWT']).fit(cov_type='HC1')
print(f"eligible_post: {model_rob5.params['eligible_post']:.4f} (Robust SE: {model_rob5.bse['eligible_post']:.4f})")

# =============================================================================
# STEP 8: Pre-Trends Check (Placebo Test)
# =============================================================================
print("\n" + "="*80)
print("STEP 8: Pre-Trends / Placebo Test")
print("="*80)

# Create year-specific treatment effects
df_pre = df_mexican[df_mexican['YEAR'] <= 2011].copy()
for year in [2007, 2008, 2009, 2010, 2011]:
    df_pre[f'year_{year}'] = (df_pre['YEAR'] == year).astype('float32')
    df_pre[f'elig_{year}'] = df_pre['eligible'] * df_pre[f'year_{year}']

model_pretrend = smf.wls('fulltime ~ eligible + elig_2007 + elig_2008 + elig_2009 + elig_2010 + elig_2011 + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)',
                          data=df_pre, weights=df_pre['PERWT']).fit()

print("Pre-trend coefficients (relative to 2006):")
for year in [2007, 2008, 2009, 2010, 2011]:
    coef = model_pretrend.params[f'elig_{year}']
    se = model_pretrend.bse[f'elig_{year}']
    pval = model_pretrend.pvalues[f'elig_{year}']
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p: {pval:.3f})")

del df_pre  # Free memory

# =============================================================================
# STEP 9: Event Study
# =============================================================================
print("\n" + "="*80)
print("STEP 9: Event Study Analysis")
print("="*80)

# Create year dummies and interactions (use 2011 as reference)
for year in range(2006, 2017):
    if year not in [2011, 2012]:  # Skip 2012 and reference year 2011
        df_mexican[f'year_{year}'] = (df_mexican['YEAR'] == year).astype('float32')
        df_mexican[f'elig_{year}'] = df_mexican['eligible'] * df_mexican[f'year_{year}']

# Build formula for event study
event_vars = ' + '.join([f'elig_{year}' for year in range(2006, 2017) if year not in [2011, 2012]])
event_formula = f'fulltime ~ eligible + {event_vars} + C(YEAR) + C(STATEFIP) + AGE + age_sq + female + married + C(EDUC)'

model_event = smf.wls(event_formula, data=df_mexican, weights=df_mexican['PERWT']).fit()

print("Event study coefficients (reference year: 2011):")
event_study_results = []
for year in range(2006, 2017):
    if year == 2011:
        event_study_results.append({'Year': year, 'Coefficient': 0, 'SE': 0, 'Post': 0})
        print(f"  {year}: 0.0000 (reference)")
    elif year == 2012:
        continue
    else:
        coef = model_event.params[f'elig_{year}']
        se = model_event.bse[f'elig_{year}']
        post = 1 if year >= 2013 else 0
        event_study_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'Post': post})
        print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

event_df = pd.DataFrame(event_study_results)

# =============================================================================
# STEP 10: Final Results Summary
# =============================================================================
print("\n" + "="*80)
print("STEP 10: FINAL RESULTS SUMMARY")
print("="*80)

print("\n*** PREFERRED ESTIMATE (Model 4: WLS with Full Controls) ***")
print(f"Effect of DACA eligibility on full-time employment: {model4.params['eligible_post']:.4f}")
print(f"Standard Error: {model4.bse['eligible_post']:.4f}")
print(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"t-statistic: {model4.tvalues['eligible_post']:.3f}")
print(f"p-value: {model4.pvalues['eligible_post']:.4f}")
print(f"Sample Size: {int(model4.nobs):,}")

# Save key results for report
results_dict = {
    'preferred_estimate': model4.params['eligible_post'],
    'preferred_se': model4.bse['eligible_post'],
    'preferred_ci_low': ci_low,
    'preferred_ci_high': ci_high,
    'preferred_pvalue': model4.pvalues['eligible_post'],
    'preferred_n': int(model4.nobs),
    'model1_estimate': model1.params['eligible_post'],
    'model1_se': model1.bse['eligible_post'],
    'model2_estimate': model2.params['eligible_post'],
    'model2_se': model2.bse['eligible_post'],
    'model3_estimate': model3.params['eligible_post'],
    'model3_se': model3.bse['eligible_post'],
    'rob_alt_estimate': model_rob1.params['eligible_alt_post'],
    'rob_alt_se': model_rob1.bse['eligible_alt_post'],
    'rob_young_estimate': model_rob2.params['eligible_post'],
    'rob_young_se': model_rob2.bse['eligible_post'],
    'rob_men_estimate': model_rob3.params['eligible_post'],
    'rob_men_se': model_rob3.bse['eligible_post'],
    'rob_women_estimate': model_rob4.params['eligible_post'],
    'rob_women_se': model_rob4.bse['eligible_post'],
    'rob_robust_se': model_rob5.bse['eligible_post'],
}

# Summary statistics for report
summary_for_report = {
    'total_sample': len(df_mexican),
    'n_eligible': int(df_mexican['eligible'].sum()),
    'n_noneligible': int((1-df_mexican['eligible']).sum()),
    'pct_eligible': df_mexican['eligible'].mean() * 100,
    'fulltime_rate_overall': df_mexican['fulltime'].mean() * 100,
    'fulltime_rate_eligible_pre': df_mexican[(df_mexican['eligible']==1) & (df_mexican['post']==0)]['fulltime'].mean() * 100,
    'fulltime_rate_eligible_post': df_mexican[(df_mexican['eligible']==1) & (df_mexican['post']==1)]['fulltime'].mean() * 100,
    'fulltime_rate_nonelig_pre': df_mexican[(df_mexican['eligible']==0) & (df_mexican['post']==0)]['fulltime'].mean() * 100,
    'fulltime_rate_nonelig_post': df_mexican[(df_mexican['eligible']==0) & (df_mexican['post']==1)]['fulltime'].mean() * 100,
    'mean_age_eligible': df_mexican[df_mexican['eligible']==1]['AGE'].mean(),
    'mean_age_noneligible': df_mexican[df_mexican['eligible']==0]['AGE'].mean(),
    'pct_female': df_mexican['female'].mean() * 100,
    'pct_married': df_mexican['married'].mean() * 100,
}

print("\n--- Summary Statistics for Report ---")
for key, value in summary_for_report.items():
    if isinstance(value, float):
        print(f"{key}: {value:.2f}")
    else:
        print(f"{key}: {value:,}")

# Save results to CSV for LaTeX
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'With Controls', 'State/Year FE', 'Full Model (Preferred)',
              'Alt Eligibility', 'Young Sample', 'Men Only', 'Women Only'],
    'Estimate': [
        results_dict['model1_estimate'],
        results_dict['model2_estimate'],
        results_dict['model3_estimate'],
        results_dict['preferred_estimate'],
        results_dict['rob_alt_estimate'],
        results_dict['rob_young_estimate'],
        results_dict['rob_men_estimate'],
        results_dict['rob_women_estimate']
    ],
    'SE': [
        results_dict['model1_se'],
        results_dict['model2_se'],
        results_dict['model3_se'],
        results_dict['preferred_se'],
        results_dict['rob_alt_se'],
        results_dict['rob_young_se'],
        results_dict['rob_men_se'],
        results_dict['rob_women_se']
    ]
})
results_df.to_csv('results_table.csv', index=False)
print("\nResults saved to results_table.csv")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save summary stats
summary_df = pd.DataFrame([summary_for_report])
summary_df.to_csv('summary_stats.csv', index=False)
print("Summary statistics saved to summary_stats.csv")

# Save 2x2 table
table_2x2.to_csv('table_2x2.csv')
print("2x2 table saved to table_2x2.csv")

# Save all detailed results
with open('detailed_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - DETAILED RESULTS\n")
    f.write("="*60 + "\n\n")

    f.write("PREFERRED ESTIMATE (Model 4):\n")
    f.write(f"  Coefficient: {model4.params['eligible_post']:.6f}\n")
    f.write(f"  Standard Error: {model4.bse['eligible_post']:.6f}\n")
    f.write(f"  95% CI: [{ci_low:.6f}, {ci_high:.6f}]\n")
    f.write(f"  t-stat: {model4.tvalues['eligible_post']:.3f}\n")
    f.write(f"  p-value: {model4.pvalues['eligible_post']:.6f}\n")
    f.write(f"  N: {int(model4.nobs):,}\n\n")

    f.write("ALL MODEL RESULTS:\n")
    for i, row in results_df.iterrows():
        f.write(f"  {row['Model']}: {row['Estimate']:.4f} (SE: {row['SE']:.4f})\n")

    f.write("\n\nSUMMARY STATISTICS:\n")
    for key, value in summary_for_report.items():
        f.write(f"  {key}: {value}\n")

    f.write("\n\n2x2 TABLE:\n")
    f.write(table_2x2.to_string())

    f.write("\n\nEVENT STUDY COEFFICIENTS:\n")
    for _, row in event_df.iterrows():
        f.write(f"  {int(row['Year'])}: {row['Coefficient']:.4f} (SE: {row['SE']:.4f})\n")

print("Detailed results saved to detailed_results.txt")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

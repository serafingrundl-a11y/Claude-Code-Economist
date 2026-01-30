"""
DACA Replication Study Analysis Script
======================================
Examines the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.

Research Design: Difference-in-Differences
- Treatment: DACA-eligible individuals aged 26-30 on June 15, 2012
- Control: Individuals aged 31-35 on June 15, 2012 (age-ineligible)
- Outcome: Full-time employment (>=35 hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*70)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("="*70)

# -----------------------------------------------------------------------------
# STEP 1: Load Data in Chunks (to handle large file)
# -----------------------------------------------------------------------------
print("\n[STEP 1] Loading data in chunks...")
data_path = "data/data.csv"

# Define columns needed
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EMPSTAT', 'UHRSWORK', 'STATEFIP', 'METRO']

def calculate_age_june_2012(birthyr, birthqtr):
    """
    Calculate age as of June 15, 2012.
    Birth quarters: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
    """
    base_age = 2012 - birthyr
    age_june_2012 = np.where(birthqtr >= 3, base_age - 1, base_age)
    return age_june_2012

# Process in chunks to filter data
chunks = []
chunk_size = 500000
total_rows = 0

for chunk in pd.read_csv(data_path, usecols=cols_needed, chunksize=chunk_size, low_memory=False):
    total_rows += len(chunk)

    # Apply filters immediately to reduce memory
    # Filter 1: Hispanic-Mexican ethnicity (HISPAN = 1)
    chunk = chunk[chunk['HISPAN'] == 1]

    # Filter 2: Born in Mexico (BPL = 200)
    chunk = chunk[chunk['BPL'] == 200]

    # Filter 3: Non-citizen (CITIZEN = 3)
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Filter 4: Valid immigration year
    chunk = chunk[(chunk['YRIMMIG'] > 0) & (chunk['YRIMMIG'] <= chunk['YEAR'])]

    # Filter 5: Arrived before age 16
    chunk['arrival_age'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['arrival_age'] < 16]

    # Filter 6: Continuously in US since 2007
    chunk = chunk[chunk['YRIMMIG'] <= 2007]

    # Calculate age on June 15, 2012
    chunk['age_june_2012'] = calculate_age_june_2012(chunk['BIRTHYR'].values, chunk['BIRTHQTR'].values)

    # Filter 7: Keep only treatment (26-30) and control (31-35) age groups
    chunk = chunk[(chunk['age_june_2012'] >= 26) & (chunk['age_june_2012'] <= 35)]

    # Filter 8: Exclude 2012
    chunk = chunk[chunk['YEAR'] != 2012]

    if len(chunk) > 0:
        chunks.append(chunk)

    print(f"  Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} observations", end='\r')
    gc.collect()

print(f"\n  Total rows processed: {total_rows:,}")

# Combine filtered chunks
df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"  Final sample size: {len(df):,}")
print(f"  Years in data: {sorted(df['YEAR'].unique())}")

# -----------------------------------------------------------------------------
# STEP 2: Create Variables
# -----------------------------------------------------------------------------
print("\n[STEP 2] Creating analysis variables...")

# Treatment indicator (1 for 26-30, 0 for 31-35)
df['treated'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)

# Post-treatment period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Full-time employment (>=35 hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Interaction term
df['treated_post'] = df['treated'] * df['post']

# Demographics
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] <= 2).astype(int)

# Education categories
df['educ_less_hs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = ((df['EDUC'] >= 6) & (df['EDUC'] <= 7)).astype(int)
df['educ_some_college'] = ((df['EDUC'] >= 8) & (df['EDUC'] <= 9)).astype(int)
df['educ_college_plus'] = (df['EDUC'] >= 10).astype(int)

# Metro area
df['metro'] = (df['METRO'] >= 2).astype(int)

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

# Year and state factors
df['year_factor'] = df['YEAR'].astype(str)
df['state_factor'] = df['STATEFIP'].astype(str)

print(f"  Treatment group (ages 26-30): {df['treated'].sum():,}")
print(f"  Control group (ages 31-35): {(1-df['treated']).sum():,}")
print(f"  Pre-period (2006-2011): {(df['post']==0).sum():,}")
print(f"  Post-period (2013-2016): {(df['post']==1).sum():,}")

# -----------------------------------------------------------------------------
# STEP 3: Summary Statistics
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[STEP 3] SUMMARY STATISTICS")
print("="*70)

def weighted_stats(data, var, weight='PERWT'):
    """Calculate weighted mean and std"""
    w = data[weight]
    mean = np.average(data[var], weights=w)
    variance = np.average((data[var] - mean)**2, weights=w)
    return mean, np.sqrt(variance)

# Sample sizes by group and period
print("\n--- Table 1: Sample Sizes by Group and Period ---")
sample_counts = df.groupby(['treated', 'post']).size().unstack(fill_value=0)
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre (2006-2011)', 'Post (2013-2016)']
print(sample_counts)
print(f"\nTotal sample: {len(df):,}")

# Mean full-time employment by group and period
print("\n--- Table 2: Full-Time Employment Rate by Group and Period ---")
for trt_label, trt_val in [('Control (31-35)', 0), ('Treatment (26-30)', 1)]:
    print(f"\n  {trt_label}:")
    for period_label, period_val in [('Pre', 0), ('Post', 1)]:
        subset = df[(df['treated'] == trt_val) & (df['post'] == period_val)]
        mean, std = weighted_stats(subset, 'fulltime')
        print(f"    {period_label}: {mean:.4f} (SD: {std:.4f}, N: {len(subset):,})")

# Calculate simple DiD
pre_treat = df[(df['treated'] == 1) & (df['post'] == 0)]
post_treat = df[(df['treated'] == 1) & (df['post'] == 1)]
pre_control = df[(df['treated'] == 0) & (df['post'] == 0)]
post_control = df[(df['treated'] == 0) & (df['post'] == 1)]

y_treat_pre, _ = weighted_stats(pre_treat, 'fulltime')
y_treat_post, _ = weighted_stats(post_treat, 'fulltime')
y_control_pre, _ = weighted_stats(pre_control, 'fulltime')
y_control_post, _ = weighted_stats(post_control, 'fulltime')

did_simple = (y_treat_post - y_treat_pre) - (y_control_post - y_control_pre)

print(f"\n--- Simple Difference-in-Differences ---")
print(f"  Treatment change: {y_treat_post:.4f} - {y_treat_pre:.4f} = {y_treat_post - y_treat_pre:.4f}")
print(f"  Control change:   {y_control_post:.4f} - {y_control_pre:.4f} = {y_control_post - y_control_pre:.4f}")
print(f"  DiD estimate:     {did_simple:.4f}")

# Covariate balance
print("\n--- Table 3: Covariate Balance (Pre-Period Means) ---")
pre_df = df[df['post'] == 0]
covars = ['female', 'married', 'educ_less_hs', 'educ_hs', 'educ_some_college',
          'educ_college_plus', 'years_in_us', 'arrival_age', 'metro']

print(f"{'Variable':<20} {'Treatment':>12} {'Control':>12} {'Diff':>12}")
print("-" * 56)
for var in covars:
    t_mean, _ = weighted_stats(pre_df[pre_df['treated']==1], var)
    c_mean, _ = weighted_stats(pre_df[pre_df['treated']==0], var)
    print(f"{var:<20} {t_mean:>12.4f} {c_mean:>12.4f} {t_mean-c_mean:>12.4f}")

# -----------------------------------------------------------------------------
# STEP 4: Regression Analysis
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[STEP 4] DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*70)

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model1.params['treated_post']:.4f}")
print(f"  Standard error:  {model1.bse['treated_post']:.4f}")
print(f"  t-statistic:     {model1.tvalues['treated_post']:.4f}")
print(f"  p-value:         {model1.pvalues['treated_post']:.4f}")

# Model 2: DiD with year FE
print("\n--- Model 2: DiD with Year Fixed Effects ---")
model2 = smf.wls('fulltime ~ treated + C(year_factor) + treated_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model2.params['treated_post']:.4f}")
print(f"  Standard error:  {model2.bse['treated_post']:.4f}")
print(f"  p-value:         {model2.pvalues['treated_post']:.4f}")

# Model 3: DiD with covariates
print("\n--- Model 3: DiD with Covariates ---")
formula3 = ('fulltime ~ treated + C(year_factor) + treated_post + '
            'female + married + educ_hs + educ_some_college + educ_college_plus + years_in_us')
model3 = smf.wls(formula3, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model3.params['treated_post']:.4f}")
print(f"  Standard error:  {model3.bse['treated_post']:.4f}")
print(f"  p-value:         {model3.pvalues['treated_post']:.4f}")
print(f"  95% CI:          [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")

# Model 4: DiD with state and year FE
print("\n--- Model 4: DiD with State + Year FE + Covariates (PREFERRED) ---")
formula4 = ('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + '
            'female + married + educ_hs + educ_some_college + educ_college_plus + years_in_us')
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {model4.params['treated_post']:.4f}")
print(f"  Standard error:  {model4.bse['treated_post']:.4f}")
print(f"  t-statistic:     {model4.tvalues['treated_post']:.4f}")
print(f"  p-value:         {model4.pvalues['treated_post']:.4f}")
print(f"  95% CI:          [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  R-squared:       {model4.rsquared:.4f}")
print(f"  N:               {int(model4.nobs):,}")

# -----------------------------------------------------------------------------
# STEP 5: Robustness Checks
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[STEP 5] ROBUSTNESS CHECKS")
print("="*70)

# Pre-trend test
print("\n--- Pre-Trend Test (Linear Trend in Pre-Period) ---")
pre_only = df[df['post'] == 0].copy()
pre_only['year_linear'] = pre_only['YEAR'] - 2006
formula_trend = 'fulltime ~ treated * year_linear'
trend_model = smf.wls(formula_trend, data=pre_only, weights=pre_only['PERWT']).fit(cov_type='HC1')
print(f"  Interaction (treated x year): {trend_model.params['treated:year_linear']:.4f}")
print(f"  Standard error:               {trend_model.bse['treated:year_linear']:.4f}")
print(f"  p-value:                      {trend_model.pvalues['treated:year_linear']:.4f}")
parallel = "Yes (p > 0.05)" if trend_model.pvalues['treated:year_linear'] > 0.05 else "No (p <= 0.05)"
print(f"  Parallel trends supported:    {parallel}")

# Placebo test
print("\n--- Placebo Test (Fake Treatment in 2009) ---")
placebo = df[df['YEAR'] <= 2011].copy()
placebo['fake_post'] = (placebo['YEAR'] >= 2009).astype(int)
placebo['treated_fake'] = placebo['treated'] * placebo['fake_post']
placebo_model = smf.wls('fulltime ~ treated + fake_post + treated_fake',
                         data=placebo, weights=placebo['PERWT']).fit(cov_type='HC1')
print(f"  Placebo DiD:     {placebo_model.params['treated_fake']:.4f}")
print(f"  Standard error:  {placebo_model.bse['treated_fake']:.4f}")
print(f"  p-value:         {placebo_model.pvalues['treated_fake']:.4f}")

# Employment (any) as outcome
print("\n--- Alternative Outcome: Any Employment ---")
formula_emp = ('employed ~ treated + C(year_factor) + C(state_factor) + treated_post + '
               'female + married + educ_hs + educ_some_college + educ_college_plus + years_in_us')
emp_model = smf.wls(formula_emp, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD coefficient: {emp_model.params['treated_post']:.4f}")
print(f"  Standard error:  {emp_model.bse['treated_post']:.4f}")
print(f"  p-value:         {emp_model.pvalues['treated_post']:.4f}")

# By gender
print("\n--- Heterogeneity by Gender ---")
formula_het = ('fulltime ~ treated + C(year_factor) + treated_post + '
               'married + educ_hs + educ_some_college + educ_college_plus + years_in_us')
male_model = smf.wls(formula_het, data=df[df['female']==0],
                      weights=df[df['female']==0]['PERWT']).fit(cov_type='HC1')
female_model = smf.wls(formula_het, data=df[df['female']==1],
                        weights=df[df['female']==1]['PERWT']).fit(cov_type='HC1')
print(f"  Males:   {male_model.params['treated_post']:.4f} (SE: {male_model.bse['treated_post']:.4f}, p: {male_model.pvalues['treated_post']:.4f})")
print(f"  Females: {female_model.params['treated_post']:.4f} (SE: {female_model.bse['treated_post']:.4f}, p: {female_model.pvalues['treated_post']:.4f})")

# -----------------------------------------------------------------------------
# STEP 6: Event Study
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[STEP 6] EVENT STUDY - YEAR-SPECIFIC EFFECTS")
print("="*70)

# Create year-specific interactions (2011 as reference)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for yr in years:
    df[f'treat_{yr}'] = ((df['treated'] == 1) & (df['YEAR'] == yr)).astype(int)

formula_event = ('fulltime ~ treated + C(year_factor) + ' +
                 ' + '.join([f'treat_{yr}' for yr in years]) +
                 ' + female + married + educ_hs + educ_some_college + educ_college_plus + years_in_us')
event_model = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nYear-Specific Treatment Effects (Reference: 2011)")
print("-" * 60)
print(f"{'Year':<8} {'Coefficient':>12} {'SE':>10} {'p-value':>10} {'95% CI':>24}")
print("-" * 60)
for yr in years:
    var = f'treat_{yr}'
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci = event_model.conf_int().loc[var]
    print(f"{yr:<8} {coef:>12.4f} {se:>10.4f} {pval:>10.4f} [{ci[0]:>9.4f}, {ci[1]:>9.4f}]")

# -----------------------------------------------------------------------------
# STEP 7: Save Results
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("[STEP 7] SAVING RESULTS")
print("="*70)

# Results dictionary
results = {
    'simple_did': float(did_simple),
    'model1_coef': float(model1.params['treated_post']),
    'model1_se': float(model1.bse['treated_post']),
    'model1_pval': float(model1.pvalues['treated_post']),
    'model2_coef': float(model2.params['treated_post']),
    'model2_se': float(model2.bse['treated_post']),
    'model2_pval': float(model2.pvalues['treated_post']),
    'model3_coef': float(model3.params['treated_post']),
    'model3_se': float(model3.bse['treated_post']),
    'model3_pval': float(model3.pvalues['treated_post']),
    'model3_ci_low': float(model3.conf_int().loc['treated_post', 0]),
    'model3_ci_high': float(model3.conf_int().loc['treated_post', 1]),
    'model4_coef': float(model4.params['treated_post']),
    'model4_se': float(model4.bse['treated_post']),
    'model4_pval': float(model4.pvalues['treated_post']),
    'model4_ci_low': float(model4.conf_int().loc['treated_post', 0]),
    'model4_ci_high': float(model4.conf_int().loc['treated_post', 1]),
    'model4_rsq': float(model4.rsquared),
    'n_total': int(len(df)),
    'n_treatment': int(df['treated'].sum()),
    'n_control': int((1-df['treated']).sum()),
    'n_pre': int((df['post']==0).sum()),
    'n_post': int((df['post']==1).sum()),
    'y_treat_pre': float(y_treat_pre),
    'y_treat_post': float(y_treat_post),
    'y_control_pre': float(y_control_pre),
    'y_control_post': float(y_control_post),
    'pretrend_coef': float(trend_model.params['treated:year_linear']),
    'pretrend_pval': float(trend_model.pvalues['treated:year_linear']),
    'placebo_coef': float(placebo_model.params['treated_fake']),
    'placebo_pval': float(placebo_model.pvalues['treated_fake']),
    'emp_coef': float(emp_model.params['treated_post']),
    'emp_pval': float(emp_model.pvalues['treated_post']),
    'male_coef': float(male_model.params['treated_post']),
    'male_se': float(male_model.bse['treated_post']),
    'male_pval': float(male_model.pvalues['treated_post']),
    'female_coef': float(female_model.params['treated_post']),
    'female_se': float(female_model.bse['treated_post']),
    'female_pval': float(female_model.pvalues['treated_post']),
}

# Event study results
for yr in years:
    var = f'treat_{yr}'
    results[f'event_{yr}_coef'] = float(event_model.params[var])
    results[f'event_{yr}_se'] = float(event_model.bse[var])

import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("  Results saved to results.json")

# Yearly means for figure
yearly_means = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
yearly_means.columns = ['YEAR', 'treated', 'fulltime_mean']
yearly_means.to_csv('yearly_means.csv', index=False)
print("  Yearly means saved to yearly_means.csv")

# Summary stats
summary = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'years_in_us': 'mean',
    'PERWT': 'sum'
}).round(4)
summary.to_csv('summary_stats.csv')
print("  Summary statistics saved to summary_stats.csv")

# Full model results
print("\n  Saving full regression output...")
with open('model4_full.txt', 'w') as f:
    f.write(model4.summary().as_text())
print("  Full model 4 results saved to model4_full.txt")

# -----------------------------------------------------------------------------
# FINAL SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"""
RESEARCH QUESTION:
  Effect of DACA eligibility on full-time employment among
  Hispanic-Mexican, Mexican-born non-citizens in the US.

IDENTIFICATION STRATEGY:
  Difference-in-Differences comparing:
  - Treatment: Ages 26-30 on June 15, 2012 (DACA-eligible)
  - Control:   Ages 31-35 on June 15, 2012 (age-ineligible)
  - Pre-period:  2006-2011
  - Post-period: 2013-2016

SAMPLE:
  Total observations: {len(df):,}
  Treatment group:    {df['treated'].sum():,}
  Control group:      {(1-df['treated']).sum():,}

PREFERRED ESTIMATE (Model 4: State + Year FE + Covariates):
  Effect Size:    {model4.params['treated_post']:.4f}
  Standard Error: {model4.bse['treated_post']:.4f}
  95% CI:         [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]
  p-value:        {model4.pvalues['treated_post']:.4f}
  R-squared:      {model4.rsquared:.4f}

INTERPRETATION:
  DACA eligibility is associated with a {abs(model4.params['treated_post'])*100:.2f} percentage point
  {'increase' if model4.params['treated_post'] > 0 else 'decrease'} in the probability of full-time employment.
  {'This effect is statistically significant at the 5% level.' if model4.pvalues['treated_post'] < 0.05 else 'This effect is not statistically significant at the 5% level.'}
""")

print("\nAnalysis complete!")

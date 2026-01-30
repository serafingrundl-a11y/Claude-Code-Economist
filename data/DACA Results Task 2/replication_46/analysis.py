"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals

Treatment: Ages 26-30 on June 15, 2012
Control: Ages 31-35 on June 15, 2012
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output files
OUTPUT_DIR = "C:/Users/seraf/DACA Results Task 2/replication_46/"
DATA_PATH = "C:/Users/seraf/DACA Results Task 2/replication_46/data/data.csv"

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# ============================================================================
# STEP 1: Load and filter data
# ============================================================================
print("\n[STEP 1] Loading data...")

# Define columns we need
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
               'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST', 'STATEFIP']

# Load data in chunks and filter
chunk_size = 500000
filtered_chunks = []

print("Loading data in chunks and filtering...")
for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican (HISPAN == 1)
    chunk = chunk[chunk['HISPAN'] == 1]
    # Filter to born in Mexico (BPL == 200)
    chunk = chunk[chunk['BPL'] == 200]
    # Filter to non-citizens (CITIZEN == 3)
    chunk = chunk[chunk['CITIZEN'] == 3]

    if len(chunk) > 0:
        filtered_chunks.append(chunk)

    if (i + 1) % 20 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

print("Combining filtered chunks...")
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"  Initial filtered sample: {len(df):,} observations")

# ============================================================================
# STEP 2: Create age on June 15, 2012 variable
# ============================================================================
print("\n[STEP 2] Creating age variables...")

# Approximate whether born before or after June 15 based on birth quarter
# Q1 (Jan-Mar): born before June 15
# Q2 (Apr-Jun): assume born before June 15 (conservative, most of Q2 is before)
# Q3 (Jul-Sep): born after June 15
# Q4 (Oct-Dec): born after June 15

# Calculate age on June 15, 2012
# If born in Q1 or Q2 (before June 15), age = 2012 - birthyear
# If born in Q3 or Q4 (after June 15), age = 2012 - birthyear - 1

df['age_june2012'] = np.where(
    df['BIRTHQTR'].isin([1, 2]),
    2012 - df['BIRTHYR'],
    2012 - df['BIRTHYR'] - 1
)

print(f"  Age on June 15, 2012 range: {df['age_june2012'].min()} to {df['age_june2012'].max()}")

# ============================================================================
# STEP 3: Define treatment and control groups
# ============================================================================
print("\n[STEP 3] Defining treatment and control groups...")

# Treatment group: ages 26-30 on June 15, 2012
# Control group: ages 31-35 on June 15, 2012

df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"  Sample after age restriction: {len(df):,} observations")

# ============================================================================
# STEP 4: Apply DACA eligibility criteria
# ============================================================================
print("\n[STEP 4] Applying DACA eligibility criteria...")

# Criterion: Arrived in US before age 16
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]
print(f"  After arrived before age 16: {len(df):,} observations")

# Criterion: Continuous residence since June 15, 2007
# Proxy: Arrived by 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"  After arrived by 2007: {len(df):,} observations")

# ============================================================================
# STEP 5: Define time periods
# ============================================================================
print("\n[STEP 5] Defining time periods...")

# Pre-treatment: 2006-2011
# Post-treatment: 2013-2016
# Exclude 2012 (treatment year - cannot distinguish before/after June 15)

df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"  Sample excluding 2012: {len(df):,} observations")
print(f"  Pre-period (2006-2011): {(df['post']==0).sum():,}")
print(f"  Post-period (2013-2016): {(df['post']==1).sum():,}")

# ============================================================================
# STEP 6: Define outcome variable
# ============================================================================
print("\n[STEP 6] Defining outcome variable...")

# Full-time employment: Usually working 35+ hours per week
# Must also be employed (EMPSTAT == 1)

df['fulltime'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

# Summary statistics
print(f"  Overall full-time rate: {df['fulltime'].mean()*100:.2f}%")
print(f"  Full-time rate (weighted): {np.average(df['fulltime'], weights=df['PERWT'])*100:.2f}%")

# ============================================================================
# STEP 7: Summary statistics
# ============================================================================
print("\n[STEP 7] Summary statistics...")

# By group and period
summary = df.groupby(['treated', 'post']).apply(
    lambda x: pd.Series({
        'n': len(x),
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'fulltime_se': np.sqrt(np.average((x['fulltime'] - np.average(x['fulltime'], weights=x['PERWT']))**2, weights=x['PERWT']) / len(x)),
        'age_mean': np.average(x['AGE'], weights=x['PERWT']),
        'female_pct': np.average(x['SEX']==2, weights=x['PERWT']),
        'married_pct': np.average(x['MARST'].isin([1,2]), weights=x['PERWT'])
    })
).reset_index()

print("\nSummary by Group and Period:")
print("-" * 70)
for _, row in summary.iterrows():
    group = "Treatment" if row['treated'] == 1 else "Control"
    period = "Post" if row['post'] == 1 else "Pre"
    print(f"{group:10} {period:5}: n={int(row['n']):6,}, FT={row['fulltime_mean']*100:5.2f}%, "
          f"Age={row['age_mean']:.1f}, Female={row['female_pct']*100:.1f}%, Married={row['married_pct']*100:.1f}%")

# ============================================================================
# STEP 8: Simple Difference-in-Differences
# ============================================================================
print("\n[STEP 8] Simple Difference-in-Differences...")

# Calculate means
treat_pre = df[(df['treated']==1) & (df['post']==0)]
treat_post = df[(df['treated']==1) & (df['post']==1)]
ctrl_pre = df[(df['treated']==0) & (df['post']==0)]
ctrl_post = df[(df['treated']==0) & (df['post']==1)]

treat_pre_mean = np.average(treat_pre['fulltime'], weights=treat_pre['PERWT'])
treat_post_mean = np.average(treat_post['fulltime'], weights=treat_post['PERWT'])
ctrl_pre_mean = np.average(ctrl_pre['fulltime'], weights=ctrl_pre['PERWT'])
ctrl_post_mean = np.average(ctrl_post['fulltime'], weights=ctrl_post['PERWT'])

# DiD estimate
treat_diff = treat_post_mean - treat_pre_mean
ctrl_diff = ctrl_post_mean - ctrl_pre_mean
did_estimate = treat_diff - ctrl_diff

print(f"\n  Treatment group pre-period:  {treat_pre_mean*100:.3f}%")
print(f"  Treatment group post-period: {treat_post_mean*100:.3f}%")
print(f"  Treatment group change:      {treat_diff*100:+.3f} pp")
print(f"\n  Control group pre-period:    {ctrl_pre_mean*100:.3f}%")
print(f"  Control group post-period:   {ctrl_post_mean*100:.3f}%")
print(f"  Control group change:        {ctrl_diff*100:+.3f} pp")
print(f"\n  DiD Estimate:                {did_estimate*100:+.3f} pp")

# ============================================================================
# STEP 9: Regression-based DiD (no controls)
# ============================================================================
print("\n[STEP 9] Regression-based DiD (no controls)...")

# Create interaction term
df['treat_post'] = df['treated'] * df['post']

# Basic DiD regression
model1 = smf.wls('fulltime ~ treated + post + treat_post',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nBasic DiD Regression Results:")
print("-" * 50)
print(f"{'Variable':<15} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}")
print("-" * 50)
for var in ['Intercept', 'treated', 'post', 'treat_post']:
    coef = model1.params[var]
    se = model1.bse[var]
    t = model1.tvalues[var]
    p = model1.pvalues[var]
    print(f"{var:<15} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f}")
print("-" * 50)
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {int(model1.nobs):,}")

did_coef = model1.params['treat_post']
did_se = model1.bse['treat_post']
did_ci_low = did_coef - 1.96 * did_se
did_ci_high = did_coef + 1.96 * did_se

print(f"\nDiD Estimate (treat_post): {did_coef:.4f} ({did_coef*100:.2f} pp)")
print(f"Standard Error: {did_se:.4f}")
print(f"95% CI: [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"        [{did_ci_low*100:.2f} pp, {did_ci_high*100:.2f} pp]")

# ============================================================================
# STEP 10: Regression-based DiD with covariates
# ============================================================================
print("\n[STEP 10] Regression-based DiD with covariates...")

# Create control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = df['MARST'].isin([1, 2]).astype(int)
df['educ_hs'] = (df['EDUC'] >= 6).astype(int)  # High school or more
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # Some college or more

# Model with basic controls
model2 = smf.wls('fulltime ~ treated + post + treat_post + female + married + educ_hs + educ_college',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nDiD Regression with Demographic Controls:")
print("-" * 50)
print(f"{'Variable':<15} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8}")
print("-" * 50)
for var in model2.params.index:
    coef = model2.params[var]
    se = model2.bse[var]
    t = model2.tvalues[var]
    p = model2.pvalues[var]
    print(f"{var:<15} {coef:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f}")
print("-" * 50)
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {int(model2.nobs):,}")

did_coef2 = model2.params['treat_post']
did_se2 = model2.bse['treat_post']
did_ci_low2 = did_coef2 - 1.96 * did_se2
did_ci_high2 = did_coef2 + 1.96 * did_se2

print(f"\nDiD Estimate with controls: {did_coef2:.4f} ({did_coef2*100:.2f} pp)")
print(f"Standard Error: {did_se2:.4f}")
print(f"95% CI: [{did_ci_low2:.4f}, {did_ci_high2:.4f}]")

# ============================================================================
# STEP 11: Year fixed effects model
# ============================================================================
print("\n[STEP 11] DiD with year fixed effects...")

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)

year_vars = [col for col in year_dummies.columns]
formula3 = 'fulltime ~ treated + treat_post + female + married + educ_hs + educ_college + ' + ' + '.join(year_vars)

model3 = smf.wls(formula3, data=df_with_years, weights=df_with_years['PERWT']).fit(cov_type='HC1')

did_coef3 = model3.params['treat_post']
did_se3 = model3.bse['treat_post']
did_ci_low3 = did_coef3 - 1.96 * did_se3
did_ci_high3 = did_coef3 + 1.96 * did_se3

print(f"\nDiD Estimate with year FE: {did_coef3:.4f} ({did_coef3*100:.2f} pp)")
print(f"Standard Error: {did_se3:.4f}")
print(f"95% CI: [{did_ci_low3:.4f}, {did_ci_high3:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")

# ============================================================================
# STEP 12: State fixed effects model
# ============================================================================
print("\n[STEP 12] DiD with state and year fixed effects...")

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_fe = pd.concat([df_with_years, state_dummies], axis=1)

state_vars = [col for col in state_dummies.columns]
formula4 = 'fulltime ~ treated + treat_post + female + married + educ_hs + educ_college + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(state_vars)

model4 = smf.wls(formula4, data=df_with_fe, weights=df_with_fe['PERWT']).fit(cov_type='HC1')

did_coef4 = model4.params['treat_post']
did_se4 = model4.bse['treat_post']
did_ci_low4 = did_coef4 - 1.96 * did_se4
did_ci_high4 = did_coef4 + 1.96 * did_se4

print(f"\nDiD Estimate with state+year FE: {did_coef4:.4f} ({did_coef4*100:.2f} pp)")
print(f"Standard Error: {did_se4:.4f}")
print(f"95% CI: [{did_ci_low4:.4f}, {did_ci_high4:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")

# ============================================================================
# STEP 13: Event study / Pre-trends analysis
# ============================================================================
print("\n[STEP 13] Event study analysis...")

# Create year-specific treatment effects (relative to 2011)
event_study_data = df.copy()

# Create year dummies for event study (excluding 2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_study_data[f'year_{year}'] = (event_study_data['YEAR'] == year).astype(int)
    event_study_data[f'treat_year_{year}'] = ((event_study_data['YEAR'] == year) &
                                               (event_study_data['treated'] == 1)).astype(int)

event_vars = [f'treat_year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
year_vars_event = [f'year_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
formula_event = 'fulltime ~ treated + ' + ' + '.join(event_vars) + ' + ' + ' + '.join(year_vars_event)

model_event = smf.wls(formula_event, data=event_study_data, weights=event_study_data['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
print(f"{'Year':<10} {'Coef':>10} {'Std Err':>10} {'95% CI':<25}")
print("-" * 50)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    print(f"{year:<10} {coef:>10.4f} {se:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}]")
print("-" * 50)

# ============================================================================
# STEP 14: Robustness - By gender
# ============================================================================
print("\n[STEP 14] Robustness check: By gender...")

for gender, gender_name in [(1, 'Male'), (2, 'Female')]:
    df_gender = df[df['SEX'] == gender]
    model_gender = smf.wls('fulltime ~ treated + post + treat_post',
                           data=df_gender, weights=df_gender['PERWT']).fit(cov_type='HC1')
    coef = model_gender.params['treat_post']
    se = model_gender.bse['treat_post']
    print(f"  {gender_name}: DiD = {coef:.4f} ({coef*100:.2f} pp), SE = {se:.4f}, N = {int(model_gender.nobs):,}")

# ============================================================================
# STEP 15: Robustness - Alternative age bandwidths
# ============================================================================
print("\n[STEP 15] Robustness check: Alternative age bandwidths...")

# Narrower bandwidth: ages 27-29 vs 32-34
df_narrow = df[(df['age_june2012'].isin([27, 28, 29, 32, 33, 34]))]
df_narrow['treated_narrow'] = df_narrow['age_june2012'].isin([27, 28, 29]).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated_narrow + post + treat_post_narrow',
                       data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"  Narrow (27-29 vs 32-34): DiD = {model_narrow.params['treat_post_narrow']:.4f}, "
      f"SE = {model_narrow.bse['treat_post_narrow']:.4f}, N = {int(model_narrow.nobs):,}")

# ============================================================================
# STEP 16: Robustness - Placebo test (using 2008 as fake treatment year)
# ============================================================================
print("\n[STEP 16] Placebo test (fake treatment in 2008, pre-period only)...")

df_placebo = df[df['YEAR'] <= 2011]
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['treated'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treated + post_placebo + treat_post_placebo',
                        data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"  Placebo DiD (2008): {model_placebo.params['treat_post_placebo']:.4f}, "
      f"SE = {model_placebo.bse['treat_post_placebo']:.4f}, p = {model_placebo.pvalues['treat_post_placebo']:.4f}")

# ============================================================================
# STEP 17: Summary of Results
# ============================================================================
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)

print("\nSample Information:")
print(f"  Total observations: {len(df):,}")
print(f"  Treatment group (ages 26-30): {df['treated'].sum():,}")
print(f"  Control group (ages 31-35): {(df['treated']==0).sum():,}")
print(f"  Pre-period observations: {(df['post']==0).sum():,}")
print(f"  Post-period observations: {(df['post']==1).sum():,}")

print("\nPreferred Estimate (Model with Covariates):")
print(f"  Effect: {did_coef2*100:.2f} percentage points")
print(f"  Standard Error: {did_se2*100:.2f} pp")
print(f"  95% Confidence Interval: [{did_ci_low2*100:.2f}, {did_ci_high2*100:.2f}] pp")
print(f"  p-value: {model2.pvalues['treat_post']:.4f}")

print("\nAll Model Specifications:")
print(f"  1. Basic DiD:              {did_coef*100:+.2f} pp (SE: {did_se*100:.2f})")
print(f"  2. With covariates:        {did_coef2*100:+.2f} pp (SE: {did_se2*100:.2f})")
print(f"  3. With year FE:           {did_coef3*100:+.2f} pp (SE: {did_se3*100:.2f})")
print(f"  4. With state+year FE:     {did_coef4*100:+.2f} pp (SE: {did_se4*100:.2f})")

# ============================================================================
# STEP 18: Save results for report
# ============================================================================
print("\n[STEP 18] Saving results...")

results = {
    'n_total': len(df),
    'n_treated': df['treated'].sum(),
    'n_control': (df['treated']==0).sum(),
    'n_pre': (df['post']==0).sum(),
    'n_post': (df['post']==1).sum(),
    'did_basic': did_coef,
    'did_basic_se': did_se,
    'did_controls': did_coef2,
    'did_controls_se': did_se2,
    'did_yearfe': did_coef3,
    'did_yearfe_se': did_se3,
    'did_stateyrfe': did_coef4,
    'did_stateyrfe_se': did_se4,
    'treat_pre_mean': treat_pre_mean,
    'treat_post_mean': treat_post_mean,
    'ctrl_pre_mean': ctrl_pre_mean,
    'ctrl_post_mean': ctrl_post_mean,
}

# Save summary stats
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['count', 'mean'],
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean',
    'PERWT': 'sum'
}).round(4)

summary_stats.to_csv(OUTPUT_DIR + 'summary_stats.csv')

# Save event study coefficients
event_study_results = []
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_year_{year}'
    event_study_results.append({
        'year': year,
        'coefficient': model_event.params[var],
        'std_error': model_event.bse[var],
        'ci_low': model_event.params[var] - 1.96 * model_event.bse[var],
        'ci_high': model_event.params[var] + 1.96 * model_event.bse[var]
    })
event_study_df = pd.DataFrame(event_study_results)
event_study_df.to_csv(OUTPUT_DIR + 'event_study.csv', index=False)

# Save main results
results_df = pd.DataFrame([results])
results_df.to_csv(OUTPUT_DIR + 'main_results.csv', index=False)

print("\nResults saved to:")
print(f"  {OUTPUT_DIR}summary_stats.csv")
print(f"  {OUTPUT_DIR}event_study.csv")
print(f"  {OUTPUT_DIR}main_results.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

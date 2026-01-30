"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals

Memory-optimized version using chunked reading
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

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# Define columns to use (minimal set needed)
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
           'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EMPSTAT', 'UHRSWORK']

# Define dtypes to reduce memory
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8'
}

print("\n[1] Loading and filtering data in chunks...")

# Process in chunks to filter as we go
chunks = []
chunk_size = 1000000

for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes, chunksize=chunk_size):
    # Filter to Hispanic-Mexican (HISPAN == 1) AND Mexican-born (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    print(f"  Processed chunk, kept {len(filtered):,} rows")

df_mex = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican, Mexican-born observations: {len(df_mex):,}")

# Free memory
del chunks

# ============================================================================
# STEP 2: Filter to non-citizens (proxy for undocumented)
# ============================================================================
print("\n[2] Filtering to non-citizens (CITIZEN == 3)...")
print(f"CITIZEN value counts in Mexican-born Hispanic-Mexican population:")
print(df_mex['CITIZEN'].value_counts())

df_noncit = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"\nObservations after filtering to non-citizens: {len(df_noncit):,}")
del df_mex

# ============================================================================
# STEP 3: Create DACA eligibility criteria
# ============================================================================
print("\n[3] Creating DACA eligibility variables...")

# Calculate age at immigration
df_noncit['age_at_immigration'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# DACA Criteria:
# 1. Arrived before age 16
# 2. Under 31 as of June 15, 2012
# 3. In US since June 15, 2007

df_noncit['arrived_before_16'] = (df_noncit['age_at_immigration'] < 16).astype('int8')
df_noncit['in_us_since_2007'] = (df_noncit['YRIMMIG'] <= 2007).astype('int8')

# Under 31 as of June 15, 2012 (vectorized)
df_noncit['under_31_june_2012'] = (
    (df_noncit['BIRTHYR'] >= 1982) |
    ((df_noncit['BIRTHYR'] == 1981) & (df_noncit['BIRTHQTR'].isin([3, 4])))
).astype('int8')

# Combine all criteria
df_noncit['daca_eligible'] = (
    (df_noncit['arrived_before_16'] == 1) &
    (df_noncit['in_us_since_2007'] == 1) &
    (df_noncit['under_31_june_2012'] == 1)
).astype('int8')

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df_noncit['arrived_before_16'].sum():,}")
print(f"  In US since 2007: {df_noncit['in_us_since_2007'].sum():,}")
print(f"  Under 31 as of June 2012: {df_noncit['under_31_june_2012'].sum():,}")
print(f"  DACA eligible (all criteria): {df_noncit['daca_eligible'].sum():,}")
print(f"  Not DACA eligible: {(df_noncit['daca_eligible'] == 0).sum():,}")

# ============================================================================
# STEP 4: Create outcome variable - Full-time employment
# ============================================================================
print("\n[4] Creating outcome variable...")

df_noncit['fulltime'] = (df_noncit['UHRSWORK'] >= 35).astype('int8')
df_noncit['employed'] = (df_noncit['EMPSTAT'] == 1).astype('int8')

print(f"\nEmployment status (EMPSTAT value counts):")
print(df_noncit['EMPSTAT'].value_counts())

print(f"\nUHRSWORK summary statistics:")
print(df_noncit['UHRSWORK'].describe())

# ============================================================================
# STEP 5: Create treatment period indicator
# ============================================================================
print("\n[5] Creating treatment period indicator...")

df_noncit['post'] = (df_noncit['YEAR'] >= 2013).astype('int8')

# Exclude 2012 for main analysis
df_analysis = df_noncit[df_noncit['YEAR'] != 2012].copy()
print(f"\nObservations after excluding 2012: {len(df_analysis):,}")

print(f"\nYear distribution:")
print(df_analysis['YEAR'].value_counts().sort_index())

# ============================================================================
# STEP 6: Restrict to working-age population
# ============================================================================
print("\n[6] Restricting to working-age population (16-64)...")

df_analysis = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)].copy()
print(f"Observations after age restriction: {len(df_analysis):,}")

# ============================================================================
# STEP 7: Summary statistics
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Summary by eligibility and period
print("\n[A] Sample sizes by eligibility and period:")
summary_table = pd.crosstab(df_analysis['daca_eligible'], df_analysis['post'], margins=True)
summary_table.index = ['Not Eligible', 'DACA Eligible', 'Total']
summary_table.columns = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
print(summary_table)

# Mean full-time employment rates
print("\n[B] Full-time employment rates by eligibility and period:")
ft_rates = df_analysis.groupby(['daca_eligible', 'post'])['fulltime'].agg(['mean', 'std', 'count'])
print(ft_rates)

# Calculate raw DiD
eligible_pre = df_analysis[(df_analysis['daca_eligible'] == 1) & (df_analysis['post'] == 0)]['fulltime'].mean()
eligible_post = df_analysis[(df_analysis['daca_eligible'] == 1) & (df_analysis['post'] == 1)]['fulltime'].mean()
ineligible_pre = df_analysis[(df_analysis['daca_eligible'] == 0) & (df_analysis['post'] == 0)]['fulltime'].mean()
ineligible_post = df_analysis[(df_analysis['daca_eligible'] == 0) & (df_analysis['post'] == 1)]['fulltime'].mean()

print(f"\n[C] Raw Difference-in-Differences Calculation:")
print(f"  DACA Eligible Pre:    {eligible_pre:.4f}")
print(f"  DACA Eligible Post:   {eligible_post:.4f}")
print(f"  Eligible Diff:        {eligible_post - eligible_pre:.4f}")
print(f"  Not Eligible Pre:     {ineligible_pre:.4f}")
print(f"  Not Eligible Post:    {ineligible_post:.4f}")
print(f"  Not Eligible Diff:    {ineligible_post - ineligible_pre:.4f}")
print(f"  DiD Estimate:         {(eligible_post - eligible_pre) - (ineligible_post - ineligible_pre):.4f}")

# ============================================================================
# STEP 8: Prepare variables for regression
# ============================================================================
print("\n[7] Preparing regression variables...")

df_analysis['eligible_x_post'] = (df_analysis['daca_eligible'] * df_analysis['post']).astype('int8')
df_analysis['age_sq'] = (df_analysis['AGE'] ** 2).astype('int16')
df_analysis['female'] = (df_analysis['SEX'] == 2).astype('int8')
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype('int8')

# ============================================================================
# STEP 9: Difference-in-Differences Regression
# ============================================================================
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD without controls
print("\n[Model 1] Basic DiD (no controls):")
model1 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n[Model 2] DiD with demographic controls:")
model2 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married',
                 data=df_analysis).fit()
print(model2.summary().tables[1])

# Model 3: DiD with state fixed effects
print("\n[Model 3] DiD with demographic controls and state fixed effects:")
model3 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married + C(STATEFIP)',
                 data=df_analysis).fit()
print(f"  daca_eligible:    {model3.params['daca_eligible']:.6f} (SE: {model3.bse['daca_eligible']:.6f})")
print(f"  post:             {model3.params['post']:.6f} (SE: {model3.bse['post']:.6f})")
print(f"  eligible_x_post:  {model3.params['eligible_x_post']:.6f} (SE: {model3.bse['eligible_x_post']:.6f})")

# Model 4: State and year fixed effects
print("\n[Model 4] DiD with state and year fixed effects:")
model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + C(STATEFIP) + C(YEAR)',
                 data=df_analysis).fit()
print(f"  daca_eligible:    {model4.params['daca_eligible']:.6f} (SE: {model4.bse['daca_eligible']:.6f})")
print(f"  eligible_x_post:  {model4.params['eligible_x_post']:.6f} (SE: {model4.bse['eligible_x_post']:.6f})")

# ============================================================================
# STEP 10: Weighted and Clustered SEs
# ============================================================================
print("\n" + "=" * 80)
print("WEIGHTED AND CLUSTERED STANDARD ERRORS")
print("=" * 80)

# Weighted regression
print("\n[Model 5] Weighted DiD:")
X_vars = ['daca_eligible', 'post', 'eligible_x_post', 'AGE', 'age_sq', 'female', 'married']
X = df_analysis[X_vars].copy()
X = sm.add_constant(X)
y = df_analysis['fulltime']
weights = df_analysis['PERWT']
model5 = sm.WLS(y, X, weights=weights).fit()
print(model5.summary().tables[1])

# Clustered standard errors
print("\n[Model 6] DiD with clustered standard errors (by state):")
model6 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(model6.summary().tables[1])

# ============================================================================
# STEP 11: Event Study
# ============================================================================
print("\n" + "=" * 80)
print("EVENT STUDY ANALYSIS")
print("=" * 80)

years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype('int8')
    df_analysis[f'eligible_x_{year}'] = (df_analysis['daca_eligible'] * df_analysis[f'year_{year}']).astype('int8')

year_interactions = ' + '.join([f'eligible_x_{year}' for year in years])
year_dummies = ' + '.join([f'year_{year}' for year in years])
formula = f'fulltime ~ daca_eligible + {year_dummies} + {year_interactions} + AGE + age_sq + female + married'
model_event = smf.ols(formula, data=df_analysis).fit()

print("\nEvent Study Coefficients (eligible x year interaction):")
print("Reference year: 2011")
event_results = []
for year in years:
    coef = model_event.params[f'eligible_x_{year}']
    se = model_event.bse[f'eligible_x_{year}']
    pval = model_event.pvalues[f'eligible_x_{year}']
    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
    print(f"  {year}: {coef:8.5f} (SE: {se:.5f}) {sig}")
    event_results.append({'Year': year, 'Coefficient': coef, 'SE': se, 'p_value': pval})

# ============================================================================
# STEP 12: Heterogeneity Analysis
# ============================================================================
print("\n" + "=" * 80)
print("HETEROGENEITY ANALYSIS")
print("=" * 80)

# By gender
print("\n[A] By Gender:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    model_sub = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + married',
                        data=df_sub).fit()
    print(f"  {label}: DiD = {model_sub.params['eligible_x_post']:.5f} (SE: {model_sub.bse['eligible_x_post']:.5f}), N = {len(df_sub):,}")

# By age group
print("\n[B] By Age Group:")
df_analysis['age_group'] = pd.cut(df_analysis['AGE'], bins=[16, 25, 35, 45, 64], labels=['16-25', '26-35', '36-45', '46-64'])
for ag in ['16-25', '26-35', '36-45', '46-64']:
    df_sub = df_analysis[df_analysis['age_group'] == ag]
    if len(df_sub) > 100 and df_sub['daca_eligible'].sum() > 10:
        model_sub = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + female + married',
                            data=df_sub).fit()
        print(f"  {ag}: DiD = {model_sub.params['eligible_x_post']:.5f} (SE: {model_sub.bse['eligible_x_post']:.5f}), N = {len(df_sub):,}")

# ============================================================================
# STEP 13: Robustness Checks
# ============================================================================
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# R2: Alternative outcome - employment (any work)
print("\n[R1] Alternative Outcome: Any employment (employed = 1):")
model_r2 = smf.ols('employed ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married',
                   data=df_analysis).fit()
print(f"  DiD estimate: {model_r2.params['eligible_x_post']:.5f} (SE: {model_r2.bse['eligible_x_post']:.5f})")

# R3: Including 2012
print("\n[R2] Including 2012 (partial treatment year):")
df_with_2012 = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 64)].copy()
df_with_2012['post'] = (df_with_2012['YEAR'] >= 2012).astype('int8')
df_with_2012['eligible_x_post'] = (df_with_2012['daca_eligible'] * df_with_2012['post']).astype('int8')
df_with_2012['age_sq'] = (df_with_2012['AGE'] ** 2).astype('int16')
df_with_2012['female'] = (df_with_2012['SEX'] == 2).astype('int8')
df_with_2012['married'] = (df_with_2012['MARST'].isin([1, 2])).astype('int8')
model_r3 = smf.ols('fulltime ~ daca_eligible + post + eligible_x_post + AGE + age_sq + female + married',
                   data=df_with_2012).fit()
print(f"  DiD estimate: {model_r3.params['eligible_x_post']:.5f} (SE: {model_r3.bse['eligible_x_post']:.5f})")
print(f"  Sample size: {len(df_with_2012):,}")

# R4: Placebo test - pre-period only
print("\n[R3] Placebo Test (Pre-period only, fake treatment at 2009):")
df_pre = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype('int8')
df_pre['eligible_x_placebo'] = (df_pre['daca_eligible'] * df_pre['placebo_post']).astype('int8')
model_placebo = smf.ols('fulltime ~ daca_eligible + placebo_post + eligible_x_placebo + AGE + age_sq + female + married',
                        data=df_pre).fit()
print(f"  Placebo DiD estimate: {model_placebo.params['eligible_x_placebo']:.5f} (SE: {model_placebo.bse['eligible_x_placebo']:.5f})")
print(f"  p-value: {model_placebo.pvalues['eligible_x_placebo']:.4f}")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print("\n" + "-" * 60)
print("PREFERRED SPECIFICATION: Model 2 (with demographic controls)")
print("-" * 60)
print(f"DiD Coefficient (eligible_x_post): {model2.params['eligible_x_post']:.6f}")
print(f"Standard Error:                    {model2.bse['eligible_x_post']:.6f}")
print(f"95% CI: [{model2.conf_int().loc['eligible_x_post', 0]:.6f}, {model2.conf_int().loc['eligible_x_post', 1]:.6f}]")
print(f"t-statistic:                       {model2.tvalues['eligible_x_post']:.4f}")
print(f"p-value:                           {model2.pvalues['eligible_x_post']:.6f}")
print(f"Sample size:                       {int(model2.nobs):,}")

print("\n" + "-" * 60)
print("WITH CLUSTERED STANDARD ERRORS (Model 6)")
print("-" * 60)
print(f"DiD Coefficient (eligible_x_post): {model6.params['eligible_x_post']:.6f}")
print(f"Clustered SE (by state):           {model6.bse['eligible_x_post']:.6f}")
print(f"95% CI: [{model6.conf_int().loc['eligible_x_post', 0]:.6f}, {model6.conf_int().loc['eligible_x_post', 1]:.6f}]")

# ============================================================================
# Save results
# ============================================================================
results_dict = {
    'Model': ['Basic DiD', 'With Controls', 'State FE', 'State+Year FE', 'Weighted', 'Clustered SE'],
    'DiD_Coefficient': [
        model1.params['eligible_x_post'],
        model2.params['eligible_x_post'],
        model3.params['eligible_x_post'],
        model4.params['eligible_x_post'],
        model5.params['eligible_x_post'],
        model6.params['eligible_x_post']
    ],
    'Standard_Error': [
        model1.bse['eligible_x_post'],
        model2.bse['eligible_x_post'],
        model3.bse['eligible_x_post'],
        model4.bse['eligible_x_post'],
        model5.bse['eligible_x_post'],
        model6.bse['eligible_x_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model5.nobs),
        int(model6.nobs)
    ]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'AGE': ['mean'],
    'female': ['mean'],
    'married': ['mean']
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

print("\nResults saved to CSV files.")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

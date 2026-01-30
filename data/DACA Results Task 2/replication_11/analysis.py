"""
DACA Replication Study - Analysis Script
Replication 11

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals?

Identification Strategy: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded)
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA IN CHUNKS AND FILTER
# =============================================================================
print("\n1. Loading and filtering data in chunks...")
data_path = "data/data.csv"

# Define columns we need
use_cols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
            'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
            'EMPSTAT', 'UHRSWORK']

# Read in chunks and filter
chunks = []
chunk_size = 500000
total_rows = 0

for chunk in pd.read_csv(data_path, usecols=use_cols, chunksize=chunk_size):
    total_rows += len(chunk)

    # Apply filters
    # Exclude 2012
    chunk = chunk[chunk['YEAR'] != 2012]

    # Hispanic-Mexican (HISPAN = 1)
    chunk = chunk[chunk['HISPAN'] == 1]

    # Mexican-born (BPL = 200)
    chunk = chunk[chunk['BPL'] == 200]

    # Not a citizen (CITIZEN = 3)
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Calculate age at DACA
    chunk['age_at_daca'] = 2012 - chunk['BIRTHYR']

    # Restrict to ages 26-35 at DACA
    chunk = chunk[(chunk['age_at_daca'] >= 26) & (chunk['age_at_daca'] <= 35)]

    # Must have immigration year
    chunk = chunk[chunk['YRIMMIG'] > 0]

    # Arrived before age 16
    chunk['age_at_immig'] = chunk['YRIMMIG'] - chunk['BIRTHYR']
    chunk = chunk[chunk['age_at_immig'] < 16]

    # In US since 2007
    chunk = chunk[chunk['YRIMMIG'] <= 2007]

    if len(chunk) > 0:
        chunks.append(chunk)

    print(f"   Processed {total_rows:,} rows, kept {sum(len(c) for c in chunks):,} so far...")

# Combine chunks
df = pd.concat(chunks, ignore_index=True)
print(f"\n   Final sample size: {len(df):,}")
print(f"   Years in sample: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. CREATE ANALYSIS VARIABLES
# =============================================================================
print("\n2. Creating analysis variables...")

# Treatment indicator (ages 26-30 vs 31-35)
df['treat'] = ((df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 30)).astype(int)

# Post-DACA indicator
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['treat_post'] = df['treat'] * df['post']

# Full-time employment outcome (employed AND 35+ hours)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['EMPSTAT'] == 1) & (df['UHRSWORK'] >= 35)).astype(int)

# Demographic controls
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)

# Education categories
df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
df['hs_grad'] = (df['EDUC'] == 6).astype(int)
df['some_college'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['college_plus'] = (df['EDUC'] >= 10).astype(int)

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("3. DESCRIPTIVE STATISTICS")
print("="*80)

print("\nSample sizes by group and period:")
sample_counts = df.groupby(['treat', 'post']).size().unstack()
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(sample_counts)

print("\nWeighted sample sizes (using PERWT):")
weighted_counts = df.groupby(['treat', 'post'])['PERWT'].sum().unstack()
weighted_counts.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_counts.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(weighted_counts.round(0))

print("\nFull-time employment rates by group and period (unweighted):")
ft_rates = df.groupby(['treat', 'post'])['fulltime'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print((ft_rates * 100).round(2))

print("\nFull-time employment rates by group and period (weighted):")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_rates_wt = df.groupby(['treat', 'post']).apply(weighted_mean).unstack()
ft_rates_wt.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates_wt.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print((ft_rates_wt * 100).round(2))

# Simple DiD calculation
print("\n" + "-"*80)
print("Simple Difference-in-Differences Calculation (Weighted):")
print("-"*80)
pre_treat = ft_rates_wt.iloc[1, 0]
post_treat = ft_rates_wt.iloc[1, 1]
pre_control = ft_rates_wt.iloc[0, 0]
post_control = ft_rates_wt.iloc[0, 1]

diff_treat = post_treat - pre_treat
diff_control = post_control - pre_control
did_simple = diff_treat - diff_control

print(f"Treatment group (26-30): Pre = {pre_treat*100:.2f}%, Post = {post_treat*100:.2f}%")
print(f"  Change: {diff_treat*100:.2f} percentage points")
print(f"Control group (31-35):   Pre = {pre_control*100:.2f}%, Post = {post_control*100:.2f}%")
print(f"  Change: {diff_control*100:.2f} percentage points")
print(f"\nDifference-in-Differences: {did_simple*100:.2f} percentage points")

# =============================================================================
# 4. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (no controls, unweighted) ---")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD with weights
print("\n--- Model 2: Basic DiD (no controls, weighted) ---")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls (weighted)
print("\n--- Model 3: DiD with demographic controls (weighted) ---")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit()
print("Key coefficients:")
print(f"  treat_post: {model3.params['treat_post']:.4f} (SE: {model3.bse['treat_post']:.4f})")
print(f"  t-stat: {model3.tvalues['treat_post']:.2f}, p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with year fixed effects (weighted) ---")
model4 = smf.wls('fulltime ~ treat + C(YEAR) + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit()
print("Key coefficients:")
print(f"  treat_post: {model4.params['treat_post']:.4f} (SE: {model4.bse['treat_post']:.4f})")
print(f"  t-stat: {model4.tvalues['treat_post']:.2f}, p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with state fixed effects (weighted) - PREFERRED MODEL
print("\n--- Model 5: DiD with state and year fixed effects (weighted) - PREFERRED ---")
model5 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit()
print("Key coefficients:")
print(f"  treat_post: {model5.params['treat_post']:.4f} (SE: {model5.bse['treat_post']:.4f})")
print(f"  t-stat: {model5.tvalues['treat_post']:.2f}, p-value: {model5.pvalues['treat_post']:.4f}")

# Model 6: Robust standard errors (preferred specification)
print("\n--- Model 6: Preferred model with robust standard errors ---")
model6 = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                  data=df, weights=df['PERWT']).fit(cov_type='HC1')
print("Key coefficients:")
print(f"  treat_post: {model6.params['treat_post']:.4f} (SE: {model6.bse['treat_post']:.4f})")
print(f"  t-stat: {model6.tvalues['treat_post']:.2f}, p-value: {model6.pvalues['treat_post']:.4f}")

# =============================================================================
# 5. PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "="*80)
print("5. PREFERRED ESTIMATE SUMMARY")
print("="*80)

preferred = model6
coef = preferred.params['treat_post']
se = preferred.bse['treat_post']
ci_low = coef - 1.96 * se
ci_high = coef + 1.96 * se

print(f"\nPreferred Model: DiD with state FE, year FE, demographic controls, weighted")
print(f"  Effect Size: {coef:.4f} ({coef*100:.2f} percentage points)")
print(f"  Standard Error: {se:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}] or [{ci_low*100:.2f}, {ci_high*100:.2f}] pp")
print(f"  t-statistic: {preferred.tvalues['treat_post']:.3f}")
print(f"  p-value: {preferred.pvalues['treat_post']:.4f}")
print(f"  Sample Size: {int(preferred.nobs):,}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("6. ROBUSTNESS CHECKS")
print("="*80)

# 6a. Alternative age bandwidths
print("\n--- 6a. Alternative age bandwidths ---")

# Narrower: 27-29 vs 32-34
df_narrow = df[(df['age_at_daca'] >= 27) & (df['age_at_daca'] <= 34)]
df_narrow = df_narrow[~((df_narrow['age_at_daca'] == 30) | (df_narrow['age_at_daca'] == 31))]
df_narrow['treat_narrow'] = ((df_narrow['age_at_daca'] >= 27) & (df_narrow['age_at_daca'] <= 29)).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treat_narrow + C(YEAR) + C(STATEFIP) + treat_post_narrow + female + married + C(EDUC)',
                        data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"Narrower bandwidth (27-29 vs 32-34):")
print(f"  Effect: {model_narrow.params['treat_post_narrow']:.4f} (SE: {model_narrow.bse['treat_post_narrow']:.4f})")
print(f"  Sample: {int(model_narrow.nobs):,}")

# 6b. By gender
print("\n--- 6b. By gender ---")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + married + C(EDUC)',
                      data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_post + married + C(EDUC)',
                        data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"Males:   Effect = {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}), N = {int(model_male.nobs):,}")
print(f"Females: Effect = {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}), N = {int(model_female.nobs):,}")

# 6c. Pre-trends test
print("\n--- 6c. Pre-trends test ---")
df_pre = df[df['YEAR'] < 2012].copy()
df_pre['year_trend'] = df_pre['YEAR'] - 2006
df_pre['treat_trend'] = df_pre['treat'] * df_pre['year_trend']

model_pretrend = smf.wls('fulltime ~ treat + C(YEAR) + C(STATEFIP) + treat_trend + female + married + C(EDUC)',
                          data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Treatment x Pre-Trend: {model_pretrend.params['treat_trend']:.4f} (SE: {model_pretrend.bse['treat_trend']:.4f})")
print(f"p-value: {model_pretrend.pvalues['treat_trend']:.4f}")

# 6d. Event study
print("\n--- 6d. Event study (year-specific treatment effects) ---")
df_event = df.copy()
for year in sorted(df['YEAR'].unique()):
    df_event[f'treat_year_{year}'] = df_event['treat'] * (df_event['YEAR'] == year).astype(int)

# Use 2011 as reference year
year_vars = ' + '.join([f'treat_year_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011])
formula = f'fulltime ~ treat + C(YEAR) + C(STATEFIP) + {year_vars} + female + married + C(EDUC)'
model_event = smf.wls(formula, data=df_event, weights=df_event['PERWT']).fit(cov_type='HC1')

print("Year-specific effects (relative to 2011):")
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var = f'treat_year_{year}'
        print(f"  {year}: {model_event.params[var]:.4f} (SE: {model_event.bse[var]:.4f})")

# 6e. Employment (any work) as alternative outcome
print("\n--- 6e. Alternative outcome: Any employment ---")
model_emp = smf.wls('employed ~ treat + C(YEAR) + C(STATEFIP) + treat_post + female + married + C(EDUC)',
                     data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Effect on any employment: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# =============================================================================
# 7. ADDITIONAL STATISTICS FOR REPORT
# =============================================================================
print("\n" + "="*80)
print("7. ADDITIONAL STATISTICS FOR REPORT")
print("="*80)

print("\nSample composition:")
print(f"  Total observations: {len(df):,}")
print(f"  Treatment group: {df['treat'].sum():,} ({df['treat'].mean()*100:.1f}%)")
print(f"  Control group: {(1-df['treat']).sum():,} ({(1-df['treat'].mean())*100:.1f}%)")
print(f"  Pre-DACA: {(df['post']==0).sum():,} ({(df['post']==0).mean()*100:.1f}%)")
print(f"  Post-DACA: {df['post'].sum():,} ({df['post'].mean()*100:.1f}%)")

print("\nDemographic characteristics of full sample:")
print(f"  Female: {df['female'].mean()*100:.1f}%")
print(f"  Married: {df['married'].mean()*100:.1f}%")
print(f"  Less than HS: {df['less_than_hs'].mean()*100:.1f}%")
print(f"  HS grad: {df['hs_grad'].mean()*100:.1f}%")
print(f"  Some college: {df['some_college'].mean()*100:.1f}%")
print(f"  College+: {df['college_plus'].mean()*100:.1f}%")

print("\nOutcome statistics:")
print(f"  Employment rate: {df['employed'].mean()*100:.1f}%")
print(f"  Full-time employment rate: {df['fulltime'].mean()*100:.1f}%")
print(f"  Mean usual hours worked (employed): {df[df['employed']==1]['UHRSWORK'].mean():.1f}")

# Balance table by treatment status
print("\n" + "-"*80)
print("Balance Table (Pre-DACA period only):")
print("-"*80)
df_pre_balance = df[df['post'] == 0]
balance_vars = ['female', 'married', 'less_than_hs', 'hs_grad', 'some_college', 'college_plus', 'age_at_daca']
print(f"{'Variable':<20} {'Control':>12} {'Treatment':>12} {'Diff':>10}")
for var in balance_vars:
    ctrl_mean = df_pre_balance[df_pre_balance['treat']==0][var].mean()
    treat_mean = df_pre_balance[df_pre_balance['treat']==1][var].mean()
    diff = treat_mean - ctrl_mean
    print(f"{var:<20} {ctrl_mean:>12.3f} {treat_mean:>12.3f} {diff:>10.3f}")

# =============================================================================
# 8. SAVE RESULTS FOR LATEX
# =============================================================================
print("\n" + "="*80)
print("8. SUMMARY TABLE FOR LATEX")
print("="*80)

print("\n\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Effect of DACA Eligibility on Full-Time Employment}")
print("\\label{tab:main_results}")
print("\\begin{tabular}{lccccc}")
print("\\hline\\hline")
print(" & (1) & (2) & (3) & (4) & (5) \\\\")
print("\\hline")
print(f"DACA $\\times$ Post & {model1.params['treat_post']:.4f} & {model2.params['treat_post']:.4f} & {model3.params['treat_post']:.4f} & {model4.params['treat_post']:.4f} & {model6.params['treat_post']:.4f} \\\\")
print(f" & ({model1.bse['treat_post']:.4f}) & ({model2.bse['treat_post']:.4f}) & ({model3.bse['treat_post']:.4f}) & ({model4.bse['treat_post']:.4f}) & ({model6.bse['treat_post']:.4f}) \\\\")
print("\\hline")
print("Weights & No & Yes & Yes & Yes & Yes \\\\")
print("Demographics & No & No & Yes & Yes & Yes \\\\")
print("Year FE & No & No & No & Yes & Yes \\\\")
print("State FE & No & No & No & No & Yes \\\\")
print("Robust SE & No & No & No & No & Yes \\\\")
print(f"N & {int(model1.nobs):,} & {int(model2.nobs):,} & {int(model3.nobs):,} & {int(model4.nobs):,} & {int(model6.nobs):,} \\\\")
print("\\hline\\hline")
print("\\end{tabular}")
print("\\end{table}")

print("\n" + "="*80)
print("Analysis complete.")
print("="*80)

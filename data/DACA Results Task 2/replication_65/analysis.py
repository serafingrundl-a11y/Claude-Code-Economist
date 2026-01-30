#!/usr/bin/env python3
"""
DACA Replication Analysis - ID 65
Effect of DACA eligibility on full-time employment among Hispanic-Mexican, Mexican-born individuals

Research Design: Difference-in-Differences
- Treatment Group: Ages 26-30 at time of DACA (June 15, 2012)
- Control Group: Ages 31-35 at time of DACA
- Pre-period: Before 2012 (using 2006-2011)
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/data.csv'
OUTPUT_DIR = '.'

print("=" * 60)
print("DACA Replication Analysis - ID 65")
print("=" * 60)

# ============================================================================
# STEP 1: Load and Initial Filter
# ============================================================================
print("\n[Step 1] Loading data...")

# Read in chunks due to large file size
chunks = []
chunksize = 1000000

# Key columns we need
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX', 'MARST',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK',
               'STATEFIP', 'METRO', 'FAMSIZE', 'NCHILD']

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=cols_needed, chunksize=chunksize)):
    # Initial filter: Hispanic Mexican, Born in Mexico
    chunk = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  After initial filter (Hispanic-Mexican, Born in Mexico): {len(df):,} observations")

# ============================================================================
# STEP 2: Apply DACA Eligibility Criteria
# ============================================================================
print("\n[Step 2] Applying DACA eligibility criteria...")

# Non-citizen (CITIZEN == 3)
# Note: We cannot distinguish documented vs undocumented, so we assume
# non-citizens who haven't received papers are potentially undocumented
df = df[df['CITIZEN'] == 3]
print(f"  After filtering to non-citizens: {len(df):,}")

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
# Arrived before age 16
df = df[df['age_at_immig'] < 16]
print(f"  After filtering to arrived before age 16: {len(df):,}")

# Must have been in US continuously since June 2007
# Approximation: immigration year <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"  After filtering to in US since 2007: {len(df):,}")

# ============================================================================
# STEP 3: Define Treatment and Control Groups
# ============================================================================
print("\n[Step 3] Defining treatment and control groups...")

# DACA implementation: June 15, 2012
# Must be under 31 on June 15, 2012 to be eligible
# Treatment: ages 26-30 on June 15, 2012
# Control: ages 31-35 on June 15, 2012

# Birth year determines eligibility:
# Age 26 on June 15, 2012 -> Born June 15, 1986 or shortly before
# Age 30 on June 15, 2012 -> Born June 15, 1982 or shortly before
# Age 31 on June 15, 2012 -> Born June 15, 1981 or shortly before
# Age 35 on June 15, 2012 -> Born June 15, 1977 or shortly before

# For simplicity, use birth year:
# Treatment (26-30): Born 1982-1986
# Control (31-35): Born 1977-1981

df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control'] == 1)]
print(f"  After keeping only treatment/control groups: {len(df):,}")

# Create treated indicator (1 = younger/treatment group, 0 = older/control group)
df['treated'] = df['treatment']

# ============================================================================
# STEP 4: Define Time Periods
# ============================================================================
print("\n[Step 4] Defining time periods...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA implementation)
# Exclude 2012 as it's the implementation year

df = df[df['YEAR'] != 2012]
df['post'] = (df['YEAR'] >= 2013).astype(int)
print(f"  After excluding 2012: {len(df):,}")
print(f"  Pre-period (2006-2011) observations: {len(df[df['post']==0]):,}")
print(f"  Post-period (2013-2016) observations: {len(df[df['post']==1]):,}")

# ============================================================================
# STEP 5: Define Outcome Variable
# ============================================================================
print("\n[Step 5] Defining outcome variable...")

# Full-time employment: Usually works 35+ hours per week
# First check employment status
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

print(f"  Employment rate: {df['employed'].mean()*100:.1f}%")
print(f"  Full-time employment rate: {df['fulltime'].mean()*100:.1f}%")

# ============================================================================
# STEP 6: Descriptive Statistics
# ============================================================================
print("\n[Step 6] Generating descriptive statistics...")

# Create demographic variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_less_hs'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] < 65)).astype(int)  # High school
df['educ_some_college'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)  # Some college
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)  # Bachelor's or higher

# Summary by treatment status and period
print("\n  Summary Statistics by Group and Period:")
print("-" * 70)

for treat_val, treat_name in [(1, "Treatment (26-30)"), (0, "Control (31-35)")]:
    for post_val, period_name in [(0, "Pre (2006-2011)"), (1, "Post (2013-2016)")]:
        subset = df[(df['treated'] == treat_val) & (df['post'] == post_val)]
        n = len(subset)
        ft_rate = subset['fulltime'].mean() * 100
        emp_rate = subset['employed'].mean() * 100
        print(f"  {treat_name}, {period_name}:")
        print(f"    N = {n:,}, Full-time: {ft_rate:.1f}%, Employed: {emp_rate:.1f}%")

# ============================================================================
# STEP 7: Main Difference-in-Differences Analysis
# ============================================================================
print("\n[Step 7] Running main DiD analysis...")

# Create interaction term
df['treated_post'] = df['treated'] * df['post']

# Model 1: Basic DiD (unweighted)
print("\n  Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df).fit()
print(f"    DiD Estimate: {model1.params['treated_post']:.4f}")
print(f"    Std Error: {model1.bse['treated_post']:.4f}")
print(f"    95% CI: [{model1.conf_int().loc['treated_post', 0]:.4f}, {model1.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model1.pvalues['treated_post']:.4f}")

# Model 2: Weighted DiD
print("\n  Model 2: Weighted DiD")
model2 = smf.wls('fulltime ~ treated + post + treated_post', data=df, weights=df['PERWT']).fit()
print(f"    DiD Estimate: {model2.params['treated_post']:.4f}")
print(f"    Std Error: {model2.bse['treated_post']:.4f}")
print(f"    95% CI: [{model2.conf_int().loc['treated_post', 0]:.4f}, {model2.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model2.pvalues['treated_post']:.4f}")

# Model 3: With year fixed effects
print("\n  Model 3: Weighted DiD with Year Fixed Effects")
df['year_factor'] = pd.Categorical(df['YEAR'])
model3 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post', data=df, weights=df['PERWT']).fit()
print(f"    DiD Estimate: {model3.params['treated_post']:.4f}")
print(f"    Std Error: {model3.bse['treated_post']:.4f}")
print(f"    95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: With demographic controls
print("\n  Model 4: Weighted DiD with Demographic Controls")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + '
                 'educ_hs + educ_some_college + educ_college + NCHILD',
                 data=df, weights=df['PERWT']).fit()
print(f"    DiD Estimate: {model4.params['treated_post']:.4f}")
print(f"    Std Error: {model4.bse['treated_post']:.4f}")
print(f"    95% CI: [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model4.pvalues['treated_post']:.4f}")

# Model 5: With state fixed effects
print("\n  Model 5: Weighted DiD with State Fixed Effects")
model5 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + '
                 'educ_hs + educ_some_college + educ_college + NCHILD',
                 data=df, weights=df['PERWT']).fit()
print(f"    DiD Estimate: {model5.params['treated_post']:.4f}")
print(f"    Std Error: {model5.bse['treated_post']:.4f}")
print(f"    95% CI: [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model5.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 8: Robust Standard Errors
# ============================================================================
print("\n[Step 8] Computing robust standard errors...")

# Model with heteroskedasticity-robust standard errors
model_robust = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + '
                        'educ_hs + educ_some_college + educ_college + NCHILD',
                        data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"\n  Preferred Model (Weighted DiD with Controls, Robust SEs):")
print(f"    DiD Estimate: {model_robust.params['treated_post']:.4f}")
print(f"    Robust Std Error: {model_robust.bse['treated_post']:.4f}")
print(f"    95% CI: [{model_robust.conf_int().loc['treated_post', 0]:.4f}, {model_robust.conf_int().loc['treated_post', 1]:.4f}]")
print(f"    p-value: {model_robust.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 9: Heterogeneity Analysis
# ============================================================================
print("\n[Step 9] Heterogeneity analysis...")

# By sex
print("\n  By Sex:")
for sex_val, sex_name in [(1, "Male"), (2, "Female")]:
    subset = df[df['SEX'] == sex_val]
    model_sex = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                        data=subset, weights=subset['PERWT']).fit()
    print(f"    {sex_name}: DiD = {model_sex.params['treated_post']:.4f} (SE: {model_sex.bse['treated_post']:.4f})")

# By education
print("\n  By Education:")
# Less than high school
subset_lths = df[df['educ_less_hs'] == 1]
if len(subset_lths) > 100:
    model_lths = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                         data=subset_lths, weights=subset_lths['PERWT']).fit()
    print(f"    Less than HS: DiD = {model_lths.params['treated_post']:.4f} (SE: {model_lths.bse['treated_post']:.4f})")

# High school or more
subset_hs = df[df['educ_less_hs'] == 0]
if len(subset_hs) > 100:
    model_hs = smf.wls('fulltime ~ treated + C(YEAR) + treated_post',
                       data=subset_hs, weights=subset_hs['PERWT']).fit()
    print(f"    HS or more: DiD = {model_hs.params['treated_post']:.4f} (SE: {model_hs.bse['treated_post']:.4f})")

# ============================================================================
# STEP 10: Parallel Trends Check
# ============================================================================
print("\n[Step 10] Parallel trends analysis...")

# Create year-specific treatment effects for pre-period
df_pre = df[df['post'] == 0].copy()
years_pre = sorted(df_pre['YEAR'].unique())
print(f"\n  Pre-treatment years: {years_pre}")

# Using 2006 as reference year
for year in years_pre:
    df_pre[f'year_{year}'] = (df_pre['YEAR'] == year).astype(int)
    df_pre[f'treated_year_{year}'] = df_pre['treated'] * df_pre[f'year_{year}']

# Event study for pre-period
pre_effects = {}
for year in years_pre[1:]:  # Skip reference year
    try:
        model_yr = smf.wls(f'fulltime ~ treated + treated_year_{year}',
                           data=df_pre, weights=df_pre['PERWT']).fit()
        pre_effects[year] = {
            'coef': model_yr.params[f'treated_year_{year}'],
            'se': model_yr.bse[f'treated_year_{year}']
        }
    except:
        pass

print("\n  Pre-treatment year-specific effects (relative to 2006):")
for year, effect in pre_effects.items():
    print(f"    {year}: {effect['coef']:.4f} (SE: {effect['se']:.4f})")

# ============================================================================
# STEP 11: Event Study Analysis
# ============================================================================
print("\n[Step 11] Event study analysis...")

# Create relative time variable (relative to 2012)
df['rel_year'] = df['YEAR'] - 2012

# Create interaction terms for each relative year
rel_years = sorted(df['rel_year'].unique())
print(f"  Relative years in data: {rel_years}")

# Run event study regression
es_results = []
for ry in rel_years:
    if ry != -1:  # Use -1 (2011) as reference
        df[f'treat_ry_{ry}'] = (df['treated'] * (df['rel_year'] == ry)).astype(int)

# Create formula
treat_vars = [f'treat_ry_{ry}' for ry in rel_years if ry != -1]
formula = 'fulltime ~ treated + C(YEAR) + ' + ' + '.join(treat_vars)

try:
    model_es = smf.wls(formula, data=df, weights=df['PERWT']).fit()

    print("\n  Event Study Coefficients:")
    for ry in rel_years:
        if ry != -1:
            var_name = f'treat_ry_{ry}'
            if var_name in model_es.params:
                coef = model_es.params[var_name]
                se = model_es.bse[var_name]
                print(f"    Year {2012 + ry} (t={ry:+d}): {coef:.4f} (SE: {se:.4f})")
                es_results.append({'rel_year': ry, 'coef': coef, 'se': se})
except Exception as e:
    print(f"  Event study error: {e}")

# ============================================================================
# STEP 12: Save Results
# ============================================================================
print("\n[Step 12] Saving results...")

# Prepare results summary
results_summary = {
    'Specification': [],
    'Estimate': [],
    'Std_Error': [],
    'CI_Lower': [],
    'CI_Upper': [],
    'P_Value': [],
    'N': []
}

models_list = [
    ('1. Basic DiD (unweighted)', model1),
    ('2. Weighted DiD', model2),
    ('3. Weighted DiD + Year FE', model3),
    ('4. Weighted DiD + Controls', model4),
    ('5. Weighted DiD + State FE', model5),
    ('6. Preferred: Weighted + Controls + Robust SE', model_robust)
]

for name, model in models_list:
    results_summary['Specification'].append(name)
    results_summary['Estimate'].append(model.params['treated_post'])
    results_summary['Std_Error'].append(model.bse['treated_post'])
    results_summary['CI_Lower'].append(model.conf_int().loc['treated_post', 0])
    results_summary['CI_Upper'].append(model.conf_int().loc['treated_post', 1])
    results_summary['P_Value'].append(model.pvalues['treated_post'])
    results_summary['N'].append(int(model.nobs))

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("  Results saved to results_summary.csv")

# Save descriptive statistics
desc_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_less_hs': 'mean',
    'educ_college': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).reset_index()
desc_stats.to_csv('descriptive_stats.csv', index=False)
print("  Descriptive statistics saved to descriptive_stats.csv")

# Save event study results
if es_results:
    es_df = pd.DataFrame(es_results)
    es_df.to_csv('event_study_results.csv', index=False)
    print("  Event study results saved to event_study_results.csv")

# ============================================================================
# STEP 13: Final Summary
# ============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"\nSample Size: {len(df):,}")
print(f"  Treatment group: {len(df[df['treated']==1]):,}")
print(f"  Control group: {len(df[df['treated']==0]):,}")

print(f"\nPreferred Estimate (Model 6):")
print(f"  Effect of DACA eligibility on full-time employment: {model_robust.params['treated_post']:.4f}")
print(f"  Standard Error (robust): {model_robust.bse['treated_post']:.4f}")
print(f"  95% Confidence Interval: [{model_robust.conf_int().loc['treated_post', 0]:.4f}, {model_robust.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  P-value: {model_robust.pvalues['treated_post']:.4f}")

# Interpretation
effect_pct = model_robust.params['treated_post'] * 100
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {abs(effect_pct):.1f} percentage point")
if effect_pct > 0:
    print(f"  INCREASE in the probability of full-time employment.")
else:
    print(f"  DECREASE in the probability of full-time employment.")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)

# Save full model summary for preferred specification
with open('model_summary.txt', 'w') as f:
    f.write("DACA Replication Analysis - Preferred Model Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(model_robust.summary()))
    f.write("\n\n")
    f.write("Full Model with State Fixed Effects:\n")
    f.write("=" * 60 + "\n\n")
    f.write(str(model5.summary()))

print("\nModel summaries saved to model_summary.txt")

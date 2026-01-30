"""
DACA Replication Analysis
========================
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican individuals born in Mexico.

Treatment: Ages 26-30 at time of DACA (June 15, 2012)
Control: Ages 31-35 at time of DACA
Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION STUDY")
print("Effect of DACA eligibility on full-time employment")
print("=" * 70)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1] Loading data...")

# Read data in chunks for memory efficiency
chunksize = 500000
chunks = []

# We need: YEAR, HISPAN, BPL, CITIZEN, YRIMMIG, BIRTHYR, BIRTHQTR, UHRSWORK, PERWT, SEX, AGE, EDUC, MARST, STATEFIP
usecols = ['YEAR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'BIRTHYR', 'BIRTHQTR', 'UHRSWORK', 'PERWT', 'SEX', 'AGE',
           'EDUC', 'EDUCD', 'MARST', 'STATEFIP', 'EMPSTAT', 'LABFORCE']

for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunksize):
    chunks.append(chunk)
    print(f"  Loaded {len(chunks)} chunks ({len(chunks) * chunksize:,} rows processed)...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total rows loaded: {len(df):,}")

# ============================================================================
# STEP 2: Filter to Target Population
# ============================================================================
print("\n[2] Filtering to target population...")

# Step 2a: Hispanic-Mexican ethnicity
# HISPAN = 1 indicates Mexican
print(f"  Initial sample size: {len(df):,}")

df_mex = df[df['HISPAN'] == 1].copy()
print(f"  After filtering Hispanic-Mexican (HISPAN=1): {len(df_mex):,}")

# Step 2b: Born in Mexico
# BPL = 200 indicates Mexico
df_mex = df_mex[df_mex['BPL'] == 200]
print(f"  After filtering birthplace Mexico (BPL=200): {len(df_mex):,}")

# Step 2c: Non-citizen (likely undocumented for DACA purposes)
# CITIZEN = 3 indicates "Not a citizen"
# Per instructions: assume non-citizens without papers are undocumented
df_mex = df_mex[df_mex['CITIZEN'] == 3]
print(f"  After filtering non-citizens (CITIZEN=3): {len(df_mex):,}")

# Step 2d: Arrived in US before age 16
# Calculate age at immigration: YRIMMIG - BIRTHYR
# Note: YRIMMIG = 0 means N/A, we exclude these
df_mex = df_mex[df_mex['YRIMMIG'] > 0]
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex = df_mex[df_mex['age_at_immig'] < 16]
print(f"  After filtering arrived before age 16: {len(df_mex):,}")

# Step 2e: Present in US since June 15, 2007 (continuous residence)
# Immigration year must be <= 2007
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007]
print(f"  After filtering continuous residence since 2007: {len(df_mex):,}")

# ============================================================================
# STEP 3: Calculate age at DACA implementation (June 15, 2012)
# ============================================================================
print("\n[3] Calculating age at DACA implementation...")

# DACA implementation date: June 15, 2012
# Age calculation: For those born in Q1 (Jan-Mar) or Q2 (Apr-Jun),
# they would have had their birthday by June 15, 2012
# Q1 = Jan-Mar (birthday before June 15)
# Q2 = Apr-Jun (some before, some after - we'll assume before for simplicity)
# Q3 = Jul-Sep (birthday after June 15 in same year)
# Q4 = Oct-Dec (birthday after June 15 in same year)

# Age at June 15, 2012:
# - If birthday before June 15 (Q1, Q2): age = 2012 - birthyear
# - If birthday after June 15 (Q3, Q4): age = 2012 - birthyear - 1

df_mex['age_at_daca'] = np.where(
    df_mex['BIRTHQTR'].isin([1, 2]),
    2012 - df_mex['BIRTHYR'],
    2012 - df_mex['BIRTHYR'] - 1
)

print(f"  Age at DACA distribution:")
print(df_mex['age_at_daca'].describe())

# ============================================================================
# STEP 4: Define Treatment and Control Groups
# ============================================================================
print("\n[4] Defining treatment and control groups...")

# Treatment group: Ages 26-30 at DACA (born ~1982-1986)
# Control group: Ages 31-35 at DACA (born ~1977-1981)

# The control group would have been eligible if not for the age cutoff (age <31)
# DACA required not yet having 31st birthday as of June 15, 2012

df_mex['treated'] = ((df_mex['age_at_daca'] >= 26) & (df_mex['age_at_daca'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_at_daca'] >= 31) & (df_mex['age_at_daca'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()
print(f"  Sample after selecting age groups: {len(df_analysis):,}")
print(f"  Treatment group (ages 26-30): {(df_analysis['treated'] == 1).sum():,}")
print(f"  Control group (ages 31-35): {(df_analysis['control'] == 1).sum():,}")

# ============================================================================
# STEP 5: Define Time Periods
# ============================================================================
print("\n[5] Defining time periods...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA, excluding 2012)
# 2012 is excluded because DACA was implemented mid-year (June)

df_analysis = df_analysis[df_analysis['YEAR'] != 2012]
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"  Sample after excluding 2012: {len(df_analysis):,}")
print(f"  Pre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"  Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# ============================================================================
# STEP 6: Create Outcome Variable
# ============================================================================
print("\n[6] Creating outcome variable...")

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK = usual hours worked per week
# UHRSWORK = 0 indicates N/A or not working

df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"  Full-time employment rate: {df_analysis['fulltime'].mean():.3f}")
print(f"  Full-time among treated: {df_analysis[df_analysis['treated']==1]['fulltime'].mean():.3f}")
print(f"  Full-time among control: {df_analysis[df_analysis['control']==1]['fulltime'].mean():.3f}")

# ============================================================================
# STEP 7: Create Covariates
# ============================================================================
print("\n[7] Creating covariates...")

# Female indicator
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Married indicator (MARST = 1 or 2 indicates married)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_somecoll'] = ((df_analysis['EDUC'] >= 7) & (df_analysis['EDUC'] <= 9)).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

# Age at survey (for age controls)
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

print(f"  Female share: {df_analysis['female'].mean():.3f}")
print(f"  Married share: {df_analysis['married'].mean():.3f}")

# ============================================================================
# STEP 8: Summary Statistics
# ============================================================================
print("\n[8] Summary statistics...")

print("\n  --- Pre-period summary ---")
pre = df_analysis[df_analysis['post'] == 0]
print(f"  Treatment group full-time rate: {pre[pre['treated']==1]['fulltime'].mean():.4f}")
print(f"  Control group full-time rate: {pre[pre['control']==1]['fulltime'].mean():.4f}")

print("\n  --- Post-period summary ---")
post_df = df_analysis[df_analysis['post'] == 1]
print(f"  Treatment group full-time rate: {post_df[post_df['treated']==1]['fulltime'].mean():.4f}")
print(f"  Control group full-time rate: {post_df[post_df['control']==1]['fulltime'].mean():.4f}")

# ============================================================================
# STEP 9: Difference-in-Differences Analysis
# ============================================================================
print("\n[9] Difference-in-Differences Analysis...")

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Model 1: Basic DiD
print("\n  Model 1: Basic DiD")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                 data=df_analysis, weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n  Model 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + age_sq',
                 data=df_analysis, weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: DiD with education controls
print("\n  Model 3: DiD with demographic and education controls")
model3 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college',
                 data=df_analysis, weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
print(results3.summary().tables[1])

# Model 4: DiD with state fixed effects
print("\n  Model 4: DiD with state fixed effects")
df_analysis['state'] = df_analysis['STATEFIP'].astype(str)
model4 = smf.wls('fulltime ~ treated + post + treated_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college + C(state)',
                 data=df_analysis, weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')
# Print only key coefficients
print(f"  treated_post coefficient: {results4.params['treated_post']:.4f}")
print(f"  treated_post SE: {results4.bse['treated_post']:.4f}")
print(f"  treated_post t-stat: {results4.tvalues['treated_post']:.4f}")
print(f"  treated_post p-value: {results4.pvalues['treated_post']:.4f}")

# Model 5: DiD with year fixed effects
print("\n  Model 5: DiD with year and state fixed effects")
df_analysis['year_str'] = df_analysis['YEAR'].astype(str)
model5 = smf.wls('fulltime ~ treated + treated_post + female + married + age + age_sq + educ_hs + educ_somecoll + educ_college + C(state) + C(year_str)',
                 data=df_analysis, weights=df_analysis['PERWT'])
results5 = model5.fit(cov_type='HC1')
print(f"  treated_post coefficient: {results5.params['treated_post']:.4f}")
print(f"  treated_post SE: {results5.bse['treated_post']:.4f}")
print(f"  treated_post t-stat: {results5.tvalues['treated_post']:.4f}")
print(f"  treated_post p-value: {results5.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 10: Robustness Checks
# ============================================================================
print("\n[10] Robustness Checks...")

# 10a: Linear Probability Model vs Logit
print("\n  10a: Logistic regression")
try:
    logit_model = smf.glm('fulltime ~ treated + post + treated_post + female + married + age + age_sq',
                          data=df_analysis, family=sm.families.Binomial(),
                          freq_weights=df_analysis['PERWT'].values)
    logit_results = logit_model.fit()
    # Calculate marginal effect at means
    print(f"  Logit coefficient for treated_post: {logit_results.params['treated_post']:.4f}")
    print(f"  Logit SE: {logit_results.bse['treated_post']:.4f}")
except Exception as e:
    print(f"  Logit estimation failed: {e}")

# 10b: Subgroup analysis by gender
print("\n  10b: Subgroup analysis by gender")
for sex, label in [(0, 'Male'), (1, 'Female')]:
    sub_df = df_analysis[df_analysis['female'] == sex]
    sub_model = smf.wls('fulltime ~ treated + post + treated_post + married + age + age_sq',
                        data=sub_df, weights=sub_df['PERWT'])
    sub_results = sub_model.fit(cov_type='HC1')
    print(f"  {label}: treated_post = {sub_results.params['treated_post']:.4f} (SE: {sub_results.bse['treated_post']:.4f})")

# ============================================================================
# STEP 11: Event Study / Parallel Trends
# ============================================================================
print("\n[11] Event Study Analysis...")

# Create year dummies and interactions
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for yr in years:
    df_analysis[f'year_{yr}'] = (df_analysis['YEAR'] == yr).astype(int)
    df_analysis[f'treated_year_{yr}'] = df_analysis['treated'] * df_analysis[f'year_{yr}']

# Reference year: 2011 (last pre-treatment year)
year_vars = ' + '.join([f'treated_year_{yr}' for yr in years if yr != 2011])
event_formula = f'fulltime ~ treated + {year_vars} + female + married + age + age_sq + C(year_str)'

event_model = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT'])
event_results = event_model.fit(cov_type='HC1')

print("\n  Event study coefficients (relative to 2011):")
for yr in years:
    if yr != 2011:
        var = f'treated_year_{yr}'
        coef = event_results.params.get(var, np.nan)
        se = event_results.bse.get(var, np.nan)
        print(f"  Year {yr}: {coef:.4f} (SE: {se:.4f})")

# ============================================================================
# STEP 12: Final Results Summary
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\nPreferred estimate (Model 3 - DiD with controls):")
print(f"  Effect size: {results3.params['treated_post']:.4f}")
print(f"  Standard error: {results3.bse['treated_post']:.4f}")
print(f"  95% CI: [{results3.conf_int().loc['treated_post', 0]:.4f}, {results3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  t-statistic: {results3.tvalues['treated_post']:.4f}")
print(f"  p-value: {results3.pvalues['treated_post']:.4f}")
print(f"  Sample size: {int(results3.nobs):,}")

# Save detailed results
print("\n[12] Saving results...")

# Create results dataframe
results_df = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'With Education', 'State FE', 'Year+State FE'],
    'Coefficient': [results1.params['treated_post'], results2.params['treated_post'],
                   results3.params['treated_post'], results4.params['treated_post'],
                   results5.params['treated_post']],
    'Std_Error': [results1.bse['treated_post'], results2.bse['treated_post'],
                 results3.bse['treated_post'], results4.bse['treated_post'],
                 results5.bse['treated_post']],
    'CI_Lower': [results1.conf_int().loc['treated_post', 0], results2.conf_int().loc['treated_post', 0],
                results3.conf_int().loc['treated_post', 0], results4.conf_int().loc['treated_post', 0],
                results5.conf_int().loc['treated_post', 0]],
    'CI_Upper': [results1.conf_int().loc['treated_post', 1], results2.conf_int().loc['treated_post', 1],
                results3.conf_int().loc['treated_post', 1], results4.conf_int().loc['treated_post', 1],
                results5.conf_int().loc['treated_post', 1]],
    'N': [int(results1.nobs), int(results2.nobs), int(results3.nobs),
          int(results4.nobs), int(results5.nobs)]
})

results_df.to_csv('regression_results.csv', index=False)
print("  Saved regression_results.csv")

# Summary statistics
summary_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'age': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("  Saved summary_statistics.csv")

# Event study results
event_coefs = []
for yr in years:
    if yr != 2011:
        var = f'treated_year_{yr}'
        event_coefs.append({
            'Year': yr,
            'Coefficient': event_results.params.get(var, np.nan),
            'SE': event_results.bse.get(var, np.nan)
        })
    else:
        event_coefs.append({'Year': yr, 'Coefficient': 0, 'SE': 0})

event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_results.csv', index=False)
print("  Saved event_study_results.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

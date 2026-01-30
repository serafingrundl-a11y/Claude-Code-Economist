"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# =============================================================================
# STEP 1: LOAD AND INITIAL FILTER
# =============================================================================
print("\n[1] Loading data...")

# Read in chunks to manage memory, filtering early
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int32',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'MARST': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'UHRSWORK': 'int8',
    'STATEFIP': 'int8',
    'FAMSIZE': 'int8',
    'NCHILD': 'int8'
}

# Load relevant columns
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
               'EMPSTAT', 'UHRSWORK', 'STATEFIP', 'FAMSIZE', 'NCHILD']

print("Loading full dataset (this may take a few minutes)...")
df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype={c: dtypes.get(c, None) for c in cols_needed})
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: FILTER TO TARGET POPULATION
# =============================================================================
print("\n[2] Filtering to target population...")

# Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"After filtering Hispanic-Mexican: {len(df):,}")

# Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"After filtering born in Mexico: {len(df):,}")

# Non-citizens (CITIZEN == 3)
df = df[df['CITIZEN'] == 3]
print(f"After filtering non-citizens: {len(df):,}")

# Immigrated to US (YRIMMIG > 0)
df = df[df['YRIMMIG'] > 0]
print(f"After filtering with valid immigration year: {len(df):,}")

# =============================================================================
# STEP 3: CALCULATE AGE AS OF JUNE 15, 2012
# =============================================================================
print("\n[3] Calculating age as of June 15, 2012...")

# Age as of June 15, 2012
# If born in Q1 (Jan-Mar) or Q2 (Apr-Jun before June 15), they would have had their birthday
# Q1 = Jan-Feb-Mar (birthday by June 15)
# Q2 = Apr-May-Jun (could be before or after June 15, treat conservatively as had birthday)
# Q3 = Jul-Aug-Sep (birthday not yet by June 15)
# Q4 = Oct-Nov-Dec (birthday not yet by June 15)

# Calculate base age as of 2012
df['age_june2012'] = 2012 - df['BIRTHYR']

# Adjust for those whose birthday hasn't occurred by June 15
# Q3 and Q4 births haven't had their birthday yet by June 15
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] = df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] - 1

print(f"Age as of June 15, 2012 calculated.")
print(f"Age distribution in sample:")
print(df['age_june2012'].describe())

# =============================================================================
# STEP 4: APPLY DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n[4] Applying DACA eligibility criteria...")

# 1. Arrived before 16th birthday
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"After filtering arrived before age 16: {len(df):,}")

# 2. Continuous residence since June 15, 2007 (arrived by 2007)
df = df[df['YRIMMIG'] <= 2007]
print(f"After filtering arrived by 2007: {len(df):,}")

# 3. Filter to treatment group (26-30) and control group (31-35) as of June 15, 2012
df = df[(df['age_june2012'] >= 26) & (df['age_june2012'] <= 35)]
print(f"After filtering to ages 26-35 as of June 15, 2012: {len(df):,}")

# =============================================================================
# STEP 5: CREATE TREATMENT AND CONTROL GROUPS
# =============================================================================
print("\n[5] Creating treatment and control groups...")

# Treatment: Ages 26-30 as of June 15, 2012 (DACA-eligible by age)
# Control: Ages 31-35 as of June 15, 2012 (just missed DACA by age)
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)

print(f"Treatment group (ages 26-30): {df['treated'].sum():,}")
print(f"Control group (ages 31-35): {(df['treated'] == 0).sum():,}")

# =============================================================================
# STEP 6: CREATE TIME PERIOD INDICATORS
# =============================================================================
print("\n[6] Creating time period indicators...")

# Pre-period: 2006-2011
# Post-period: 2013-2016
# Exclude 2012 due to implementation timing ambiguity

df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

df['post'] = (df['YEAR'] >= 2013).astype(int)

print(f"Pre-period observations (2006-2011): {(df['post'] == 0).sum():,}")
print(f"Post-period observations (2013-2016): {(df['post'] == 1).sum():,}")

# =============================================================================
# STEP 7: CREATE OUTCOME VARIABLE
# =============================================================================
print("\n[7] Creating outcome variable...")

# Full-time employment: usually working 35+ hours per week
# Only for those who are employed (EMPSTAT == 1)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['employed'] == 1)).astype(int)

print(f"Employment rate: {df['employed'].mean():.4f}")
print(f"Full-time employment rate: {df['fulltime'].mean():.4f}")

# =============================================================================
# STEP 8: CREATE ADDITIONAL VARIABLES
# =============================================================================
print("\n[8] Creating additional variables...")

# Interaction term for DiD
df['treated_post'] = df['treated'] * df['post']

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories (based on EDUC)
# 0-5: Less than high school
# 6: High school
# 7-9: Some college
# 10-11: College+
df['educ_lths'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)
df['educ_coll'] = (df['EDUC'] >= 10).astype(int)

# Age at survey time for controls
df['age_sq'] = df['AGE'] ** 2

print("Variables created successfully.")

# =============================================================================
# STEP 9: SUMMARY STATISTICS
# =============================================================================
print("\n[9] Summary statistics...")
print("\n" + "="*70)
print("SUMMARY STATISTICS BY GROUP AND PERIOD")
print("="*70)

# Create group labels
df['group'] = df['treated'].map({1: 'Treatment (26-30)', 0: 'Control (31-35)'})
df['period'] = df['post'].map({1: 'Post (2013-2016)', 0: 'Pre (2006-2011)'})

# Summary by group and period
summary = df.groupby(['group', 'period']).agg({
    'fulltime': ['mean', 'std'],
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print(summary)

# Sample sizes
print("\n" + "-"*70)
print("Sample sizes (unweighted):")
sample_sizes = df.groupby(['group', 'period']).size()
print(sample_sizes)

# Weighted sample sizes
print("\nWeighted sample sizes:")
weighted_sizes = df.groupby(['group', 'period'])['PERWT'].sum()
print(weighted_sizes)

# =============================================================================
# STEP 10: DIFFERENCE-IN-DIFFERENCES ESTIMATION
# =============================================================================
print("\n" + "="*70)
print("DIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("="*70)

# Calculate simple 2x2 DiD
print("\n[10a] Simple 2x2 DiD calculation:")
means = df.groupby(['treated', 'post'])['fulltime'].mean()
print("\nMean full-time employment by group and period:")
print(means)

# DiD estimate
treat_diff = means[(1, 1)] - means[(1, 0)]
control_diff = means[(0, 1)] - means[(0, 0)]
did_simple = treat_diff - control_diff

print(f"\nTreatment group change (post-pre): {treat_diff:.4f}")
print(f"Control group change (post-pre): {control_diff:.4f}")
print(f"Difference-in-Differences estimate: {did_simple:.4f}")

# =============================================================================
# STEP 11: REGRESSION-BASED DID
# =============================================================================
print("\n[10b] Regression-based DiD (without covariates):")

# Basic DiD regression
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df, weights=df['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results1.summary())

print("\n[10c] Regression-based DiD (with covariates):")

# DiD regression with controls
model2 = smf.wls('fulltime ~ treated + post + treated_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                  data=df, weights=df['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(results2.summary())

# =============================================================================
# STEP 12: YEAR-BY-YEAR EFFECTS (EVENT STUDY)
# =============================================================================
print("\n" + "="*70)
print("EVENT STUDY / YEAR-BY-YEAR EFFECTS")
print("="*70)

# Create year dummies and interactions
df['year_factor'] = df['YEAR'].astype('category')

# Use 2011 as reference year
years_except_ref = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]

for year in years_except_ref:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'treated_year_{year}'] = df['treated'] * df[f'year_{year}']

# Year-specific effects
year_vars = ' + '.join([f'year_{y}' for y in years_except_ref])
interaction_vars = ' + '.join([f'treated_year_{y}' for y in years_except_ref])

formula_event = f'fulltime ~ treated + {year_vars} + {interaction_vars} + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll'

model3 = smf.wls(formula_event, data=df, weights=df['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nYear-by-year treatment effects (event study coefficients):")
print("-"*50)
for year in years_except_ref:
    coef = results3.params[f'treated_year_{year}']
    se = results3.bse[f'treated_year_{year}']
    pval = results3.pvalues[f'treated_year_{year}']
    print(f"Year {year}: coef = {coef:.4f}, SE = {se:.4f}, p = {pval:.4f}")

# =============================================================================
# STEP 13: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*70)
print("ROBUSTNESS CHECKS")
print("="*70)

# A. Alternative age bandwidth (24-32 vs 33-37)
print("\n[13a] Alternative age bandwidth (narrower: 27-29 vs 32-34):")
df_narrow = df[(df['age_june2012'] >= 27) & (df['age_june2012'] <= 34)]
df_narrow['treated_narrow'] = ((df_narrow['age_june2012'] >= 27) & (df_narrow['age_june2012'] <= 29)).astype(int)
df_narrow['treated_post_narrow'] = df_narrow['treated_narrow'] * df_narrow['post']

model_narrow = smf.wls('fulltime ~ treated_narrow + post + treated_post_narrow + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                        data=df_narrow, weights=df_narrow['PERWT'])
results_narrow = model_narrow.fit(cov_type='cluster', cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"Narrower bandwidth DiD estimate: {results_narrow.params['treated_post_narrow']:.4f}")
print(f"SE: {results_narrow.bse['treated_post_narrow']:.4f}")
print(f"95% CI: [{results_narrow.conf_int().loc['treated_post_narrow', 0]:.4f}, {results_narrow.conf_int().loc['treated_post_narrow', 1]:.4f}]")

# B. Unweighted regression
print("\n[13b] Unweighted regression:")
model_unwtd = smf.ols('fulltime ~ treated + post + treated_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                       data=df)
results_unwtd = model_unwtd.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"Unweighted DiD estimate: {results_unwtd.params['treated_post']:.4f}")
print(f"SE: {results_unwtd.bse['treated_post']:.4f}")

# C. Employment as outcome (instead of full-time)
print("\n[13c] Employment (any) as outcome:")
model_emp = smf.wls('employed ~ treated + post + treated_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                     data=df, weights=df['PERWT'])
results_emp = model_emp.fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"Employment DiD estimate: {results_emp.params['treated_post']:.4f}")
print(f"SE: {results_emp.bse['treated_post']:.4f}")

# D. By gender
print("\n[13d] Heterogeneity by gender:")
df_male = df[df['female'] == 0]
df_female = df[df['female'] == 1]

model_male = smf.wls('fulltime ~ treated + post + treated_post + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                      data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})

model_female = smf.wls('fulltime ~ treated + post + treated_post + married + AGE + age_sq + educ_hs + educ_somecoll + educ_coll',
                        data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})

print(f"Male DiD estimate: {results_male.params['treated_post']:.4f} (SE: {results_male.bse['treated_post']:.4f})")
print(f"Female DiD estimate: {results_female.params['treated_post']:.4f} (SE: {results_female.bse['treated_post']:.4f})")

# =============================================================================
# STEP 14: SAVE KEY RESULTS
# =============================================================================
print("\n" + "="*70)
print("KEY RESULTS SUMMARY")
print("="*70)

# Preferred estimate (with covariates, clustered SE)
preferred_coef = results2.params['treated_post']
preferred_se = results2.bse['treated_post']
preferred_ci = results2.conf_int().loc['treated_post']
n_total = len(df)

print(f"\nPREFERRED ESTIMATE:")
print(f"  Effect size: {preferred_coef:.4f}")
print(f"  Standard error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  p-value: {results2.pvalues['treated_post']:.4f}")
print(f"  Sample size: {n_total:,}")

# Save results for report
results_dict = {
    'preferred_estimate': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_lower': preferred_ci[0],
    'preferred_ci_upper': preferred_ci[1],
    'preferred_pvalue': results2.pvalues['treated_post'],
    'sample_size': n_total,
    'simple_did': did_simple,
    'treat_pre_mean': means[(1, 0)],
    'treat_post_mean': means[(1, 1)],
    'control_pre_mean': means[(0, 0)],
    'control_post_mean': means[(0, 1)]
}

# Save summary stats for tables
summary_stats = df.groupby(['treated', 'post']).agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'educ_lths': 'mean',
    'educ_hs': 'mean',
    'educ_somecoll': 'mean',
    'educ_coll': 'mean',
    'PERWT': ['sum', 'count']
}).round(4)

summary_stats.to_csv('summary_stats.csv')
print("\nSummary statistics saved to summary_stats.csv")

# Save year-by-year effects for event study plot
event_study_results = []
for year in years_except_ref:
    event_study_results.append({
        'year': year,
        'coef': results3.params[f'treated_year_{year}'],
        'se': results3.bse[f'treated_year_{year}'],
        'ci_lower': results3.conf_int().loc[f'treated_year_{year}', 0],
        'ci_upper': results3.conf_int().loc[f'treated_year_{year}', 1]
    })
# Add reference year (2011)
event_study_results.append({
    'year': 2011,
    'coef': 0,
    'se': 0,
    'ci_lower': 0,
    'ci_upper': 0
})

event_df = pd.DataFrame(event_study_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# =============================================================================
# STEP 15: CREATE VISUALIZATION DATA
# =============================================================================
print("\n[15] Creating data for visualizations...")

# Trends by year and group
trends = df.groupby(['YEAR', 'treated']).agg({
    'fulltime': 'mean',
    'PERWT': 'sum'
}).reset_index()
trends.columns = ['year', 'treated', 'fulltime_rate', 'weighted_n']
trends['group'] = trends['treated'].map({1: 'Treatment (26-30)', 0: 'Control (31-35)'})
trends.to_csv('trends_by_year.csv', index=False)
print("Trend data saved to trends_by_year.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

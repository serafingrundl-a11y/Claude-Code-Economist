"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico.

Treatment group: Ages 26-30 as of June 15, 2012
Control group: Ages 31-35 as of June 15, 2012
Outcome: Full-time employment (usually working 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = "C:/Users/seraf/DACA Results Task 2/replication_36/"

print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)

# ============================================================
# STEP 1: Load and examine data
# ============================================================
print("\n[STEP 1] Loading data...")

# Read the CSV file
df = pd.read_csv(OUTPUT_DIR + "data/data.csv")

print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
# STEP 2: Filter for Hispanic-Mexican individuals born in Mexico
# ============================================================
print("\n[STEP 2] Filtering for Hispanic-Mexican individuals born in Mexico...")

# HISPAN == 1 means Mexican Hispanic origin
# BPL == 200 means born in Mexico (from data dictionary)
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()

print(f"Observations after filtering for Hispanic-Mexican born in Mexico: {len(df_mex):,}")

# ============================================================
# STEP 3: Filter for non-citizens (proxy for undocumented)
# ============================================================
print("\n[STEP 3] Filtering for non-citizens (proxy for undocumented status)...")

# CITIZEN == 3 means "Not a citizen"
# Per instructions: "Assume anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes."
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()

print(f"Observations after filtering for non-citizens: {len(df_mex):,}")

# ============================================================
# STEP 4: Determine DACA eligibility based on age at implementation
# ============================================================
print("\n[STEP 4] Determining DACA eligibility based on age criteria...")

# DACA was implemented June 15, 2012
# Eligibility requirement: "Arrived unlawfully in the US before their 16th birthday"
# and "Had not yet had their 31st birthday as of June 15, 2012"

# For each survey year, calculate what the person's age was on June 15, 2012
# Age on June 15, 2012 = 2012 - BIRTHYR (approximate, ignoring birth month)
# More precisely, we need to consider birth quarter

# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For someone surveyed in year Y with birth year BY:
# Their age on June 15, 2012 depends on their birth quarter
# If born Q1 or Q2 (Jan-Jun): age on June 15, 2012 = 2012 - BY
# If born Q3 or Q4 (Jul-Dec): age on June 15, 2012 = 2012 - BY - 1 (hadn't had birthday yet)

def calc_age_june_2012(row):
    """Calculate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    if birth_qtr in [1, 2]:  # Born Jan-Jun, already had birthday by June 15
        return 2012 - birth_year
    else:  # Born Jul-Dec, hadn't had birthday yet by June 15
        return 2012 - birth_year - 1

df_mex['age_june_2012'] = df_mex.apply(calc_age_june_2012, axis=1)

print(f"Age distribution as of June 15, 2012:")
print(df_mex['age_june_2012'].describe())

# ============================================================
# STEP 5: Additional DACA eligibility criteria
# ============================================================
print("\n[STEP 5] Applying additional DACA eligibility criteria...")

# Requirement: Arrived before 16th birthday
# We need: year of immigration - birth year < 16
# YRIMMIG: Year of immigration

# Filter for valid immigration year data (exclude 0 which is N/A)
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()

# Calculate age at immigration
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Filter: Must have arrived before 16th birthday
df_mex = df_mex[df_mex['age_at_immig'] < 16].copy()

print(f"Observations after filtering for arrival before age 16: {len(df_mex):,}")

# Requirement: "Lived continuously in the US since June 15, 2007"
# This implies immigration year must be 2007 or earlier
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()

print(f"Observations after filtering for continuous US residence since 2007: {len(df_mex):,}")

# ============================================================
# STEP 6: Define treatment and control groups
# ============================================================
print("\n[STEP 6] Defining treatment and control groups...")

# Treatment group: Ages 26-30 as of June 15, 2012
# Control group: Ages 31-35 as of June 15, 2012

df_mex['treated'] = ((df_mex['age_june_2012'] >= 26) & (df_mex['age_june_2012'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_june_2012'] >= 31) & (df_mex['age_june_2012'] <= 35)).astype(int)

# Keep only those in treatment or control groups
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()

print(f"Observations in analysis sample: {len(df_analysis):,}")
print(f"Treatment group (age 26-30): {df_analysis['treated'].sum():,}")
print(f"Control group (age 31-35): {(df_analysis['control'] == 1).sum():,}")

# ============================================================
# STEP 7: Define outcome variable (full-time employment)
# ============================================================
print("\n[STEP 7] Defining outcome variable (full-time employment)...")

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A)
# EMPSTAT: Employment status (1 = Employed)

# Full-time employed = UHRSWORK >= 35 and EMPSTAT == 1
df_analysis['fulltime'] = ((df_analysis['UHRSWORK'] >= 35) & (df_analysis['EMPSTAT'] == 1)).astype(int)

print(f"Full-time employment rate: {df_analysis['fulltime'].mean():.3f}")

# ============================================================
# STEP 8: Define time periods
# ============================================================
print("\n[STEP 8] Defining time periods...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA - as specified in instructions)
# Note: 2012 is excluded because we cannot distinguish pre/post implementation

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Exclude 2012 observations (ambiguous timing)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()

print(f"Observations after excluding 2012: {len(df_analysis):,}")
print(f"Pre-period (2006-2011) observations: {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016) observations: {(df_analysis['post'] == 1).sum():,}")

# ============================================================
# STEP 9: Summary statistics
# ============================================================
print("\n[STEP 9] Summary statistics...")

# Summary by treatment and time period
summary_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'AGE': 'mean',
    'SEX': 'mean',
    'EDUC': 'mean',
    'PERWT': 'sum'
}).round(3)

print("\nSummary Statistics by Group and Period:")
print(summary_stats)

# Save to CSV for the report
summary_df = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean(),  # Proportion female
    'PERWT': 'sum'
}).round(4)
summary_df.to_csv(OUTPUT_DIR + "summary_stats.csv")

# ============================================================
# STEP 10: Simple difference-in-differences
# ============================================================
print("\n[STEP 10] Simple difference-in-differences calculation...")

# Calculate means for 2x2 table
means = df_analysis.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

print("\nMean Full-Time Employment Rates (weighted):")
print(f"                    Pre-DACA    Post-DACA")
print(f"Treatment (26-30):  {means.loc[1, 0]:.4f}      {means.loc[1, 1]:.4f}")
print(f"Control (31-35):    {means.loc[0, 0]:.4f}      {means.loc[0, 1]:.4f}")

# DiD calculation
did_simple = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# ============================================================
# STEP 11: Regression-based DiD
# ============================================================
print("\n[STEP 11] Regression-based difference-in-differences...")

# Create interaction term
df_analysis['did'] = df_analysis['treated'] * df_analysis['post']

# Basic DiD regression without covariates
model1 = smf.wls('fulltime ~ treated + post + did',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nModel 1: Basic DiD (no covariates)")
print(model1.summary().tables[1])

# ============================================================
# STEP 12: DiD with covariates
# ============================================================
print("\n[STEP 12] DiD with demographic covariates...")

# Add demographic controls
# SEX: 1=Male, 2=Female
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Create age controls (centered)
df_analysis['age_centered'] = df_analysis['AGE'] - df_analysis['AGE'].mean()
df_analysis['age_sq'] = df_analysis['age_centered'] ** 2

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUCD'] >= 62).astype(int)  # HS diploma or higher
df_analysis['educ_coll'] = (df_analysis['EDUCD'] >= 81).astype(int)  # Some college or higher

# Marital status
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)  # Married

# Number of children
df_analysis['has_children'] = (df_analysis['NCHILD'] > 0).astype(int)

# Model with covariates
model2 = smf.wls('fulltime ~ treated + post + did + female + age_centered + age_sq + educ_hs + married + has_children',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nModel 2: DiD with demographic covariates")
print(model2.summary().tables[1])

# ============================================================
# STEP 13: DiD with year and state fixed effects
# ============================================================
print("\n[STEP 13] DiD with year fixed effects...")

# Create year dummies
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])

# Model with year fixed effects
model3 = smf.wls('fulltime ~ treated + C(YEAR) + did + female + age_centered + age_sq + educ_hs + married + has_children',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nModel 3: DiD with year fixed effects")
print(f"DiD coefficient: {model3.params['did']:.4f}")
print(f"Std. Error: {model3.bse['did']:.4f}")
print(f"t-statistic: {model3.tvalues['did']:.3f}")
print(f"p-value: {model3.pvalues['did']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['did', 0]:.4f}, {model3.conf_int().loc['did', 1]:.4f}]")

# ============================================================
# STEP 14: Robustness checks - by gender
# ============================================================
print("\n[STEP 14] Heterogeneity analysis by gender...")

# Males
df_male = df_analysis[df_analysis['female'] == 0]
model_male = smf.wls('fulltime ~ treated + post + did',
                      data=df_male,
                      weights=df_male['PERWT']).fit(cov_type='HC1')

# Females
df_female = df_analysis[df_analysis['female'] == 1]
model_female = smf.wls('fulltime ~ treated + post + did',
                        data=df_female,
                        weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"\nMales - DiD estimate: {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f})")
print(f"Females - DiD estimate: {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f})")

# ============================================================
# STEP 15: Event study / Dynamic DiD
# ============================================================
print("\n[STEP 15] Event study analysis...")

# Create year-specific treatment effects (relative to 2011)
df_analysis['year_x_treat'] = df_analysis['YEAR'].astype(str) + '_' + df_analysis['treated'].astype(str)

# Create interactions for each year (excluding 2011 as reference)
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'treat_x_{year}'] = ((df_analysis['YEAR'] == year) & (df_analysis['treated'] == 1)).astype(int)

event_formula = 'fulltime ~ treated + C(YEAR) + ' + ' + '.join([f'treat_x_{y}' for y in years])
model_event = smf.wls(event_formula,
                       data=df_analysis,
                       weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in years:
    coef_name = f'treat_x_{year}'
    print(f"  {year}: {model_event.params[coef_name]:.4f} (SE: {model_event.bse[coef_name]:.4f})")

# ============================================================
# STEP 16: Save results
# ============================================================
print("\n[STEP 16] Saving results...")

# Create results dictionary
results = {
    'Simple DiD': {
        'coefficient': did_simple,
        'se': np.nan,
        'n': len(df_analysis)
    },
    'Model 1 (Basic)': {
        'coefficient': model1.params['did'],
        'se': model1.bse['did'],
        'n': int(model1.nobs)
    },
    'Model 2 (Covariates)': {
        'coefficient': model2.params['did'],
        'se': model2.bse['did'],
        'n': int(model2.nobs)
    },
    'Model 3 (Year FE)': {
        'coefficient': model3.params['did'],
        'se': model3.bse['did'],
        'n': int(model3.nobs)
    }
}

# Create results DataFrame
results_df = pd.DataFrame(results).T
results_df.to_csv(OUTPUT_DIR + "regression_results.csv")

# Event study results
event_results = []
for year in years:
    coef_name = f'treat_x_{year}'
    event_results.append({
        'year': year,
        'coefficient': model_event.params[coef_name],
        'se': model_event.bse[coef_name],
        'ci_lower': model_event.conf_int().loc[coef_name, 0],
        'ci_upper': model_event.conf_int().loc[coef_name, 1]
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv(OUTPUT_DIR + "event_study_results.csv", index=False)

# Save detailed model summaries
with open(OUTPUT_DIR + "model_summaries.txt", 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("DACA REPLICATION ANALYSIS - DETAILED MODEL RESULTS\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL 1: Basic DiD\n")
    f.write("-" * 40 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: DiD with Covariates\n")
    f.write("-" * 40 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 3: DiD with Year Fixed Effects\n")
    f.write("-" * 40 + "\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("EVENT STUDY MODEL\n")
    f.write("-" * 40 + "\n")
    f.write(str(model_event.summary()) + "\n\n")

# ============================================================
# STEP 17: Create summary statistics table
# ============================================================
print("\n[STEP 17] Creating detailed summary statistics...")

# Detailed summary statistics
def weighted_stats(group, var, weight):
    """Calculate weighted mean and std"""
    w_mean = np.average(group[var], weights=group[weight])
    w_var = np.average((group[var] - w_mean)**2, weights=group[weight])
    w_std = np.sqrt(w_var)
    return pd.Series({'mean': w_mean, 'std': w_std, 'n': len(group)})

# Summary by group
detailed_summary = []
for (treat, post), group in df_analysis.groupby(['treated', 'post']):
    row = {
        'treated': treat,
        'post': post,
        'n_obs': len(group),
        'weighted_n': group['PERWT'].sum(),
        'fulltime_mean': np.average(group['fulltime'], weights=group['PERWT']),
        'age_mean': np.average(group['AGE'], weights=group['PERWT']),
        'female_prop': np.average(group['female'], weights=group['PERWT']),
        'married_prop': np.average(group['married'], weights=group['PERWT']),
        'educ_hs_prop': np.average(group['educ_hs'], weights=group['PERWT']),
        'has_children_prop': np.average(group['has_children'], weights=group['PERWT'])
    }
    detailed_summary.append(row)

detailed_df = pd.DataFrame(detailed_summary)
detailed_df.to_csv(OUTPUT_DIR + "detailed_summary.csv", index=False)
print(detailed_df.to_string())

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print("\n--- PREFERRED ESTIMATE (Model 3: DiD with Year FE and Covariates) ---")
print(f"Effect Size: {model3.params['did']:.4f}")
print(f"Standard Error: {model3.bse['did']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['did', 0]:.4f}, {model3.conf_int().loc['did', 1]:.4f}]")
print(f"Sample Size: {int(model3.nobs):,}")
print(f"p-value: {model3.pvalues['did']:.4f}")

print("\n--- ALL DiD ESTIMATES ---")
print(f"Simple DiD:           {did_simple:.4f}")
print(f"Model 1 (Basic):      {model1.params['did']:.4f} (SE: {model1.bse['did']:.4f})")
print(f"Model 2 (Covariates): {model2.params['did']:.4f} (SE: {model2.bse['did']:.4f})")
print(f"Model 3 (Year FE):    {model3.params['did']:.4f} (SE: {model3.bse['did']:.4f})")

print("\n--- BY GENDER ---")
print(f"Males:   {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f}), n={len(df_male):,}")
print(f"Females: {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f}), n={len(df_female):,}")

print("\nFiles saved:")
print("  - summary_stats.csv")
print("  - regression_results.csv")
print("  - event_study_results.csv")
print("  - detailed_summary.csv")
print("  - model_summaries.txt")

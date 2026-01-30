"""
DACA Replication Analysis
==========================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of eligibility for DACA on probability of full-time employment?

Treatment Group: Ages 26-30 at time of policy (June 15, 2012)
Control Group: Ages 31-35 at time of policy (June 15, 2012)

Full-time employment: Defined as usually working 35+ hours per week (UHRSWORK >= 35)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output file for results
output_file = open('analysis_results.txt', 'w')

def log_print(msg):
    """Print to console and write to output file"""
    print(msg)
    output_file.write(str(msg) + '\n')

log_print("="*80)
log_print("DACA REPLICATION ANALYSIS")
log_print("="*80)

# Load the data
log_print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
log_print(f"Total observations loaded: {len(df):,}")
log_print(f"Years in data: {sorted(df['YEAR'].unique())}")
log_print(f"Columns: {list(df.columns)}")

# Step 2: Filter to Hispanic-Mexican population born in Mexico
log_print("\n2. FILTERING TO HISPANIC-MEXICAN POPULATION BORN IN MEXICO...")
log_print(f"\nHISPAN values distribution:")
log_print(df['HISPAN'].value_counts().sort_index())

# HISPAN = 1 is Mexican
# BPL = 200 is Mexico
log_print(f"\nBPL (birthplace) values for non-zero counts (first 20):")
bpl_counts = df['BPL'].value_counts().sort_index()
log_print(bpl_counts.head(20))

# Filter: Hispanic-Mexican (HISPAN == 1) AND Born in Mexico (BPL == 200)
df_mexican = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
log_print(f"\nAfter filtering to Hispanic-Mexican born in Mexico: {len(df_mexican):,}")

# Step 3: Filter to non-citizens (undocumented proxy)
log_print("\n3. FILTERING TO NON-CITIZENS (PROXY FOR UNDOCUMENTED)...")
log_print(f"\nCITIZEN values distribution in Mexican-born Hispanic sample:")
log_print(df_mexican['CITIZEN'].value_counts().sort_index())

# CITIZEN = 3 means "Not a citizen"
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes."
df_noncit = df_mexican[df_mexican['CITIZEN'] == 3].copy()
log_print(f"\nAfter filtering to non-citizens: {len(df_noncit):,}")

# Step 4: Filter by year of immigration (must have arrived before age 16 and lived in US since June 2007)
log_print("\n4. FILTERING BY DACA ELIGIBILITY CRITERIA...")
log_print(f"\nYRIMMIG distribution (first 30 values):")
yrimmig_counts = df_noncit['YRIMMIG'].value_counts().sort_index()
log_print(yrimmig_counts.head(30))

# DACA requirements:
# - Arrived unlawfully in the US before their 16th birthday
# - Lived continuously in the US since June 15, 2007
# - Were present in the US on June 15, 2012

# Age at arrival = YRIMMIG - BIRTHYR
# Must have arrived before 16th birthday: arrival_age < 16
df_noncit['arrival_age'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']

# Filter: arrived before age 16 and by 2007 at the latest
# YRIMMIG <= 2007 ensures continuous residence since June 2007
df_eligible = df_noncit[
    (df_noncit['arrival_age'] < 16) &
    (df_noncit['arrival_age'] >= 0) &  # Sanity check
    (df_noncit['YRIMMIG'] <= 2007) &    # Continuous residence since 2007
    (df_noncit['YRIMMIG'] > 0)          # Valid immigration year
].copy()

log_print(f"\nAfter applying DACA eligibility criteria (arrived before 16, in US since 2007): {len(df_eligible):,}")

# Step 5: Calculate age at time of policy (June 15, 2012)
log_print("\n5. CALCULATING AGE AT POLICY IMPLEMENTATION...")

# DACA was implemented June 15, 2012
# We need to calculate age as of June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Simple approach: use birth year to calculate age as of 2012
# Age in 2012 = 2012 - BIRTHYR
df_eligible['age_2012'] = 2012 - df_eligible['BIRTHYR']

# Account for birth quarter for more precision
# If born in Q3 or Q4 (Jul-Dec), subtract 1 since they hadn't had birthday by June 15
# Actually, for June 15: Q1 (Jan-Mar) had birthday, Q2 (Apr-Jun) approximately had birthday,
# Q3, Q4 haven't had birthday yet
# For simplicity and to be conservative, use the straightforward calculation
# This is a common approach in the literature

log_print(f"Age at 2012 distribution:")
log_print(df_eligible['age_2012'].describe())

# Step 6: Define treatment and control groups
log_print("\n6. DEFINING TREATMENT AND CONTROL GROUPS...")

# Treatment: ages 26-30 at June 15, 2012 (born 1982-1986, eligible for DACA)
# Control: ages 31-35 at June 15, 2012 (born 1977-1981, too old for DACA)
# Note: DACA required not yet having 31st birthday as of June 15, 2012

# Create treatment indicator
df_eligible['treated'] = ((df_eligible['age_2012'] >= 26) & (df_eligible['age_2012'] <= 30)).astype(int)
df_eligible['control'] = ((df_eligible['age_2012'] >= 31) & (df_eligible['age_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_eligible[(df_eligible['treated'] == 1) | (df_eligible['control'] == 1)].copy()

log_print(f"\nTreatment group (ages 26-30 in 2012): {df_analysis['treated'].sum():,}")
log_print(f"Control group (ages 31-35 in 2012): {df_analysis['control'].sum():,}")
log_print(f"Total for analysis: {len(df_analysis):,}")

# Step 7: Define pre/post periods
log_print("\n7. DEFINING PRE AND POST PERIODS...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA - examining effects per instructions)
# 2012 is excluded since policy was implemented mid-year

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['pre'] = (df_analysis['YEAR'] <= 2011).astype(int)

# Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()

log_print(f"\nPre-period (2006-2011) observations: {len(df_analysis[df_analysis['pre']==1]):,}")
log_print(f"Post-period (2013-2016) observations: {len(df_analysis[df_analysis['post']==1]):,}")

# Step 8: Create outcome variable - full-time employment
log_print("\n8. CREATING OUTCOME VARIABLE: FULL-TIME EMPLOYMENT...")

# Full-time employment: UHRSWORK >= 35 hours per week
# First check UHRSWORK distribution
log_print(f"\nUHRSWORK distribution:")
log_print(df_analysis['UHRSWORK'].describe())

# EMPSTAT: 1 = Employed, 2 = Unemployed, 3 = Not in labor force
log_print(f"\nEMPSTAT distribution:")
log_print(df_analysis['EMPSTAT'].value_counts().sort_index())

# Create full-time employment variable
# Full-time = 1 if UHRSWORK >= 35, else 0
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

log_print(f"\nFull-time employment rate: {df_analysis['fulltime'].mean():.4f}")

# Step 9: Create DiD interaction term
log_print("\n9. SETTING UP DIFFERENCE-IN-DIFFERENCES...")

df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Summary statistics by group and period
log_print("\nFull-time employment rates by group and period:")
summary = df_analysis.groupby(['treated', 'post'])['fulltime'].agg(['mean', 'count'])
log_print(summary)

# Calculate simple DiD
pre_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treated']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treated']==0) & (df_analysis['post']==1)]['fulltime'].mean()

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
log_print(f"\nSimple DiD calculation:")
log_print(f"Treatment group: Pre = {pre_treat:.4f}, Post = {post_treat:.4f}, Change = {post_treat - pre_treat:.4f}")
log_print(f"Control group: Pre = {pre_control:.4f}, Post = {post_control:.4f}, Change = {post_control - pre_control:.4f}")
log_print(f"DiD estimate: {simple_did:.4f}")

# Step 10: Run DiD Regression
log_print("\n10. RUNNING DIFFERENCE-IN-DIFFERENCES REGRESSION...")

# Basic DiD regression: fulltime ~ treated + post + treated*post
# Using person weights (PERWT)
log_print("\n--- Model 1: Basic DiD (no covariates) ---")
model1 = smf.wls('fulltime ~ treated + post + treated_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results1 = model1.fit(cov_type='HC1')  # Robust standard errors
log_print(results1.summary())

# Model 2: Add demographic covariates
log_print("\n--- Model 2: DiD with demographic covariates ---")
# Create covariate dummies
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_somecoll'] = ((df_analysis['EDUC'] > 6) & (df_analysis['EDUC'] < 10)).astype(int)
df_analysis['educ_coll'] = (df_analysis['EDUC'] >= 10).astype(int)

model2 = smf.wls('fulltime ~ treated + post + treated_post + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results2 = model2.fit(cov_type='HC1')
log_print(results2.summary())

# Model 3: Add year fixed effects
log_print("\n--- Model 3: DiD with year fixed effects ---")
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model3 = smf.wls('fulltime ~ treated + C(YEAR) + treated_post + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results3 = model3.fit(cov_type='HC1')
log_print(results3.summary())

# Model 4: Add state fixed effects
log_print("\n--- Model 4: DiD with state and year fixed effects ---")
model4 = smf.wls('fulltime ~ treated + C(YEAR) + C(STATEFIP) + treated_post + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df_analysis,
                  weights=df_analysis['PERWT'])
results4 = model4.fit(cov_type='HC1')
# Print only the DiD coefficient
log_print(f"\nDiD coefficient (treated_post): {results4.params['treated_post']:.4f}")
log_print(f"Standard error: {results4.bse['treated_post']:.4f}")
log_print(f"t-statistic: {results4.tvalues['treated_post']:.4f}")
log_print(f"p-value: {results4.pvalues['treated_post']:.4f}")
log_print(f"95% CI: [{results4.conf_int().loc['treated_post', 0]:.4f}, {results4.conf_int().loc['treated_post', 1]:.4f}]")

# Step 11: Additional analyses
log_print("\n11. ADDITIONAL ANALYSES...")

# By gender
log_print("\n--- Heterogeneity by Gender ---")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_gender = df_analysis[df_analysis['SEX'] == gender]
    model_g = smf.wls('fulltime ~ treated + post + treated_post',
                      data=df_gender,
                      weights=df_gender['PERWT'])
    results_g = model_g.fit(cov_type='HC1')
    log_print(f"{label}: DiD = {results_g.params['treated_post']:.4f} (SE = {results_g.bse['treated_post']:.4f}), p = {results_g.pvalues['treated_post']:.4f}")

# Pre-trends analysis
log_print("\n--- Pre-trends Analysis ---")
df_pre = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_pre['year_trend'] = df_pre['YEAR'] - 2006
df_pre['treated_trend'] = df_pre['treated'] * df_pre['year_trend']

model_trend = smf.wls('fulltime ~ treated + year_trend + treated_trend',
                       data=df_pre,
                       weights=df_pre['PERWT'])
results_trend = model_trend.fit(cov_type='HC1')
log_print(f"Differential pre-trend (treated_trend): {results_trend.params['treated_trend']:.4f}")
log_print(f"Standard error: {results_trend.bse['treated_trend']:.4f}")
log_print(f"p-value: {results_trend.pvalues['treated_trend']:.4f}")

# Event study
log_print("\n--- Event Study Coefficients ---")
df_analysis['event_time'] = df_analysis['YEAR'] - 2012
# Reference year: 2011 (last pre-treatment year)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
    df_analysis[f'treated_year_{year}'] = df_analysis['treated'] * df_analysis[f'year_{year}']

# Create formula for event study
event_terms = ' + '.join([f'treated_year_{y}' for y in event_years])
formula_event = f'fulltime ~ treated + C(YEAR) + {event_terms}'
model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT'])
results_event = model_event.fit(cov_type='HC1')

log_print("\nEvent study coefficients (relative to 2011):")
for year in event_years:
    coef = results_event.params[f'treated_year_{year}']
    se = results_event.bse[f'treated_year_{year}']
    log_print(f"Year {year}: {coef:.4f} (SE = {se:.4f})")

# Step 12: Summary statistics for report
log_print("\n12. SUMMARY STATISTICS FOR REPORT...")

log_print("\n--- Sample Characteristics ---")
log_print(f"Total sample size: {len(df_analysis):,}")
log_print(f"Treatment group (ages 26-30): {(df_analysis['treated']==1).sum():,}")
log_print(f"Control group (ages 31-35): {(df_analysis['control']==1).sum():,}")

log_print("\n--- Demographics by Treatment Status ---")
for var, label in [('female', 'Female'), ('married', 'Married'), ('educ_lesshs', 'Less than HS'),
                   ('educ_hs', 'High School'), ('educ_somecoll', 'Some College'), ('educ_coll', 'College+')]:
    treat_mean = df_analysis[df_analysis['treated']==1][var].mean()
    control_mean = df_analysis[df_analysis['treated']==0][var].mean()
    log_print(f"{label}: Treatment = {treat_mean:.3f}, Control = {control_mean:.3f}")

log_print("\n--- Full-time Employment by Year ---")
by_year = df_analysis.groupby(['YEAR', 'treated'])['fulltime'].agg(['mean', 'count']).unstack()
log_print(by_year)

# Save key results for the report
log_print("\n" + "="*80)
log_print("PREFERRED ESTIMATE (Model 4 - Full specification with state and year FE)")
log_print("="*80)
log_print(f"DiD Coefficient: {results4.params['treated_post']:.4f}")
log_print(f"Standard Error: {results4.bse['treated_post']:.4f}")
log_print(f"95% Confidence Interval: [{results4.conf_int().loc['treated_post', 0]:.4f}, {results4.conf_int().loc['treated_post', 1]:.4f}]")
log_print(f"Sample Size: {len(df_analysis):,}")
log_print(f"p-value: {results4.pvalues['treated_post']:.4f}")

# Export results for LaTeX tables
results_dict = {
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + Year FE', '(4) + State FE'],
    'Coefficient': [results1.params['treated_post'], results2.params['treated_post'],
                    results3.params['treated_post'], results4.params['treated_post']],
    'Std_Error': [results1.bse['treated_post'], results2.bse['treated_post'],
                  results3.bse['treated_post'], results4.bse['treated_post']],
    'p_value': [results1.pvalues['treated_post'], results2.pvalues['treated_post'],
                results3.pvalues['treated_post'], results4.pvalues['treated_post']],
    'CI_lower': [results1.conf_int().loc['treated_post', 0], results2.conf_int().loc['treated_post', 0],
                 results3.conf_int().loc['treated_post', 0], results4.conf_int().loc['treated_post', 0]],
    'CI_upper': [results1.conf_int().loc['treated_post', 1], results2.conf_int().loc['treated_post', 1],
                 results3.conf_int().loc['treated_post', 1], results4.conf_int().loc['treated_post', 1]],
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)

# Export event study results
event_results = []
for year in event_years:
    event_results.append({
        'Year': year,
        'Coefficient': results_event.params[f'treated_year_{year}'],
        'Std_Error': results_event.bse[f'treated_year_{year}'],
        'CI_lower': results_event.conf_int().loc[f'treated_year_{year}', 0],
        'CI_upper': results_event.conf_int().loc[f'treated_year_{year}', 1]
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Export summary statistics
summary_stats = df_analysis.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'female': 'mean',
    'married': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')

output_file.close()
log_print("\nAnalysis complete. Results saved to CSV files.")

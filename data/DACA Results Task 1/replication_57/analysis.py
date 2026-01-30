"""
DACA Replication Study: Effect on Full-Time Employment
Analysis Script
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
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n1. LOADING DATA...")
print("-" * 40)

# Load the main data file
df = pd.read_csv('data/data.csv')

print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {list(df.columns)}")

# =============================================================================
# 2. INITIAL DATA EXPLORATION
# =============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Check year distribution
print("\nObservations by year:")
print(df['YEAR'].value_counts().sort_index())

# Check Hispanic-Mexican from Mexico
print("\nHISPAN values (1=Mexican):")
print(df['HISPAN'].value_counts().head(10))

print("\nBPL values (200=Mexico):")
print(df['BPL'].value_counts().head(10))

print("\nCITIZEN values (3=Not a citizen):")
print(df['CITIZEN'].value_counts())

# =============================================================================
# 3. SAMPLE RESTRICTIONS
# =============================================================================
print("\n3. APPLYING SAMPLE RESTRICTIONS")
print("-" * 40)

# Start with full sample
print(f"Starting sample: {len(df):,}")

# Restrict to Hispanic-Mexican ethnicity (HISPAN == 1)
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican restriction: {len(df_sample):,}")

# Restrict to born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexico birthplace restriction: {len(df_sample):,}")

# Restrict to non-citizens (CITIZEN == 3)
# This is our proxy for undocumented status
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen restriction: {len(df_sample):,}")

# Exclude 2012 (DACA implementation year - can't distinguish pre/post)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# =============================================================================
# 4. CONSTRUCT DACA ELIGIBILITY VARIABLES
# =============================================================================
print("\n4. CONSTRUCTING DACA ELIGIBILITY VARIABLES")
print("-" * 40)

# DACA Eligibility Requirements:
# 1. Arrived before 16th birthday
# 2. Born after June 15, 1981 (under 31 on June 15, 2012)
# 3. In US continuously since June 15, 2007 (arrived by 2007)
# 4. Present in US on June 15, 2012 without lawful status

# Calculate age at arrival
# YRIMMIG is year of immigration
# BIRTHYR is birth year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Check for valid immigration year (0 means N/A)
print(f"\nObservations with valid YRIMMIG (>0): {(df_sample['YRIMMIG'] > 0).sum():,}")
print(f"Observations with missing YRIMMIG (0): {(df_sample['YRIMMIG'] == 0).sum():,}")

# Restrict to those with valid immigration year
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"After requiring valid YRIMMIG: {len(df_sample):,}")

# Criteria 1: Arrived before 16th birthday
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16).astype(int)
print(f"\nArrived before age 16: {df_sample['arrived_before_16'].sum():,}")

# Criteria 2: Born after June 15, 1981 (under 31 on June 15, 2012)
# Use BIRTHYR and BIRTHQTR for more precision
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means: BIRTHYR > 1981 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
# Actually, June 15 falls in Q2 (Apr-Jun), so BIRTHQTR > 2 means after June
# To be safe, we'll use BIRTHYR >= 1982 OR (BIRTHYR == 1981 AND BIRTHQTR >= 3)
df_sample['under_31_june2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
).astype(int)
print(f"Under 31 on June 15, 2012: {df_sample['under_31_june2012'].sum():,}")

# Criteria 3: In US since June 15, 2007 (arrived by 2007)
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)
print(f"In US since 2007: {df_sample['in_us_since_2007'].sum():,}")

# DACA eligible = all criteria met
df_sample['daca_eligible'] = (
    (df_sample['arrived_before_16'] == 1) &
    (df_sample['under_31_june2012'] == 1) &
    (df_sample['in_us_since_2007'] == 1)
).astype(int)
print(f"\nDACA eligible: {df_sample['daca_eligible'].sum():,}")
print(f"Not DACA eligible: {(df_sample['daca_eligible'] == 0).sum():,}")

# =============================================================================
# 5. DEFINE OUTCOME VARIABLE
# =============================================================================
print("\n5. DEFINING OUTCOME VARIABLE")
print("-" * 40)

# Full-time employment: Usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A or not working)
# EMPSTAT: 1 = Employed

# First, define employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)
print(f"Employed: {df_sample['employed'].sum():,}")

# Full-time = employed AND works 35+ hours
df_sample['fulltime'] = (
    (df_sample['EMPSTAT'] == 1) &
    (df_sample['UHRSWORK'] >= 35)
).astype(int)
print(f"Full-time employed (35+ hrs): {df_sample['fulltime'].sum():,}")
print(f"Full-time rate: {df_sample['fulltime'].mean():.3f}")

# =============================================================================
# 6. DEFINE TREATMENT PERIOD
# =============================================================================
print("\n6. DEFINING TREATMENT PERIOD")
print("-" * 40)

# Post-DACA: 2013-2016
# Pre-DACA: 2006-2011
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
print(f"Pre-DACA observations (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"Post-DACA observations (2013-2016): {(df_sample['post'] == 1).sum():,}")

# =============================================================================
# 7. RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n7. RESTRICTING TO WORKING-AGE POPULATION")
print("-" * 40)

# Restrict to ages 18-55 for meaningful employment analysis
# This also helps focus on the relevant population
df_sample = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 55)].copy()
print(f"After age 18-55 restriction: {len(df_sample):,}")

# =============================================================================
# 8. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n8. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Summary by DACA eligibility and period
print("\n--- Sample Sizes ---")
crosstab = pd.crosstab(df_sample['daca_eligible'], df_sample['post'], margins=True)
crosstab.index = ['Not Eligible', 'Eligible', 'Total']
crosstab.columns = ['Pre-DACA', 'Post-DACA', 'Total']
print(crosstab)

# Mean full-time employment rates
print("\n--- Full-Time Employment Rates ---")
ft_rates = df_sample.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
ft_rates.index = ['Not Eligible', 'Eligible']
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates.round(4))

# Calculate simple DiD estimate
pre_diff = ft_rates.loc['Eligible', 'Pre-DACA'] - ft_rates.loc['Not Eligible', 'Pre-DACA']
post_diff = ft_rates.loc['Eligible', 'Post-DACA'] - ft_rates.loc['Not Eligible', 'Post-DACA']
did_simple = post_diff - pre_diff
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# =============================================================================
# 9. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n9. DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("-" * 40)

# Create interaction term
df_sample['did'] = df_sample['daca_eligible'] * df_sample['post']

# Basic DiD regression (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('fulltime ~ daca_eligible + post + did', data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient: {model1.params['did']:.4f}")
print(f"Std Error (robust): {model1.bse['did']:.4f}")
print(f"t-statistic: {model1.tvalues['did']:.4f}")
print(f"p-value: {model1.pvalues['did']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['did', 0]:.4f}, {model1.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model1.nobs):,}")

# Model with controls
print("\n--- Model 2: DiD with Controls ---")
# Add control variables
df_sample['age_sq'] = df_sample['AGE'] ** 2
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] == 1).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married + educ_hs',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient: {model2.params['did']:.4f}")
print(f"Std Error (robust): {model2.bse['did']:.4f}")
print(f"t-statistic: {model2.tvalues['did']:.4f}")
print(f"p-value: {model2.pvalues['did']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['did', 0]:.4f}, {model2.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model2.nobs):,}")

# Model with year and state fixed effects
print("\n--- Model 3: DiD with Controls + Year FE ---")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model3 = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + C(year_factor)',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient: {model3.params['did']:.4f}")
print(f"Std Error (robust): {model3.bse['did']:.4f}")
print(f"t-statistic: {model3.tvalues['did']:.4f}")
print(f"p-value: {model3.pvalues['did']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['did', 0]:.4f}, {model3.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model3.nobs):,}")

# Model with state fixed effects
print("\n--- Model 4: DiD with Controls + Year FE + State FE ---")
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)
model4 = smf.ols('fulltime ~ daca_eligible + did + AGE + age_sq + female + married + educ_hs + C(year_factor) + C(state_factor)',
                 data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient: {model4.params['did']:.4f}")
print(f"Std Error (robust): {model4.bse['did']:.4f}")
print(f"t-statistic: {model4.tvalues['did']:.4f}")
print(f"p-value: {model4.pvalues['did']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['did', 0]:.4f}, {model4.conf_int().loc['did', 1]:.4f}]")
print(f"N: {int(model4.nobs):,}")

# =============================================================================
# 10. WEIGHTED ANALYSIS
# =============================================================================
print("\n10. WEIGHTED ANALYSIS (Using PERWT)")
print("-" * 40)

from statsmodels.regression.linear_model import WLS

# Prepare data for weighted regression
X_weighted = df_sample[['daca_eligible', 'post', 'did', 'AGE', 'age_sq', 'female', 'married', 'educ_hs']].copy()
X_weighted = sm.add_constant(X_weighted)
y_weighted = df_sample['fulltime']
weights = df_sample['PERWT']

model_weighted = WLS(y_weighted, X_weighted, weights=weights).fit(cov_type='HC1')
print(f"DiD Coefficient (weighted): {model_weighted.params['did']:.4f}")
print(f"Std Error (robust): {model_weighted.bse['did']:.4f}")
print(f"p-value: {model_weighted.pvalues['did']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['did', 0]:.4f}, {model_weighted.conf_int().loc['did', 1]:.4f}]")

# =============================================================================
# 11. ROBUSTNESS CHECKS
# =============================================================================
print("\n11. ROBUSTNESS CHECKS")
print("-" * 40)

# 11.1 Alternative age restrictions
print("\n--- 11.1 Alternative Age Restrictions (20-45) ---")
df_alt_age = df_sample[(df_sample['AGE'] >= 20) & (df_sample['AGE'] <= 45)].copy()
model_alt_age = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + female + married + educ_hs',
                        data=df_alt_age).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_alt_age.params['did']:.4f}")
print(f"Std Error: {model_alt_age.bse['did']:.4f}")
print(f"N: {int(model_alt_age.nobs):,}")

# 11.2 Employment (any employment) as outcome
print("\n--- 11.2 Alternative Outcome: Any Employment ---")
model_emp = smf.ols('employed ~ daca_eligible + post + did + AGE + age_sq + female + married + educ_hs',
                    data=df_sample).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_emp.params['did']:.4f}")
print(f"Std Error: {model_emp.bse['did']:.4f}")
print(f"N: {int(model_emp.nobs):,}")

# 11.3 By gender
print("\n--- 11.3 By Gender ---")
df_male = df_sample[df_sample['female'] == 0]
df_female = df_sample[df_sample['female'] == 1]

model_male = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + married + educ_hs',
                     data=df_male).fit(cov_type='HC1')
model_female = smf.ols('fulltime ~ daca_eligible + post + did + AGE + age_sq + married + educ_hs',
                       data=df_female).fit(cov_type='HC1')

print(f"Males - DiD: {model_male.params['did']:.4f} (SE: {model_male.bse['did']:.4f}), N={int(model_male.nobs):,}")
print(f"Females - DiD: {model_female.params['did']:.4f} (SE: {model_female.bse['did']:.4f}), N={int(model_female.nobs):,}")

# =============================================================================
# 12. EVENT STUDY / DYNAMIC DiD
# =============================================================================
print("\n12. EVENT STUDY ANALYSIS")
print("-" * 40)

# Create year dummies interacted with treatment
years = sorted(df_sample['YEAR'].unique())
print(f"Years: {years}")

# Reference year: 2011 (last pre-treatment year)
for yr in years:
    if yr != 2011:  # Omit 2011 as reference
        df_sample[f'year_{yr}'] = (df_sample['YEAR'] == yr).astype(int)
        df_sample[f'did_{yr}'] = df_sample[f'year_{yr}'] * df_sample['daca_eligible']

# Event study regression
year_dummies = ' + '.join([f'year_{yr}' for yr in years if yr != 2011])
did_terms = ' + '.join([f'did_{yr}' for yr in years if yr != 2011])

event_formula = f'fulltime ~ daca_eligible + {year_dummies} + {did_terms} + AGE + age_sq + female + married + educ_hs'
model_event = smf.ols(event_formula, data=df_sample).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year):")
print("(Reference: 2011)")
for yr in sorted(years):
    if yr != 2011:
        coef = model_event.params[f'did_{yr}']
        se = model_event.bse[f'did_{yr}']
        pval = model_event.pvalues[f'did_{yr}']
        print(f"  {yr}: {coef:7.4f} (SE: {se:.4f}, p={pval:.3f})")

# =============================================================================
# 13. PLACEBO TEST
# =============================================================================
print("\n13. PLACEBO TEST")
print("-" * 40)

# Use only pre-DACA data and pretend 2009 was the treatment year
df_placebo = df_sample[df_sample['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['did_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']

model_placebo = smf.ols('fulltime ~ daca_eligible + post_placebo + did_placebo + AGE + age_sq + female + married + educ_hs',
                        data=df_placebo).fit(cov_type='HC1')
print(f"Placebo DiD (2009 fake treatment): {model_placebo.params['did_placebo']:.4f}")
print(f"Std Error: {model_placebo.bse['did_placebo']:.4f}")
print(f"p-value: {model_placebo.pvalues['did_placebo']:.4f}")

# =============================================================================
# 14. SUMMARY STATISTICS FOR REPORT
# =============================================================================
print("\n14. SUMMARY STATISTICS FOR REPORT")
print("-" * 40)

# Detailed summary by group
print("\n--- Demographics by Group ---")
for name, group in df_sample.groupby('daca_eligible'):
    label = "DACA Eligible" if name == 1 else "Not Eligible"
    print(f"\n{label}:")
    print(f"  N: {len(group):,}")
    print(f"  Mean Age: {group['AGE'].mean():.1f}")
    print(f"  Female %: {group['female'].mean()*100:.1f}%")
    print(f"  Married %: {group['married'].mean()*100:.1f}%")
    print(f"  HS+ Education %: {group['educ_hs'].mean()*100:.1f}%")
    print(f"  Full-time Employment %: {group['fulltime'].mean()*100:.1f}%")

# Year-by-year rates
print("\n--- Full-Time Employment by Year and Group ---")
year_rates = df_sample.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
year_rates.columns = ['Not Eligible', 'Eligible']
print(year_rates.round(4))

# =============================================================================
# 15. SAVE KEY RESULTS
# =============================================================================
print("\n15. SAVING RESULTS")
print("-" * 40)

# Preferred estimate (Model 4 with all fixed effects)
preferred_coef = model4.params['did']
preferred_se = model4.bse['did']
preferred_ci = model4.conf_int().loc['did']
preferred_n = int(model4.nobs)
preferred_pval = model4.pvalues['did']

print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 4: Controls + Year FE + State FE)")
print("=" * 80)
print(f"Effect Size: {preferred_coef:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"p-value: {preferred_pval:.4f}")
print(f"Sample Size: {preferred_n:,}")

# Save full model summary
print("\n--- Full Model 4 Summary ---")
print(model4.summary())

# Store results for LaTeX
results_dict = {
    'model1': {'coef': model1.params['did'], 'se': model1.bse['did'], 'n': int(model1.nobs), 'pval': model1.pvalues['did']},
    'model2': {'coef': model2.params['did'], 'se': model2.bse['did'], 'n': int(model2.nobs), 'pval': model2.pvalues['did']},
    'model3': {'coef': model3.params['did'], 'se': model3.bse['did'], 'n': int(model3.nobs), 'pval': model3.pvalues['did']},
    'model4': {'coef': model4.params['did'], 'se': model4.bse['did'], 'n': int(model4.nobs), 'pval': model4.pvalues['did']},
    'weighted': {'coef': model_weighted.params['did'], 'se': model_weighted.bse['did'], 'pval': model_weighted.pvalues['did']},
    'male': {'coef': model_male.params['did'], 'se': model_male.bse['did'], 'n': int(model_male.nobs)},
    'female': {'coef': model_female.params['did'], 'se': model_female.bse['did'], 'n': int(model_female.nobs)},
    'placebo': {'coef': model_placebo.params['did_placebo'], 'se': model_placebo.bse['did_placebo'], 'pval': model_placebo.pvalues['did_placebo']},
    'employed': {'coef': model_emp.params['did'], 'se': model_emp.bse['did'], 'n': int(model_emp.nobs)},
}

# Save event study results
event_results = {}
for yr in sorted(years):
    if yr != 2011:
        event_results[yr] = {
            'coef': model_event.params[f'did_{yr}'],
            'se': model_event.bse[f'did_{yr}'],
            'pval': model_event.pvalues[f'did_{yr}']
        }

# Export for use in report
import json
with open('analysis_results.json', 'w') as f:
    json.dump({
        'main_results': results_dict,
        'event_study': {str(k): v for k, v in event_results.items()},
        'sample_sizes': crosstab.to_dict(),
        'ft_rates': ft_rates.to_dict(),
        'year_rates': year_rates.to_dict(),
        'preferred': {
            'coef': float(preferred_coef),
            'se': float(preferred_se),
            'ci_lower': float(preferred_ci[0]),
            'ci_upper': float(preferred_ci[1]),
            'pval': float(preferred_pval),
            'n': preferred_n
        }
    }, f, indent=2)

print("\nResults saved to analysis_results.json")

# Summary stats to export
summary_stats = df_sample.groupby('daca_eligible').agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'fulltime': 'mean',
    'employed': 'mean'
}).round(4)
summary_stats.index = ['Not Eligible', 'Eligible']
summary_stats.to_csv('summary_stats.csv')
print("Summary statistics saved to summary_stats.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

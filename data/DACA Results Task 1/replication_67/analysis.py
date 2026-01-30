"""
DACA Replication Study: Effect of DACA Eligibility on Full-Time Employment
Among Hispanic-Mexican, Mexican-Born Individuals in the United States

This script conducts a difference-in-differences analysis examining the causal
impact of DACA eligibility on full-time employment (working 35+ hours/week).
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
print("DACA REPLICATION STUDY")
print("Effect of DACA Eligibility on Full-Time Employment")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Data")
print("="*80)

# Read the data
df = pd.read_csv('data/data.csv')

print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: SAMPLE SELECTION - Hispanic-Mexican, Mexican-Born
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Sample Selection")
print("="*80)

# According to data dictionary:
# HISPAN = 1 is Mexican
# BPL = 200 is Mexico
# CITIZEN = 3 is "Not a citizen" (we assume undocumented for DACA purposes)

# Filter to Hispanic-Mexican ethnicity (HISPAN = 1)
print(f"\nFiltering to Hispanic-Mexican (HISPAN=1)...")
df_hisp_mex = df[df['HISPAN'] == 1].copy()
print(f"Observations with Hispanic-Mexican ethnicity: {len(df_hisp_mex):,}")

# Filter to Mexican-born (BPL = 200)
print(f"\nFiltering to Mexican-born (BPL=200)...")
df_mex_born = df_hisp_mex[df_hisp_mex['BPL'] == 200].copy()
print(f"Observations born in Mexico: {len(df_mex_born):,}")

# Filter to non-citizens (CITIZEN = 3)
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
print(f"\nFiltering to non-citizens (CITIZEN=3)...")
df_noncit = df_mex_born[df_mex_born['CITIZEN'] == 3].copy()
print(f"Observations who are non-citizens: {len(df_noncit):,}")

# ============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Defining DACA Eligibility")
print("="*80)

"""
DACA Eligibility Criteria (from instructions):
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012

For identification purposes, we use:
- Age on June 15, 2012 must be < 31 (born after June 15, 1981)
- Arrived before 16th birthday
- In US since at least June 15, 2007 (at least 5 years by June 2012)
"""

# Calculate age as of June 15, 2012
# Using BIRTHYR and BIRTHQTR
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec

# Age criterion: Must be under 31 on June 15, 2012
# This means born AFTER June 15, 1981
# Born 1982 or later definitely qualifies
# Born 1981: qualifies only if born after June 15 (BIRTHQTR 3 or 4)

df_noncit['age_jun2012'] = 2012 - df_noncit['BIRTHYR']
# Adjust for birth quarter - if born July-Dec, they're actually younger in June
df_noncit.loc[df_noncit['BIRTHQTR'].isin([3, 4]), 'age_jun2012'] -= 1

# Criterion 1: Under 31 on June 15, 2012
df_noncit['under_31_jun2012'] = df_noncit['age_jun2012'] < 31

# Criterion 2: Arrived before 16th birthday
# Age at arrival = YRIMMIG - BIRTHYR
# Must have arrived when age < 16
df_noncit['age_at_arrival'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
df_noncit['arrived_before_16'] = (df_noncit['age_at_arrival'] < 16) & (df_noncit['YRIMMIG'] > 0)

# Criterion 3: In US since June 15, 2007 (5+ years by June 2012)
# YRIMMIG <= 2007
df_noncit['in_us_since_2007'] = (df_noncit['YRIMMIG'] <= 2007) & (df_noncit['YRIMMIG'] > 0)

# Criterion 4: Present in US on June 15, 2012 - assumed for all in ACS sample

# Define DACA-eligible
df_noncit['daca_eligible'] = (
    df_noncit['under_31_jun2012'] &
    df_noncit['arrived_before_16'] &
    df_noncit['in_us_since_2007']
)

print("\nDACA Eligibility Criteria Summary:")
print(f"  Under 31 on June 15, 2012: {df_noncit['under_31_jun2012'].sum():,}")
print(f"  Arrived before age 16: {df_noncit['arrived_before_16'].sum():,}")
print(f"  In US since 2007: {df_noncit['in_us_since_2007'].sum():,}")
print(f"  DACA-eligible (all criteria): {df_noncit['daca_eligible'].sum():,}")

# ============================================================================
# STEP 4: DEFINE TREATMENT AND POST PERIODS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Defining Treatment and Post Periods")
print("="*80)

# Treatment: DACA-eligible individuals
# Post period: 2013-2016 (DACA implemented June 2012, but 2012 is ambiguous)
# Pre period: 2006-2011

# Exclude 2012 since it straddles implementation
df_analysis = df_noncit[df_noncit['YEAR'] != 2012].copy()

# Define post period
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"\nExcluding 2012 (implementation year with ambiguous timing)")
print(f"Analysis sample size: {len(df_analysis):,}")
print(f"\nObservations by period:")
print(f"  Pre-DACA (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"  Post-DACA (2013-2016): {(df_analysis['post']==1).sum():,}")

# ============================================================================
# STEP 5: DEFINE OUTCOME - FULL-TIME EMPLOYMENT
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Defining Outcome Variable")
print("="*80)

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: usual hours worked per week
# Must be employed (EMPSTAT = 1)

df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
df_analysis['fulltime'] = ((df_analysis['UHRSWORK'] >= 35) & (df_analysis['employed'] == 1)).astype(int)

print(f"\nEmployment Status:")
print(f"  Employed: {df_analysis['employed'].sum():,} ({100*df_analysis['employed'].mean():.1f}%)")
print(f"  Full-time (35+ hrs): {df_analysis['fulltime'].sum():,} ({100*df_analysis['fulltime'].mean():.1f}%)")

# ============================================================================
# STEP 6: RESTRICT SAMPLE TO WORKING-AGE POPULATION
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Sample Restrictions")
print("="*80)

# Restrict to working-age population (18-64)
# This is standard in labor economics
df_analysis = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 64)].copy()

print(f"Restricted to ages 18-64")
print(f"Final analysis sample: {len(df_analysis):,}")

print(f"\nBy DACA eligibility:")
print(f"  Eligible: {df_analysis['daca_eligible'].sum():,}")
print(f"  Not eligible: {(~df_analysis['daca_eligible']).sum():,}")

# ============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Descriptive Statistics")
print("="*80)

# Summary by treatment group and period
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'SEX': lambda x: (x==1).mean(),  # Proportion male
    'EDUC': 'mean'
}).round(4)

print("\nFull-time Employment Rate by Group and Period:")
print(summary_stats)

# Calculate simple DiD
ft_elig_pre = df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post']==0)]['fulltime'].mean()
ft_elig_post = df_analysis[(df_analysis['daca_eligible']) & (df_analysis['post']==1)]['fulltime'].mean()
ft_inelig_pre = df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post']==0)]['fulltime'].mean()
ft_inelig_post = df_analysis[(~df_analysis['daca_eligible']) & (df_analysis['post']==1)]['fulltime'].mean()

simple_did = (ft_elig_post - ft_elig_pre) - (ft_inelig_post - ft_inelig_pre)

print("\n" + "-"*60)
print("Simple Difference-in-Differences Calculation:")
print("-"*60)
print(f"Eligible, Pre-DACA:    {ft_elig_pre:.4f}")
print(f"Eligible, Post-DACA:   {ft_elig_post:.4f}")
print(f"Ineligible, Pre-DACA:  {ft_inelig_pre:.4f}")
print(f"Ineligible, Post-DACA: {ft_inelig_post:.4f}")
print(f"\nChange for Eligible:   {ft_elig_post - ft_elig_pre:.4f}")
print(f"Change for Ineligible: {ft_inelig_post - ft_inelig_pre:.4f}")
print(f"\nSimple DiD Estimate:   {simple_did:.4f}")

# ============================================================================
# STEP 8: DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Difference-in-Differences Regression")
print("="*80)

# Create interaction term
df_analysis['daca_eligible_int'] = df_analysis['daca_eligible'].astype(int)
df_analysis['did'] = df_analysis['daca_eligible_int'] * df_analysis['post']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (no controls) ---")
model1 = smf.ols('fulltime ~ daca_eligible_int + post + did', data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']}
)
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
df_analysis['hs_or_more'] = (df_analysis['EDUC'] >= 6).astype(int)
df_analysis['college'] = (df_analysis['EDUC'] >= 10).astype(int)

model2 = smf.ols('fulltime ~ daca_eligible_int + post + did + AGE + I(AGE**2) + male + married + hs_or_more',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(model2.summary())

# Model 3: DiD with year and state fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + male + married + hs_or_more',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(model3.summary())

# Model 4: Full model with state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
model4 = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + C(STATEFIP) + AGE + I(AGE**2) + male + married + hs_or_more',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

# Print key coefficients only for Model 4
print("\nModel 4 Key Results (Year and State FE):")
print(f"DiD Coefficient (Treatment Effect): {model4.params['did']:.6f}")
print(f"Standard Error: {model4.bse['did']:.6f}")
print(f"t-statistic: {model4.tvalues['did']:.4f}")
print(f"p-value: {model4.pvalues['did']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['did', 0]:.6f}, {model4.conf_int().loc['did', 1]:.6f}]")
print(f"N: {int(model4.nobs):,}")
print(f"R-squared: {model4.rsquared:.4f}")

# ============================================================================
# STEP 9: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Robustness Checks")
print("="*80)

# Robustness 1: Using person weights
print("\n--- Robustness Check 1: Weighted Regression ---")
model_weighted = smf.wls('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + male + married + hs_or_more',
                         data=df_analysis, weights=df_analysis['PERWT']).fit(
                         cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"Weighted DiD Coefficient: {model_weighted.params['did']:.6f}")
print(f"Standard Error: {model_weighted.bse['did']:.6f}")
print(f"95% CI: [{model_weighted.conf_int().loc['did', 0]:.6f}, {model_weighted.conf_int().loc['did', 1]:.6f}]")

# Robustness 2: Restrict to prime working age (25-54)
print("\n--- Robustness Check 2: Prime Working Age (25-54) ---")
df_prime = df_analysis[(df_analysis['AGE'] >= 25) & (df_analysis['AGE'] <= 54)].copy()
model_prime = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + male + married + hs_or_more',
                      data=df_prime).fit(cov_type='cluster', cov_kwds={'groups': df_prime['STATEFIP']})
print(f"Prime Age DiD Coefficient: {model_prime.params['did']:.6f}")
print(f"Standard Error: {model_prime.bse['did']:.6f}")
print(f"N: {int(model_prime.nobs):,}")

# Robustness 3: By gender
print("\n--- Robustness Check 3: By Gender ---")
df_male = df_analysis[df_analysis['male'] == 1].copy()
df_female = df_analysis[df_analysis['male'] == 0].copy()

model_male = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + married + hs_or_more',
                     data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
model_female = smf.ols('fulltime ~ daca_eligible_int + did + C(YEAR) + AGE + I(AGE**2) + married + hs_or_more',
                       data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})

print(f"Male DiD Coefficient: {model_male.params['did']:.6f} (SE: {model_male.bse['did']:.6f})")
print(f"Female DiD Coefficient: {model_female.params['did']:.6f} (SE: {model_female.bse['did']:.6f})")

# ============================================================================
# STEP 10: EVENT STUDY / PRE-TRENDS
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Event Study Analysis (Pre-Trends)")
print("="*80)

# Create year dummies interacted with treatment
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
df_analysis['year_2011'] = (df_analysis['YEAR'] == 2011).astype(int)
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Interactions (omit 2011 as reference year)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'elig_x_{year}'] = df_analysis['daca_eligible_int'] * df_analysis[f'year_{year}']

event_study_formula = """fulltime ~ daca_eligible_int +
    elig_x_2006 + elig_x_2007 + elig_x_2008 + elig_x_2009 + elig_x_2010 +
    elig_x_2013 + elig_x_2014 + elig_x_2015 + elig_x_2016 +
    C(YEAR) + AGE + I(AGE**2) + male + married + hs_or_more"""

model_event = smf.ols(event_study_formula, data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*60)
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    coef = model_event.params[f'elig_x_{year}']
    se = model_event.bse[f'elig_x_{year}']
    pval = model_event.pvalues[f'elig_x_{year}']
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {year}: {coef:8.4f} ({se:.4f}) {stars}")

# ============================================================================
# STEP 11: SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("STEP 11: Saving Results")
print("="*80)

# Create summary DataFrame
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'Year + State FE', 'Weighted', 'Prime Age'],
    'DiD_Coefficient': [
        model1.params['did'], model2.params['did'], model3.params['did'],
        model4.params['did'], model_weighted.params['did'], model_prime.params['did']
    ],
    'Std_Error': [
        model1.bse['did'], model2.bse['did'], model3.bse['did'],
        model4.bse['did'], model_weighted.bse['did'], model_prime.bse['did']
    ],
    'p_value': [
        model1.pvalues['did'], model2.pvalues['did'], model3.pvalues['did'],
        model4.pvalues['did'], model_weighted.pvalues['did'], model_prime.pvalues['did']
    ],
    'N': [
        int(model1.nobs), int(model2.nobs), int(model3.nobs),
        int(model4.nobs), int(model_weighted.nobs), int(model_prime.nobs)
    ]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to regression_results.csv")

# Save event study results
event_study_results = {
    'Year': event_years,
    'Coefficient': [model_event.params[f'elig_x_{y}'] for y in event_years],
    'Std_Error': [model_event.bse[f'elig_x_{y}'] for y in event_years],
    'CI_Lower': [model_event.conf_int().loc[f'elig_x_{y}', 0] for y in event_years],
    'CI_Upper': [model_event.conf_int().loc[f'elig_x_{y}', 1] for y in event_years]
}
event_df = pd.DataFrame(event_study_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save descriptive statistics
desc_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': 'mean',
    'male': 'mean',
    'married': 'mean',
    'hs_or_more': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

# Save sample counts by year
year_counts = df_analysis.groupby(['YEAR', 'daca_eligible']).size().unstack(fill_value=0)
year_counts.to_csv('sample_by_year.csv')
print("Sample counts by year saved to sample_by_year.csv")

# ============================================================================
# STEP 12: FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY: PREFERRED ESTIMATE")
print("="*80)

print(f"""
PREFERRED SPECIFICATION: Model 3 (Year Fixed Effects with Controls)

Research Question:
What is the causal effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born non-citizens?

Sample:
- Hispanic-Mexican ethnicity (HISPAN = 1)
- Born in Mexico (BPL = 200)
- Non-citizens (CITIZEN = 3)
- Working age (18-64)
- Years: 2006-2011 (pre) and 2013-2016 (post), excluding 2012

DACA Eligibility Criteria:
- Under 31 on June 15, 2012
- Arrived in US before age 16
- In US since at least 2007

Treatment Effect (DiD Coefficient): {model3.params['did']:.6f}
Standard Error (clustered by state): {model3.bse['did']:.6f}
95% Confidence Interval: [{model3.conf_int().loc['did', 0]:.6f}, {model3.conf_int().loc['did', 1]:.6f}]
p-value: {model3.pvalues['did']:.6f}
Sample Size: {int(model3.nobs):,}

Interpretation:
DACA eligibility is associated with a {model3.params['did']*100:.2f} percentage point
{'increase' if model3.params['did'] > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if model3.pvalues['did'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

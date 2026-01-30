"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.
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
# STEP 1: Load Data
# =============================================================================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {list(df.columns)}")

# =============================================================================
# STEP 2: Define Sample Selection Criteria
# =============================================================================
print("\n" + "="*80)
print("STEP 2: SAMPLE SELECTION")
print("="*80)

# Initial counts
print(f"\nInitial sample size: {len(df):,}")

# Step 2a: Hispanic-Mexican ethnicity
# HISPAN = 1 is Mexican (general), HISPAND 100-107 are Mexican detailed categories
df_hisp = df[(df['HISPAN'] == 1) | ((df['HISPAND'] >= 100) & (df['HISPAND'] <= 107))]
print(f"After Hispanic-Mexican filter: {len(df_hisp):,}")

# Step 2b: Born in Mexico
# BPL = 200 is Mexico, BPLD = 20000 is Mexico detailed
df_mex = df_hisp[(df_hisp['BPL'] == 200) | (df_hisp['BPLD'] == 20000)]
print(f"After Mexico birthplace filter: {len(df_mex):,}")

# Step 2c: Non-citizen (CITIZEN = 3)
# This is our proxy for undocumented status
df_noncit = df_mex[df_mex['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_noncit):,}")

# Step 2d: Arrived before age 16
# Calculate age at immigration
df_noncit = df_noncit.copy()
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
df_arrived_young = df_noncit[df_noncit['age_at_immig'] < 16]
print(f"After arrived before age 16 filter: {len(df_arrived_young):,}")

# Step 2e: Arrived by 2007 (lived continuously in US since June 15, 2007)
# Being conservative: YRIMMIG <= 2006 ensures they arrived by end of 2006
# Also allow 2007 since June 15 2007 is the cutoff
df_arrived_early = df_arrived_young[(df_arrived_young['YRIMMIG'] <= 2007) & (df_arrived_young['YRIMMIG'] > 0)]
print(f"After arrived by 2007 filter: {len(df_arrived_early):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups
# =============================================================================
print("\n" + "="*80)
print("STEP 3: DEFINE TREATMENT AND CONTROL GROUPS")
print("="*80)

# Treatment: Ages 26-30 as of June 15, 2012
# Born between June 16, 1981 and June 15, 1986
# Since we don't have exact birth dates, use birth year as proxy:
# - Born 1982-1985: definitely 26-30 on June 15, 2012
# - Born 1981: could be 30 or 31, include as treatment (conservative)
# - Born 1986: could be 25 or 26, include as treatment

# Control: Ages 31-35 as of June 15, 2012
# Born between June 16, 1976 and June 15, 1981
# - Born 1977-1980: definitely 31-35 on June 15, 2012
# - Born 1976: could be 35 or 36, include as control
# - Born 1981: borderline - could be 30 or 31

# To avoid overlap, let's be precise:
# Treatment: birthyr 1982-1986 (ages 26-30 in 2012)
# Control: birthyr 1977-1981 (ages 31-35 in 2012)
# Note: birthyr 1981 could be in either group depending on birth month
# We'll assign 1981 to control since they would be 31 in most of 2012

df_sample = df_arrived_early.copy()

# Calculate age as of June 2012
# A person born in year Y is approximately (2012 - Y) years old in 2012
df_sample['age_june2012'] = 2012 - df_sample['BIRTHYR']

# Treatment group: ages 26-30 as of 2012 (birth years 1982-1986)
# Control group: ages 31-35 as of 2012 (birth years 1977-1981)
df_sample['treatment'] = ((df_sample['BIRTHYR'] >= 1982) & (df_sample['BIRTHYR'] <= 1986)).astype(int)
df_sample['control'] = ((df_sample['BIRTHYR'] >= 1977) & (df_sample['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control observations
df_analysis = df_sample[(df_sample['treatment'] == 1) | (df_sample['control'] == 1)].copy()
print(f"Sample with treatment or control status: {len(df_analysis):,}")
print(f"  Treatment group (born 1982-1986): {df_analysis['treatment'].sum():,}")
print(f"  Control group (born 1977-1981): {df_analysis['control'].sum():,}")

# =============================================================================
# STEP 4: Define Time Periods
# =============================================================================
print("\n" + "="*80)
print("STEP 4: DEFINE TIME PERIODS")
print("="*80)

# Exclude 2012 (DACA implemented mid-year)
# Pre-period: 2006-2011
# Post-period: 2013-2016

df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

print(f"After excluding 2012: {len(df_analysis):,}")
print(f"\nObservations by year:")
print(df_analysis.groupby('YEAR').size())

print(f"\nPre-period (2006-2011): {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post'] == 1).sum():,}")

# =============================================================================
# STEP 5: Create Outcome Variable
# =============================================================================
print("\n" + "="*80)
print("STEP 5: CREATE OUTCOME VARIABLE")
print("="*80)

# Full-time employment: UHRSWORK >= 35
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate (overall): {df_analysis['fulltime'].mean():.4f}")
print(f"Full-time employment rate (treatment): {df_analysis[df_analysis['treatment']==1]['fulltime'].mean():.4f}")
print(f"Full-time employment rate (control): {df_analysis[df_analysis['control']==1]['fulltime'].mean():.4f}")

# =============================================================================
# STEP 6: Create DiD Interaction Term
# =============================================================================
print("\n" + "="*80)
print("STEP 6: DIFFERENCE-IN-DIFFERENCES SETUP")
print("="*80)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

# Summary table
print("\nMean full-time employment by group and period:")
summary = df_analysis.groupby(['treatment', 'post'])['fulltime'].agg(['mean', 'count', 'std'])
print(summary)

# Calculate simple DiD manually
pre_treat = df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==1)]['fulltime'].mean()

did_manual = (post_treat - pre_treat) - (post_control - pre_control)

print(f"\n--- Simple DiD Calculation ---")
print(f"Treatment group pre: {pre_treat:.4f}")
print(f"Treatment group post: {post_treat:.4f}")
print(f"Treatment change: {post_treat - pre_treat:.4f}")
print(f"\nControl group pre: {pre_control:.4f}")
print(f"Control group post: {post_control:.4f}")
print(f"Control change: {post_control - pre_control:.4f}")
print(f"\nDiD estimate (manual): {did_manual:.4f}")

# =============================================================================
# STEP 7: Regression Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 7: REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD (No controls) ---")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df_analysis).fit()
print(model1.summary())

# Model 2: DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model2 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post', data=df_analysis).fit()
print(model2.summary())

# Model 3: DiD with covariates
print("\n--- Model 3: DiD with Controls ---")
# Add control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)  # 1=married spouse present, 2=married spouse absent

# Education categories
df_analysis['educ_hs'] = ((df_analysis['EDUCD'] >= 62) & (df_analysis['EDUCD'] <= 64)).astype(int)  # HS graduate
df_analysis['educ_somecol'] = ((df_analysis['EDUCD'] >= 65) & (df_analysis['EDUCD'] <= 100)).astype(int)  # Some college
df_analysis['educ_ba_plus'] = (df_analysis['EDUCD'] >= 101).astype(int)  # BA or higher

model3 = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                 data=df_analysis).fit()
print(model3.summary())

# Model 4: DiD with state fixed effects and controls
print("\n--- Model 4: DiD with State and Year Fixed Effects + Controls ---")
model4 = smf.ols('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                 data=df_analysis).fit()
# Print only key coefficients
print("\nKey coefficients from Model 4:")
print(f"treat_post coefficient: {model4.params['treat_post']:.6f}")
print(f"treat_post std error: {model4.bse['treat_post']:.6f}")
print(f"treat_post t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"treat_post p-value: {model4.pvalues['treat_post']:.6f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.6f}, {model4.conf_int().loc['treat_post', 1]:.6f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {int(model4.nobs)}")

# =============================================================================
# STEP 8: Robustness Checks
# =============================================================================
print("\n" + "="*80)
print("STEP 8: ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Using person weights
print("\n--- Robustness 1: Weighted regression ---")
model_weighted = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                         data=df_analysis, weights=df_analysis['PERWT']).fit()
print(f"treat_post coefficient (weighted): {model_weighted.params['treat_post']:.6f}")
print(f"treat_post std error: {model_weighted.bse['treat_post']:.6f}")
print(f"95% CI: [{model_weighted.conf_int().loc['treat_post', 0]:.6f}, {model_weighted.conf_int().loc['treat_post', 1]:.6f}]")

# Robustness 2: Probit model
print("\n--- Robustness 2: Probit model ---")
try:
    probit_model = smf.probit('fulltime ~ treatment + post + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                              data=df_analysis).fit(disp=0)
    print(f"treat_post coefficient (probit): {probit_model.params['treat_post']:.6f}")
    print(f"treat_post std error: {probit_model.bse['treat_post']:.6f}")
    # Marginal effect at means
    margeff = probit_model.get_margeff(at='mean')
    print(f"Marginal effect of treat_post: {margeff.margeff[margeff.results.model.exog_names.index('treat_post')]:.6f}")
except Exception as e:
    print(f"Probit model error: {e}")

# Robustness 3: Logit model
print("\n--- Robustness 3: Logit model ---")
try:
    logit_model = smf.logit('fulltime ~ treatment + post + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                            data=df_analysis).fit(disp=0)
    print(f"treat_post coefficient (logit): {logit_model.params['treat_post']:.6f}")
    print(f"treat_post std error: {logit_model.bse['treat_post']:.6f}")
    margeff_logit = logit_model.get_margeff(at='mean')
    print(f"Marginal effect of treat_post: {margeff_logit.margeff[margeff_logit.results.model.exog_names.index('treat_post')]:.6f}")
except Exception as e:
    print(f"Logit model error: {e}")

# Robustness 4: Clustered standard errors at state level
print("\n--- Robustness 4: State-clustered standard errors ---")
model_clustered = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post + female + married + educ_hs + educ_somecol + educ_ba_plus',
                          data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"treat_post coefficient: {model_clustered.params['treat_post']:.6f}")
print(f"treat_post std error (clustered): {model_clustered.bse['treat_post']:.6f}")
print(f"95% CI: [{model_clustered.conf_int().loc['treat_post', 0]:.6f}, {model_clustered.conf_int().loc['treat_post', 1]:.6f}]")

# =============================================================================
# STEP 9: Parallel Trends Check
# =============================================================================
print("\n" + "="*80)
print("STEP 9: PARALLEL TRENDS CHECK")
print("="*80)

# Event study specification - year-by-treatment interactions
# Using 2011 as reference year
df_analysis['treat_2006'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['treat_2007'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['treat_2008'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['treat_2009'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['treat_2010'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference
df_analysis['treat_2013'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['treat_2014'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['treat_2015'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['treat_2016'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2016).astype(int)

event_study = smf.ols('fulltime ~ treatment + C(YEAR) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + female + married + educ_hs + educ_somecol + educ_ba_plus',
                      data=df_analysis).fit()

print("Event Study Coefficients (relative to 2011):")
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_vars:
    coef = event_study.params[var]
    se = event_study.bse[var]
    ci_low = event_study.conf_int().loc[var, 0]
    ci_high = event_study.conf_int().loc[var, 1]
    pval = event_study.pvalues[var]
    print(f"{var}: {coef:.4f} (SE: {se:.4f}) [95% CI: {ci_low:.4f}, {ci_high:.4f}] p={pval:.4f}")

# =============================================================================
# STEP 10: Heterogeneity Analysis
# =============================================================================
print("\n" + "="*80)
print("STEP 10: HETEROGENEITY ANALYSIS")
print("="*80)

# By gender
print("\n--- By Gender ---")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex]
    model_sex = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post', data=df_sex).fit()
    print(f"{label}: treat_post = {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f}), N={len(df_sex)}")

# By education
print("\n--- By Education Level ---")
df_analysis['educ_level'] = 'Less than HS'
df_analysis.loc[df_analysis['educ_hs'] == 1, 'educ_level'] = 'High School'
df_analysis.loc[df_analysis['educ_somecol'] == 1, 'educ_level'] = 'Some College'
df_analysis.loc[df_analysis['educ_ba_plus'] == 1, 'educ_level'] = 'BA or Higher'

for level in ['Less than HS', 'High School', 'Some College', 'BA or Higher']:
    df_educ = df_analysis[df_analysis['educ_level'] == level]
    if len(df_educ) > 100:
        model_educ = smf.ols('fulltime ~ treatment + C(YEAR) + treat_post', data=df_educ).fit()
        print(f"{level}: treat_post = {model_educ.params['treat_post']:.4f} (SE: {model_educ.bse['treat_post']:.4f}), N={len(df_educ)}")
    else:
        print(f"{level}: N={len(df_educ)} (too few observations)")

# =============================================================================
# STEP 11: Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("STEP 11: SUMMARY STATISTICS")
print("="*80)

print("\n--- Full Sample Summary ---")
print(f"Total observations: {len(df_analysis):,}")
print(f"Treatment group: {df_analysis['treatment'].sum():,}")
print(f"Control group: {(df_analysis['treatment'] == 0).sum():,}")
print(f"Pre-period observations: {(df_analysis['post'] == 0).sum():,}")
print(f"Post-period observations: {(df_analysis['post'] == 1).sum():,}")

print("\n--- Demographic Characteristics ---")
desc_vars = ['fulltime', 'female', 'married', 'AGE', 'educ_hs', 'educ_somecol', 'educ_ba_plus']
for var in desc_vars:
    if var in df_analysis.columns:
        print(f"{var}: mean={df_analysis[var].mean():.4f}, std={df_analysis[var].std():.4f}")

print("\n--- Characteristics by Group ---")
print("\nTreatment group (pre-period):")
treat_pre = df_analysis[(df_analysis['treatment']==1) & (df_analysis['post']==0)]
for var in desc_vars:
    if var in treat_pre.columns:
        print(f"  {var}: {treat_pre[var].mean():.4f}")

print("\nControl group (pre-period):")
control_pre = df_analysis[(df_analysis['treatment']==0) & (df_analysis['post']==0)]
for var in desc_vars:
    if var in control_pre.columns:
        print(f"  {var}: {control_pre[var].mean():.4f}")

# =============================================================================
# STEP 12: Save Key Results
# =============================================================================
print("\n" + "="*80)
print("STEP 12: KEY RESULTS SUMMARY")
print("="*80)

print("\n" + "="*80)
print("PREFERRED ESTIMATE (Model 3 with year FE and controls)")
print("="*80)
print(f"DiD Estimate (treat_post): {model3.params['treat_post']:.6f}")
print(f"Standard Error: {model3.bse['treat_post']:.6f}")
print(f"95% Confidence Interval: [{model3.conf_int().loc['treat_post', 0]:.6f}, {model3.conf_int().loc['treat_post', 1]:.6f}]")
print(f"t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"p-value: {model3.pvalues['treat_post']:.6f}")
print(f"Sample Size: {int(model3.nobs):,}")
print(f"R-squared: {model3.rsquared:.4f}")

# Save results to file
results_dict = {
    'preferred_estimate': model3.params['treat_post'],
    'std_error': model3.bse['treat_post'],
    'ci_lower': model3.conf_int().loc['treat_post', 0],
    'ci_upper': model3.conf_int().loc['treat_post', 1],
    't_stat': model3.tvalues['treat_post'],
    'p_value': model3.pvalues['treat_post'],
    'n_obs': int(model3.nobs),
    'r_squared': model3.rsquared
}

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Save the analysis dataframe for potential further use
df_analysis.to_csv('analysis_sample.csv', index=False)
print("\nAnalysis sample saved to analysis_sample.csv")

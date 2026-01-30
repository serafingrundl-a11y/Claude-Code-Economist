"""
DACA Replication Study - Analysis Script
Participant 29

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals.

Treatment: Ages 26-30 on June 15, 2012
Control: Ages 31-35 on June 15, 2012
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_29")

print("=" * 60)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 60)

# =============================================================================
# STEP 1: Load Data with Memory Optimization
# =============================================================================
print("\n[1] Loading data...")

# Define columns we need
needed_cols = [
    'YEAR', 'PERWT', 'BIRTHYR', 'BIRTHQTR', 'HISPAN', 'HISPAND',
    'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'UHRSWORK', 'EMPSTAT',
    'AGE', 'SEX', 'EDUC', 'MARST', 'STATEFIP', 'NCHILD'
]

# Define dtypes to reduce memory
dtype_dict = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'AGE': 'int16',
    'SEX': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'STATEFIP': 'int8',
    'NCHILD': 'int8'
}

# Load data
df = pd.read_csv('data/data.csv', usecols=needed_cols, dtype=dtype_dict)
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: Filter to Target Population
# =============================================================================
print("\n[2] Filtering to target population...")

# Filter 1: Hispanic-Mexican ethnicity (HISPAN == 1)
df = df[df['HISPAN'] == 1]
print(f"After Hispanic-Mexican filter: {len(df):,}")

# Filter 2: Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
print(f"After born in Mexico filter: {len(df):,}")

# Filter 3: Not a citizen (CITIZEN == 3)
# This is our proxy for undocumented status
df = df[df['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df):,}")

# Filter 4: Exclude years before 2006 and after 2016
df = df[(df['YEAR'] >= 2006) & (df['YEAR'] <= 2016)]
print(f"After year filter (2006-2016): {len(df):,}")

# Filter 5: Exclude 2012 (cannot distinguish pre/post DACA)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# =============================================================================
# STEP 3: Determine Age on June 15, 2012 and Filter to Age Groups
# =============================================================================
print("\n[3] Determining age on June 15, 2012 and filtering to age groups...")

# Calculate age on June 15, 2012
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q1 or Q2 (before July), they had their birthday by June 15
# If born in Q3 or Q4 (July onwards), they hadn't had their birthday yet

# Age = 2012 - BIRTHYR, but subtract 1 if birthday hadn't occurred yet (Q3 or Q4)
df['age_june_2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june_2012'] -= 1

# Create treatment and control groups
# Treatment: ages 26-30 on June 15, 2012
# Control: ages 31-35 on June 15, 2012
df['treated'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)

# Keep only treatment or control group
df = df[(df['treated'] == 1) | (df['control'] == 1)]
print(f"After age group filter (26-35): {len(df):,}")

# =============================================================================
# STEP 4: Apply DACA Eligibility Criteria (other than age)
# =============================================================================
print("\n[4] Applying DACA eligibility criteria...")

# Criterion: Arrived in US before age 16
# Need YRIMMIG > 0 (not N/A)
df = df[df['YRIMMIG'] > 0]
print(f"After requiring valid YRIMMIG: {len(df):,}")

# Calculate age at arrival
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
print(f"After arrived before age 16: {len(df):,}")

# Criterion: Continuous presence since June 15, 2007
# We approximate this by YRIMMIG <= 2007
df = df[df['YRIMMIG'] <= 2007]
print(f"After arrival by 2007: {len(df):,}")

# =============================================================================
# STEP 5: Create Outcome and Treatment Variables
# =============================================================================
print("\n[5] Creating outcome and treatment variables...")

# Post-treatment indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# DiD interaction term
df['treat_x_post'] = df['treated'] * df['post']

print(f"\nFinal sample size: {len(df):,}")
print(f"Treatment group: {df['treated'].sum():,}")
print(f"Control group: {df['control'].sum():,}")

# =============================================================================
# STEP 6: Summary Statistics
# =============================================================================
print("\n[6] Summary Statistics")
print("=" * 60)

# Overall summary by treatment status and period
summary = df.groupby(['treated', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'UHRSWORK': 'mean',
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary by Treatment Status and Period:")
print(summary)

# Pre-treatment comparison
pre_treat = df[(df['treated'] == 1) & (df['post'] == 0)]
pre_control = df[(df['control'] == 1) & (df['post'] == 0)]

print("\n" + "=" * 60)
print("Pre-Treatment Period Balance Check:")
print("-" * 60)
print(f"Treatment group (26-30) - Full-time rate: {pre_treat['fulltime'].mean():.4f}")
print(f"Control group (31-35) - Full-time rate: {pre_control['fulltime'].mean():.4f}")
print(f"Difference: {pre_treat['fulltime'].mean() - pre_control['fulltime'].mean():.4f}")

# =============================================================================
# STEP 7: DiD Estimation - Simple Model
# =============================================================================
print("\n[7] Difference-in-Differences Estimation")
print("=" * 60)

# Calculate simple DiD
pre_treat_ft = df[(df['treated'] == 1) & (df['post'] == 0)]['fulltime'].mean()
post_treat_ft = df[(df['treated'] == 1) & (df['post'] == 1)]['fulltime'].mean()
pre_control_ft = df[(df['treated'] == 0) & (df['post'] == 0)]['fulltime'].mean()
post_control_ft = df[(df['treated'] == 0) & (df['post'] == 1)]['fulltime'].mean()

did_simple = (post_treat_ft - pre_treat_ft) - (post_control_ft - pre_control_ft)

print("\nSimple DiD Calculation (Unweighted):")
print(f"Treatment Before: {pre_treat_ft:.4f}")
print(f"Treatment After: {post_treat_ft:.4f}")
print(f"Treatment Change: {post_treat_ft - pre_treat_ft:.4f}")
print(f"Control Before: {pre_control_ft:.4f}")
print(f"Control After: {post_control_ft:.4f}")
print(f"Control Change: {post_control_ft - pre_control_ft:.4f}")
print(f"\nDiD Estimate: {did_simple:.4f}")

# =============================================================================
# STEP 8: DiD Regression - Basic Model
# =============================================================================
print("\n[8] DiD Regression - Basic Model (Unweighted)")
print("-" * 60)

# Basic DiD regression
model1 = smf.ols('fulltime ~ treated + post + treat_x_post', data=df).fit()
print(model1.summary())

# =============================================================================
# STEP 9: DiD Regression - Weighted Model
# =============================================================================
print("\n[9] DiD Regression - Weighted Model")
print("-" * 60)

# Weighted regression using PERWT
model2 = smf.wls('fulltime ~ treated + post + treat_x_post',
                 data=df,
                 weights=df['PERWT']).fit()
print(model2.summary())

# =============================================================================
# STEP 10: DiD Regression with Covariates
# =============================================================================
print("\n[10] DiD Regression - With Covariates (Weighted)")
print("-" * 60)

# Create dummy variables for covariates
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'] == 1).astype(int)
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Education categories
df['educ_hs'] = ((df['EDUC'] >= 6) & (df['EDUC'] <= 7)).astype(int)  # High school
df['educ_somecol'] = ((df['EDUC'] >= 8) & (df['EDUC'] <= 9)).astype(int)  # Some college
df['educ_college'] = (df['EDUC'] >= 10).astype(int)  # College+

# Year fixed effects
df['year_2007'] = (df['YEAR'] == 2007).astype(int)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2011'] = (df['YEAR'] == 2011).astype(int)
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Model with covariates
model3 = smf.wls('fulltime ~ treated + post + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college',
                 data=df,
                 weights=df['PERWT']).fit()
print(model3.summary())

# =============================================================================
# STEP 11: DiD with Year Fixed Effects
# =============================================================================
print("\n[11] DiD Regression - With Year Fixed Effects (Weighted)")
print("-" * 60)

model4 = smf.wls('fulltime ~ treated + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college + year_2007 + year_2008 + year_2009 + year_2010 + year_2011 + year_2013 + year_2014 + year_2015 + year_2016',
                 data=df,
                 weights=df['PERWT']).fit()
print(model4.summary())

# =============================================================================
# STEP 12: Robust Standard Errors
# =============================================================================
print("\n[12] DiD with Heteroskedasticity-Robust Standard Errors")
print("-" * 60)

model5 = smf.wls('fulltime ~ treated + post + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college',
                 data=df,
                 weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# =============================================================================
# STEP 13: State Fixed Effects
# =============================================================================
print("\n[13] DiD Regression - With State Fixed Effects")
print("-" * 60)

# State fixed effects model
model6 = smf.wls('fulltime ~ treated + post + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college + C(STATEFIP)',
                 data=df,
                 weights=df['PERWT']).fit(cov_type='HC1')

print(f"DiD Coefficient (treat_x_post): {model6.params['treat_x_post']:.4f}")
print(f"Standard Error: {model6.bse['treat_x_post']:.4f}")
print(f"t-statistic: {model6.tvalues['treat_x_post']:.4f}")
print(f"p-value: {model6.pvalues['treat_x_post']:.4f}")
conf_int = model6.conf_int().loc['treat_x_post']
print(f"95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs):,}")

# =============================================================================
# STEP 14: Create Summary Tables
# =============================================================================
print("\n[14] Creating Summary Tables for Report")
print("=" * 60)

# Create results summary
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'Weighted', 'With Covariates', 'Year FE', 'Robust SE', 'State FE'],
    'Coefficient': [model1.params['treat_x_post'],
                   model2.params['treat_x_post'],
                   model3.params['treat_x_post'],
                   model4.params['treat_x_post'],
                   model5.params['treat_x_post'],
                   model6.params['treat_x_post']],
    'Std_Error': [model1.bse['treat_x_post'],
                  model2.bse['treat_x_post'],
                  model3.bse['treat_x_post'],
                  model4.bse['treat_x_post'],
                  model5.bse['treat_x_post'],
                  model6.bse['treat_x_post']],
    'p_value': [model1.pvalues['treat_x_post'],
                model2.pvalues['treat_x_post'],
                model3.pvalues['treat_x_post'],
                model4.pvalues['treat_x_post'],
                model5.pvalues['treat_x_post'],
                model6.pvalues['treat_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)]
})

print("\nResults Summary Table:")
print(results_summary.to_string(index=False))

# Save results
results_summary.to_csv('results_summary.csv', index=False)

# =============================================================================
# STEP 15: Create Visualizations
# =============================================================================
print("\n[15] Creating Visualizations")
print("-" * 60)

# Figure 1: Parallel Trends Plot
fig, ax = plt.subplots(figsize=(10, 6))

yearly_means = df.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']
yearly_means.plot(ax=ax, marker='o', linewidth=2)

ax.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure1_parallel_trends.png")

# Figure 2: Difference Over Time
fig, ax = plt.subplots(figsize=(10, 6))

yearly_diff = yearly_means['Treatment (26-30)'] - yearly_means['Control (31-35)']
yearly_diff.plot(ax=ax, marker='s', linewidth=2, color='green')

ax.axvline(x=2012.5, color='red', linestyle='--', label='DACA Implementation')
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference (Treatment - Control)', fontsize=12)
ax.set_title('Difference in Full-Time Employment Rate\n(Treatment - Control)', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_difference.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure2_difference.png")

# Figure 3: Sample Size by Year
fig, ax = plt.subplots(figsize=(10, 6))

sample_by_year = df.groupby(['YEAR', 'treated']).size().unstack()
sample_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']
sample_by_year.plot(kind='bar', ax=ax, width=0.8)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sample Size', fontsize=12)
ax.set_title('Sample Size by Year and Treatment Status', fontsize=14)
ax.legend(loc='best')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('figure3_sample_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure3_sample_size.png")

# =============================================================================
# STEP 16: Balance Table
# =============================================================================
print("\n[16] Creating Balance Table")
print("-" * 60)

# Pre-treatment balance
pre_df = df[df['post'] == 0]

balance_vars = ['AGE', 'female', 'married', 'has_children',
                'educ_hs', 'educ_somecol', 'educ_college',
                'fulltime', 'employed', 'UHRSWORK']

balance_table = []
for var in balance_vars:
    treat_mean = np.average(pre_df[pre_df['treated']==1][var],
                           weights=pre_df[pre_df['treated']==1]['PERWT'])
    control_mean = np.average(pre_df[pre_df['treated']==0][var],
                             weights=pre_df[pre_df['treated']==0]['PERWT'])
    diff = treat_mean - control_mean
    balance_table.append({
        'Variable': var,
        'Treatment': round(treat_mean, 4),
        'Control': round(control_mean, 4),
        'Difference': round(diff, 4)
    })

balance_df = pd.DataFrame(balance_table)
print("\nPre-Treatment Balance Table:")
print(balance_df.to_string(index=False))
balance_df.to_csv('balance_table.csv', index=False)

# =============================================================================
# STEP 17: Event Study Analysis
# =============================================================================
print("\n[17] Event Study Analysis")
print("-" * 60)

# Create year dummies for event study (relative to 2011)
df['treat_x_2006'] = df['treated'] * (df['YEAR'] == 2006).astype(int)
df['treat_x_2007'] = df['treated'] * (df['YEAR'] == 2007).astype(int)
df['treat_x_2008'] = df['treated'] * (df['YEAR'] == 2008).astype(int)
df['treat_x_2009'] = df['treated'] * (df['YEAR'] == 2009).astype(int)
df['treat_x_2010'] = df['treated'] * (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['treat_x_2013'] = df['treated'] * (df['YEAR'] == 2013).astype(int)
df['treat_x_2014'] = df['treated'] * (df['YEAR'] == 2014).astype(int)
df['treat_x_2015'] = df['treated'] * (df['YEAR'] == 2015).astype(int)
df['treat_x_2016'] = df['treated'] * (df['YEAR'] == 2016).astype(int)

event_model = smf.wls('fulltime ~ treated + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + year_2007 + year_2008 + year_2009 + year_2010 + year_2011 + year_2013 + year_2014 + year_2015 + year_2016 + female + married + has_children + educ_hs + educ_somecol + educ_college',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract event study coefficients
event_vars = ['treat_x_2006', 'treat_x_2007', 'treat_x_2008', 'treat_x_2009',
              'treat_x_2010', 'treat_x_2013', 'treat_x_2014', 'treat_x_2015', 'treat_x_2016']
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
coefs = [event_model.params[v] for v in event_vars]
ses = [event_model.bse[v] for v in event_vars]

# Add 2011 as reference (0)
years.insert(5, 2011)
coefs.insert(5, 0)
ses.insert(5, 0)

# Event Study Plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.errorbar(years, coefs, yerr=[1.96*se for se in ses],
            fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation (June 2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Treatment x Year)', fontsize=12)
ax.set_title('Event Study: Effect of DACA on Full-Time Employment', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure4_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure4_event_study.png")

# Print event study results
print("\nEvent Study Coefficients:")
event_results = pd.DataFrame({
    'Year': years,
    'Coefficient': coefs,
    'Std_Error': ses
})
print(event_results.to_string(index=False))
event_results.to_csv('event_study_results.csv', index=False)

# =============================================================================
# STEP 18: Heterogeneity Analysis
# =============================================================================
print("\n[18] Heterogeneity Analysis")
print("-" * 60)

# By Gender
print("\nBy Gender:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    sub_df = df[df['SEX'] == sex]
    model = smf.wls('fulltime ~ treated + post + treat_x_post',
                    data=sub_df, weights=sub_df['PERWT']).fit(cov_type='HC1')
    print(f"  {label}: DiD = {model.params['treat_x_post']:.4f} (SE: {model.bse['treat_x_post']:.4f}, p={model.pvalues['treat_x_post']:.4f})")

# By Education
print("\nBy Education:")
# Less than HS
sub_df = df[df['EDUC'] < 6]
if len(sub_df) > 100:
    model = smf.wls('fulltime ~ treated + post + treat_x_post',
                    data=sub_df, weights=sub_df['PERWT']).fit(cov_type='HC1')
    print(f"  Less than HS: DiD = {model.params['treat_x_post']:.4f} (SE: {model.bse['treat_x_post']:.4f}, p={model.pvalues['treat_x_post']:.4f})")

# HS or more
sub_df = df[df['EDUC'] >= 6]
if len(sub_df) > 100:
    model = smf.wls('fulltime ~ treated + post + treat_x_post',
                    data=sub_df, weights=sub_df['PERWT']).fit(cov_type='HC1')
    print(f"  HS or more: DiD = {model.params['treat_x_post']:.4f} (SE: {model.bse['treat_x_post']:.4f}, p={model.pvalues['treat_x_post']:.4f})")

# =============================================================================
# STEP 19: Robustness Checks
# =============================================================================
print("\n[19] Robustness Checks")
print("-" * 60)

# Alternative outcome: Employed (any work)
print("\nAlternative Outcome - Employed (any employment):")
model_emp = smf.wls('employed ~ treated + post + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD = {model_emp.params['treat_x_post']:.4f} (SE: {model_emp.bse['treat_x_post']:.4f}, p={model_emp.pvalues['treat_x_post']:.4f})")

# Alternative outcome: Hours worked (continuous)
print("\nAlternative Outcome - Hours Worked (continuous):")
model_hrs = smf.wls('UHRSWORK ~ treated + post + treat_x_post + female + married + has_children + educ_hs + educ_somecol + educ_college',
                    data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"  DiD = {model_hrs.params['treat_x_post']:.4f} (SE: {model_hrs.bse['treat_x_post']:.4f}, p={model_hrs.pvalues['treat_x_post']:.4f})")

# Alternative age bandwidth (24-32 vs 33-40)
print("\nAlternative Age Bandwidth (wider):")
df_alt = pd.read_csv('data/data.csv', usecols=needed_cols, dtype=dtype_dict)
df_alt = df_alt[df_alt['HISPAN'] == 1]
df_alt = df_alt[df_alt['BPL'] == 200]
df_alt = df_alt[df_alt['CITIZEN'] == 3]
df_alt = df_alt[(df_alt['YEAR'] >= 2006) & (df_alt['YEAR'] <= 2016)]
df_alt = df_alt[df_alt['YEAR'] != 2012]
df_alt['age_june_2012'] = 2012 - df_alt['BIRTHYR']
df_alt.loc[df_alt['BIRTHQTR'].isin([3, 4]), 'age_june_2012'] -= 1
df_alt = df_alt[df_alt['YRIMMIG'] > 0]
df_alt['age_at_arrival'] = df_alt['YRIMMIG'] - df_alt['BIRTHYR']
df_alt = df_alt[df_alt['age_at_arrival'] < 16]
df_alt = df_alt[df_alt['YRIMMIG'] <= 2007]

# Wider age bands: 24-32 (treatment) vs 33-40 (control)
df_alt['treated'] = ((df_alt['age_june_2012'] >= 24) & (df_alt['age_june_2012'] <= 32)).astype(int)
df_alt['control'] = ((df_alt['age_june_2012'] >= 33) & (df_alt['age_june_2012'] <= 40)).astype(int)
df_alt = df_alt[(df_alt['treated'] == 1) | (df_alt['control'] == 1)]
df_alt['post'] = (df_alt['YEAR'] >= 2013).astype(int)
df_alt['fulltime'] = (df_alt['UHRSWORK'] >= 35).astype(int)
df_alt['treat_x_post'] = df_alt['treated'] * df_alt['post']

if len(df_alt) > 100:
    model_alt = smf.wls('fulltime ~ treated + post + treat_x_post',
                        data=df_alt, weights=df_alt['PERWT']).fit(cov_type='HC1')
    print(f"  DiD (24-32 vs 33-40) = {model_alt.params['treat_x_post']:.4f} (SE: {model_alt.bse['treat_x_post']:.4f}, p={model_alt.pvalues['treat_x_post']:.4f})")

# =============================================================================
# STEP 20: Final Summary
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

# Preferred specification: Model with covariates, weighted, robust SE
preferred = model5

print(f"\nPreferred Specification: Weighted DiD with Covariates and Robust SE")
print(f"DiD Coefficient: {preferred.params['treat_x_post']:.4f}")
print(f"Standard Error: {preferred.bse['treat_x_post']:.4f}")
print(f"t-statistic: {preferred.tvalues['treat_x_post']:.4f}")
print(f"p-value: {preferred.pvalues['treat_x_post']:.4f}")
conf_int = preferred.conf_int().loc['treat_x_post']
print(f"95% Confidence Interval: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")
print(f"Sample Size: {int(preferred.nobs):,}")

# Calculate weighted sample sizes
weighted_n = df['PERWT'].sum()
weighted_treat = df[df['treated']==1]['PERWT'].sum()
weighted_control = df[df['treated']==0]['PERWT'].sum()
print(f"\nWeighted Sample Sizes:")
print(f"  Total: {weighted_n:,.0f}")
print(f"  Treatment Group: {weighted_treat:,.0f}")
print(f"  Control Group: {weighted_control:,.0f}")

# Save final results to file
with open('final_results.txt', 'w') as f:
    f.write("DACA Replication Study - Final Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Preferred Specification: Weighted DiD with Covariates and Robust SE\n\n")
    f.write(f"DiD Coefficient (DACA Effect): {preferred.params['treat_x_post']:.4f}\n")
    f.write(f"Standard Error: {preferred.bse['treat_x_post']:.4f}\n")
    f.write(f"t-statistic: {preferred.tvalues['treat_x_post']:.4f}\n")
    f.write(f"p-value: {preferred.pvalues['treat_x_post']:.4f}\n")
    f.write(f"95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]\n")
    f.write(f"Sample Size: {int(preferred.nobs):,}\n\n")
    f.write("Interpretation:\n")
    if preferred.pvalues['treat_x_post'] < 0.05:
        if preferred.params['treat_x_post'] > 0:
            f.write("DACA eligibility is associated with a statistically significant INCREASE\n")
            f.write(f"in full-time employment of {preferred.params['treat_x_post']*100:.2f} percentage points.\n")
        else:
            f.write("DACA eligibility is associated with a statistically significant DECREASE\n")
            f.write(f"in full-time employment of {abs(preferred.params['treat_x_post'])*100:.2f} percentage points.\n")
    else:
        f.write("The effect of DACA eligibility on full-time employment is NOT statistically\n")
        f.write(f"significant at the 5% level (p={preferred.pvalues['treat_x_post']:.4f}).\n")

print("\n[Complete] All analyses finished successfully!")
print("Output files created:")
print("  - results_summary.csv")
print("  - balance_table.csv")
print("  - event_study_results.csv")
print("  - final_results.txt")
print("  - figure1_parallel_trends.png")
print("  - figure2_difference.png")
print("  - figure3_sample_size.png")
print("  - figure4_event_study.png")

"""
DACA Employment Effects Replication Analysis
=============================================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the US,
what was the causal impact of DACA eligibility on full-time employment (35+ hours/week)?

Treatment Group: Ages 26-30 at DACA implementation (June 15, 2012)
Control Group: Ages 31-35 at DACA implementation
Method: Difference-in-Differences

Author: Anonymous
Date: 2024
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
print("DACA EMPLOYMENT EFFECTS REPLICATION ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

# Load data
df = pd.read_csv('data/data.csv')
print(f"Total observations in raw data: {len(df):,}")
print(f"Years available: {sorted(df['YEAR'].unique())}")
print(f"Variables: {df.columns.tolist()}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n" + "=" * 80)
print("2. SAMPLE SELECTION")
print("=" * 80)

# Step 1: Hispanic-Mexican ethnicity (HISPAN == 1 for Mexican)
df_hisp_mex = df[df['HISPAN'] == 1].copy()
print(f"After filtering Hispanic-Mexican: {len(df_hisp_mex):,}")

# Step 2: Born in Mexico (BPL == 200)
df_mex_born = df_hisp_mex[df_hisp_mex['BPL'] == 200].copy()
print(f"After filtering Mexican-born: {len(df_mex_born):,}")

# Step 3: Non-citizens (CITIZEN == 3 - Not a citizen)
# Per instructions: Assume anyone who is not a citizen and who has not received
# immigration papers is undocumented for DACA purposes
df_noncit = df_mex_born[df_mex_born['CITIZEN'] == 3].copy()
print(f"After filtering non-citizens: {len(df_noncit):,}")

# Step 4: Arrived before age 16
# Need to calculate age at arrival
# YRIMMIG is year of immigration, BIRTHYR is birth year
df_noncit['age_at_arrival'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
df_arrived_young = df_noncit[df_noncit['age_at_arrival'] < 16].copy()
print(f"After filtering arrived before age 16: {len(df_arrived_young):,}")

# Step 5: Continuous residence since June 15, 2007
# This means immigration year <= 2007
df_continuous = df_arrived_young[df_arrived_young['YRIMMIG'] <= 2007].copy()
print(f"After filtering continuous residence (arrived by 2007): {len(df_continuous):,}")

# Step 6: Define age groups based on age as of June 15, 2012
# Treatment group: ages 26-30 as of June 15, 2012 (born 1982-1986)
# Control group: ages 31-35 as of June 15, 2012 (born 1977-1981)

# Calculate age as of mid-2012
df_continuous['age_june_2012'] = 2012 - df_continuous['BIRTHYR']

# For more precision, adjust for birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# June 15 is in Q2, so those born in Q3-Q4 would be 1 year younger
# But for simplicity, we'll use birth year only for age calculation

# Treatment group: age 26-30 as of June 2012 (born 1982-1986)
# Control group: age 31-35 as of June 2012 (born 1977-1981)

df_continuous['treatment'] = ((df_continuous['BIRTHYR'] >= 1982) &
                               (df_continuous['BIRTHYR'] <= 1986)).astype(int)
df_continuous['control'] = ((df_continuous['BIRTHYR'] >= 1977) &
                             (df_continuous['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df_sample = df_continuous[(df_continuous['treatment'] == 1) |
                          (df_continuous['control'] == 1)].copy()
print(f"After filtering to age groups (26-35 as of June 2012): {len(df_sample):,}")

# Step 7: Define pre/post periods
# Pre: 2006-2011 (before DACA)
# Post: 2013-2016 (after DACA - examining effects per instructions)
# Exclude 2012 because DACA was implemented mid-year

df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)
df_sample['pre_period'] = (df_sample['YEAR'] <= 2011).astype(int)

# For main analysis, exclude 2012
df_analysis = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# =============================================================================
# 3. CREATE OUTCOME VARIABLE
# =============================================================================
print("\n" + "=" * 80)
print("3. CREATE OUTCOME VARIABLE")
print("=" * 80)

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A)
# EMPSTAT: Employment status (1 = Employed)

# Full-time employment defined as employed AND working 35+ hours
df_analysis['fulltime'] = ((df_analysis['EMPSTAT'] == 1) &
                           (df_analysis['UHRSWORK'] >= 35)).astype(int)

# Alternative: just hours >= 35 among those with valid hours
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)
df_analysis['hours_35plus'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"\nOutcome variable created: fulltime (1 if employed AND 35+ hours/week)")
print(f"Full-time employment rate: {df_analysis['fulltime'].mean()*100:.2f}%")
print(f"Employment rate: {df_analysis['employed'].mean()*100:.2f}%")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Sample sizes by group and period
print("\nSample sizes by group and period:")
sample_counts = df_analysis.groupby(['treatment', 'post']).size().unstack(fill_value=0)
sample_counts.index = ['Control (31-35)', 'Treatment (26-30)']
sample_counts.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(sample_counts)

# Full-time employment rates by group and period
print("\nFull-time employment rates by group and period:")
ft_rates = df_analysis.groupby(['treatment', 'post'])['fulltime'].mean().unstack() * 100
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
print(ft_rates.round(2))

# Demographics by treatment/control
print("\nDemographics by group:")
demographics = df_analysis.groupby('treatment').agg({
    'AGE': 'mean',
    'SEX': lambda x: (x == 2).mean() * 100,  # % female
    'MARST': lambda x: (x == 1).mean() * 100,  # % married
    'EDUC': 'mean',
    'fulltime': lambda x: x.mean() * 100,
    'employed': lambda x: x.mean() * 100
}).round(2)
demographics.index = ['Control (31-35)', 'Treatment (26-30)']
demographics.columns = ['Mean Age', '% Female', '% Married', 'Mean Educ', '% Full-time', '% Employed']
print(demographics)

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("5. DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographics ---")
# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

model2 = smf.ols('fulltime ~ treatment + post + treat_post + female + married + C(EDUC)',
                 data=df_analysis).fit(cov_type='HC1')
print(f"\nDiD coefficient (treat_post): {model2.params['treat_post']:.4f}")
print(f"Standard error: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model2.pvalues['treat_post']:.4f}")
print(f"N: {int(model2.nobs)}")

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.ols('fulltime ~ treatment + treat_post + female + married + C(EDUC) + C(YEAR)',
                 data=df_analysis).fit(cov_type='HC1')
print(f"\nDiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with year and state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
model4 = smf.ols('fulltime ~ treatment + treat_post + female + married + C(EDUC) + C(YEAR) + C(STATEFIP)',
                 data=df_analysis).fit(cov_type='HC1')
print(f"\nDiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("6. ROBUSTNESS CHECKS")
print("=" * 80)

# 6.1 Placebo test: fake treatment in 2009
print("\n--- 6.1 Placebo Test (Fake Treatment in 2009) ---")
df_pre = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['fake_treat_post'] = df_pre['treatment'] * df_pre['fake_post']

placebo_model = smf.ols('fulltime ~ treatment + fake_post + fake_treat_post + female + married + C(EDUC) + C(YEAR)',
                        data=df_pre).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {placebo_model.params['fake_treat_post']:.4f}")
print(f"Standard error: {placebo_model.bse['fake_treat_post']:.4f}")
print(f"p-value: {placebo_model.pvalues['fake_treat_post']:.4f}")

# 6.2 Alternative outcome: employment (not just full-time)
print("\n--- 6.2 Alternative Outcome: Any Employment ---")
model_emp = smf.ols('employed ~ treatment + post + treat_post + female + married + C(EDUC) + C(YEAR)',
                    data=df_analysis).fit(cov_type='HC1')
print(f"DiD coefficient for employment: {model_emp.params['treat_post']:.4f}")
print(f"Standard error: {model_emp.bse['treat_post']:.4f}")
print(f"p-value: {model_emp.pvalues['treat_post']:.4f}")

# 6.3 Gender subgroup analysis
print("\n--- 6.3 Subgroup Analysis by Gender ---")
for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    model_sub = smf.ols('fulltime ~ treatment + post + treat_post + married + C(EDUC) + C(YEAR)',
                        data=df_sub).fit(cov_type='HC1')
    print(f"\n{sex_name}:")
    print(f"  DiD coefficient: {model_sub.params['treat_post']:.4f}")
    print(f"  Standard error: {model_sub.bse['treat_post']:.4f}")
    print(f"  p-value: {model_sub.pvalues['treat_post']:.4f}")
    print(f"  N: {int(model_sub.nobs)}")

# =============================================================================
# 7. EVENT STUDY / DYNAMIC EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("7. EVENT STUDY ANALYSIS")
print("=" * 80)

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

# Interactions with treatment (2011 as reference year)
df_analysis['treat_2006'] = df_analysis['treatment'] * df_analysis['year_2006']
df_analysis['treat_2007'] = df_analysis['treatment'] * df_analysis['year_2007']
df_analysis['treat_2008'] = df_analysis['treatment'] * df_analysis['year_2008']
df_analysis['treat_2009'] = df_analysis['treatment'] * df_analysis['year_2009']
df_analysis['treat_2010'] = df_analysis['treatment'] * df_analysis['year_2010']
df_analysis['treat_2013'] = df_analysis['treatment'] * df_analysis['year_2013']
df_analysis['treat_2014'] = df_analysis['treatment'] * df_analysis['year_2014']
df_analysis['treat_2015'] = df_analysis['treatment'] * df_analysis['year_2015']
df_analysis['treat_2016'] = df_analysis['treatment'] * df_analysis['year_2016']

event_model = smf.ols('''fulltime ~ treatment +
                         treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 +
                         treat_2013 + treat_2014 + treat_2015 + treat_2016 +
                         female + married + C(EDUC) + C(YEAR)''',
                      data=df_analysis).fit(cov_type='HC1')

print("\nEvent Study Coefficients (2011 is reference year):")
print("-" * 50)
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_vars:
    year = var.replace('treat_', '')
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci_low = event_model.conf_int().loc[var, 0]
    ci_high = event_model.conf_int().loc[var, 1]
    sig = '*' if pval < 0.1 else ''
    sig = '**' if pval < 0.05 else sig
    sig = '***' if pval < 0.01 else sig
    print(f"  {year}: {coef:7.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}] {sig}")

# =============================================================================
# 8. WEIGHTED ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("8. WEIGHTED ANALYSIS (Using PERWT)")
print("=" * 80)

# WLS with person weights
model_weighted = smf.wls('fulltime ~ treatment + post + treat_post + female + married + C(EDUC) + C(YEAR)',
                         data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"\nWeighted DiD coefficient: {model_weighted.params['treat_post']:.4f}")
print(f"Standard error: {model_weighted.bse['treat_post']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['treat_post', 0]:.4f}, {model_weighted.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model_weighted.pvalues['treat_post']:.4f}")

# =============================================================================
# 9. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("9. SUMMARY OF MAIN RESULTS")
print("=" * 80)

print("\n" + "-" * 80)
print("PREFERRED SPECIFICATION: Model 3 (DiD with Year FE and Demographics)")
print("-" * 80)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"  Coefficient: {model3.params['treat_post']:.4f}")
print(f"  Standard Error: {model3.bse['treat_post']:.4f}")
print(f"  95% Confidence Interval: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  t-statistic: {model3.tvalues['treat_post']:.3f}")
print(f"  p-value: {model3.pvalues['treat_post']:.4f}")
print(f"  Sample Size: {int(model3.nobs):,}")
print(f"  R-squared: {model3.rsquared:.4f}")

# Interpretation
effect_pct = model3.params['treat_post'] * 100
print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if model3.pvalues['treat_post'] < 0.05:
    print(f"  change in full-time employment (statistically significant at 5% level).")
else:
    print(f"  change in full-time employment (not statistically significant at 5% level).")

# =============================================================================
# 10. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 80)
print("10. SAVING RESULTS")
print("=" * 80)

# Save key results to CSV for LaTeX tables
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'DiD + Demographics', 'DiD + Year FE', 'DiD + Year + State FE', 'Weighted DiD'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                   model3.params['treat_post'], model4.params['treat_post'],
                   model_weighted.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'],
           model3.bse['treat_post'], model4.bse['treat_post'],
           model_weighted.bse['treat_post']],
    'p-value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model_weighted.pvalues['treat_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model_weighted.nobs)],
    'R2': [model1.rsquared, model2.rsquared, model3.rsquared,
           model4.rsquared, model_weighted.rsquared]
})
results_summary.to_csv('results_summary.csv', index=False)
print("Saved: results_summary.csv")

# Save sample counts
sample_summary = df_analysis.groupby(['treatment', 'post']).agg({
    'fulltime': ['count', 'mean', 'std']
}).round(4)
sample_summary.to_csv('sample_summary.csv')
print("Saved: sample_summary.csv")

# Save event study results
event_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [event_model.params.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'SE': [event_model.bse.get(f'treat_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
})
event_results.loc[event_results['Year'] == 2011, ['Coefficient', 'SE']] = [0, 0]  # Reference year
event_results.to_csv('event_study_results.csv', index=False)
print("Saved: event_study_results.csv")

# Save descriptive statistics
desc_stats = df_analysis.groupby(['treatment', 'post']).agg({
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'EDUC': 'mean',
    'fulltime': 'mean',
    'employed': 'mean',
    'PERWT': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Saved: descriptive_stats.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

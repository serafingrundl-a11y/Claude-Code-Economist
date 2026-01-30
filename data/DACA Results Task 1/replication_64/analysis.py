"""
DACA Employment Effects Replication Study
Analysis Script - Replication 64

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on
full-time employment (35+ hours/week)?

Author: Replication 64
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
print("DACA EMPLOYMENT EFFECTS REPLICATION STUDY")
print("=" * 80)

# =============================================================================
# STEP 1: LOAD AND FILTER DATA
# =============================================================================
print("\n[STEP 1] Loading and filtering data...")

# Load data
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")

# =============================================================================
# STEP 2: RESTRICT TO RELEVANT POPULATION
# =============================================================================
print("\n[STEP 2] Restricting to Hispanic-Mexican, Mexican-born population...")

# Hispanic-Mexican ethnicity (HISPAN = 1 for general, or HISPAND 100-107 for detailed Mexican)
# HISPAND codes: 100=Mexican, 102=Mexican American, 103=Mexicano/Mexicana,
# 104=Chicano/Chicana, 105=La Raza, 106=Mexican American Indian, 107=Mexico
df_mex = df[(df['HISPAN'] == 1) | ((df['HISPAND'] >= 100) & (df['HISPAND'] <= 107))]
print(f"After Hispanic-Mexican filter: {len(df_mex):,}")

# Born in Mexico (BPL = 200 or BPLD = 20000)
df_mex = df_mex[(df_mex['BPL'] == 200) | (df_mex['BPLD'] == 20000)]
print(f"After Mexico birthplace filter: {len(df_mex):,}")

# =============================================================================
# STEP 3: DEFINE ANALYSIS SAMPLE
# =============================================================================
print("\n[STEP 3] Defining analysis sample...")

# Non-citizens only (CITIZEN = 3)
# Per instructions: assume anyone who is not a citizen and has not received papers is undocumented
df_mex = df_mex[df_mex['CITIZEN'] == 3]
print(f"After non-citizen filter: {len(df_mex):,}")

# Exclude 2012 (transition year - cannot distinguish pre/post DACA within year)
df_mex = df_mex[df_mex['YEAR'] != 2012]
print(f"After excluding 2012: {len(df_mex):,}")

# Define pre and post periods
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)

# =============================================================================
# STEP 4: CONSTRUCT DACA ELIGIBILITY CRITERIA
# =============================================================================
print("\n[STEP 4] Constructing DACA eligibility criteria...")

# DACA eligibility requirements:
# 1. Arrived in US before 16th birthday
# 2. Born after June 15, 1981 (not yet 31 on June 15, 2012)
# 3. Lived continuously in US since June 15, 2007 (arrived by 2007)
# 4. Present in US on June 15, 2012 and no lawful status

# Calculate age at arrival
# YRSUSA1 gives years in USA (at time of survey), but we need year of arrival
# YRIMMIG gives year of immigration
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# For birth quarter adjustment: if born in Q1-Q2 (Jan-June), use birthyear
# if born Q3-Q4 (Jul-Dec), the person's birthday in the year is after June 15
# For June 15 cutoff for age 31: person born before June 16, 1981 would be 31+ on June 15, 2012
# To be conservative, someone born in 1981 Q1 or Q2 would be 31 by June 15, 2012
# Someone born in 1981 Q3 or Q4 would NOT yet be 31 by June 15, 2012

# Criterion 1: Arrived before 16th birthday
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)

# Criterion 2: Born after June 15, 1981 (not yet 31 on June 15, 2012)
# If BIRTHYR > 1981: definitely eligible
# If BIRTHYR = 1981 and BIRTHQTR >= 3: eligible (born July-Dec 1981)
# If BIRTHYR < 1981: not eligible
df_mex['born_after_cutoff'] = (
    (df_mex['BIRTHYR'] > 1981) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
).astype(int)

# Criterion 3: Arrived by 2007 (continuous presence since June 15, 2007)
df_mex['arrived_by_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)

# Criterion 4: Present in US on June 15, 2012 - we assume all in sample were present
# (ACS samples current US residents)

# Define DACA eligible
df_mex['daca_eligible'] = (
    (df_mex['arrived_before_16'] == 1) &
    (df_mex['born_after_cutoff'] == 1) &
    (df_mex['arrived_by_2007'] == 1)
).astype(int)

print(f"DACA eligible: {df_mex['daca_eligible'].sum():,}")
print(f"DACA ineligible: {(df_mex['daca_eligible'] == 0).sum():,}")

# =============================================================================
# STEP 5: CONSTRUCT OUTCOME VARIABLE
# =============================================================================
print("\n[STEP 5] Constructing outcome variable...")

# Full-time employment: UHRSWORK >= 35
# Note: UHRSWORK = 0 for those not working
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employed: {df_mex['fulltime'].sum():,}")
print(f"Not full-time: {(df_mex['fulltime'] == 0).sum():,}")

# =============================================================================
# STEP 6: RESTRICT TO WORKING-AGE POPULATION
# =============================================================================
print("\n[STEP 6] Restricting to working-age population...")

# Restrict to ages 18-64 (standard working age)
df_mex = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 64)]
print(f"After age 18-64 filter: {len(df_mex):,}")

# =============================================================================
# STEP 7: CREATE CONTROL VARIABLES
# =============================================================================
print("\n[STEP 7] Creating control variables...")

# Age and age squared
df_mex['age_sq'] = df_mex['AGE'] ** 2

# Female indicator
df_mex['female'] = (df_mex['SEX'] == 2).astype(int)

# Married indicator
df_mex['married'] = (df_mex['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC: 0=N/A, 1=None, 2=Nursery-4th grade, 3=5th-6th, 4=7th-8th, 5=9th,
# 6=10th, 7=11th, 8=12th no diploma, 9=HS diploma, 10=Some college, 11=Associate
df_mex['less_than_hs'] = (df_mex['EDUC'] <= 5).astype(int)
df_mex['hs_graduate'] = (df_mex['EDUC'].isin([6, 7, 8, 9])).astype(int)
df_mex['some_college'] = (df_mex['EDUC'].isin([10, 11])).astype(int)
df_mex['college_plus'] = (df_mex['EDUC'] >= 10).astype(int)

# Number of children
df_mex['has_children'] = (df_mex['NCHILD'] > 0).astype(int)

# Metropolitan status
df_mex['metro'] = (df_mex['METRO'].isin([2, 3, 4])).astype(int)

# Create state fixed effects
df_mex['state'] = df_mex['STATEFIP']

# Create year fixed effects
df_mex['year_fe'] = df_mex['YEAR']

# Interaction term for DiD
df_mex['daca_post'] = df_mex['daca_eligible'] * df_mex['post']

print("Control variables created.")

# =============================================================================
# STEP 8: SAMPLE STATISTICS
# =============================================================================
print("\n[STEP 8] Sample statistics...")

# Summary by treatment status and period
print("\n--- Sample Sizes ---")
print(df_mex.groupby(['daca_eligible', 'post']).size().unstack())

print("\n--- Full-time Employment Rates ---")
ft_rates = df_mex.groupby(['daca_eligible', 'post'])['fulltime'].mean().unstack()
print(ft_rates)

# Raw DiD
print("\n--- Raw Difference-in-Differences ---")
dd = (ft_rates.loc[1, 1] - ft_rates.loc[1, 0]) - (ft_rates.loc[0, 1] - ft_rates.loc[0, 0])
print(f"DiD estimate (raw): {dd:.4f}")

# Weighted means
print("\n--- Weighted Full-time Employment Rates ---")
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

ft_rates_wt = df_mex.groupby(['daca_eligible', 'post']).apply(weighted_mean).unstack()
print(ft_rates_wt)

dd_wt = (ft_rates_wt.loc[1, 1] - ft_rates_wt.loc[1, 0]) - (ft_rates_wt.loc[0, 1] - ft_rates_wt.loc[0, 0])
print(f"DiD estimate (weighted): {dd_wt:.4f}")

# =============================================================================
# STEP 9: MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 9] MAIN REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_post',
                  data=df_mex, weights=df_mex['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(results1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_post + AGE + age_sq + female + married + has_children',
                  data=df_mex, weights=df_mex['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(results2.summary())

# Model 3: DiD with controls and year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + has_children + C(year_fe)',
                  data=df_mex, weights=df_mex['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(results3.summary())

# Model 4: DiD with controls, year FE, and state FE (PREFERRED SPECIFICATION)
print("\n--- Model 4: DiD with Year and State Fixed Effects (PREFERRED) ---")
model4 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + has_children + C(year_fe) + C(state)',
                  data=df_mex, weights=df_mex['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(results4.summary())

# Model 5: With education controls
print("\n--- Model 5: Adding Education Controls ---")
model5 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + has_children + less_than_hs + some_college + college_plus + C(year_fe) + C(state)',
                  data=df_mex, weights=df_mex['PERWT'])
results5 = model5.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(results5.summary())

# =============================================================================
# STEP 10: ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 10] ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Different age restrictions (ages 18-40 - younger population more relevant to DACA)
print("\n--- Robustness 1: Ages 18-40 ---")
df_young = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 40)]
model_r1 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + female + married + has_children + C(year_fe) + C(state)',
                   data=df_young, weights=df_young['PERWT'])
results_r1 = model_r1.fit(cov_type='cluster', cov_kwds={'groups': df_young['state']})
print(f"N = {len(df_young):,}")
print(f"DiD coefficient: {results_r1.params['daca_post']:.4f} (SE: {results_r1.bse['daca_post']:.4f})")
print(f"p-value: {results_r1.pvalues['daca_post']:.4f}")

# Robustness 2: Males only
print("\n--- Robustness 2: Males Only ---")
df_male = df_mex[df_mex['female'] == 0]
model_r2 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + has_children + C(year_fe) + C(state)',
                   data=df_male, weights=df_male['PERWT'])
results_r2 = model_r2.fit(cov_type='cluster', cov_kwds={'groups': df_male['state']})
print(f"N = {len(df_male):,}")
print(f"DiD coefficient: {results_r2.params['daca_post']:.4f} (SE: {results_r2.bse['daca_post']:.4f})")
print(f"p-value: {results_r2.pvalues['daca_post']:.4f}")

# Robustness 3: Females only
print("\n--- Robustness 3: Females Only ---")
df_female = df_mex[df_mex['female'] == 1]
model_r3 = smf.wls('fulltime ~ daca_eligible + daca_post + AGE + age_sq + married + has_children + C(year_fe) + C(state)',
                   data=df_female, weights=df_female['PERWT'])
results_r3 = model_r3.fit(cov_type='cluster', cov_kwds={'groups': df_female['state']})
print(f"N = {len(df_female):,}")
print(f"DiD coefficient: {results_r3.params['daca_post']:.4f} (SE: {results_r3.bse['daca_post']:.4f})")
print(f"p-value: {results_r3.pvalues['daca_post']:.4f}")

# Robustness 4: Employment (any) instead of full-time
print("\n--- Robustness 4: Any Employment as Outcome ---")
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)
model_r4 = smf.wls('employed ~ daca_eligible + daca_post + AGE + age_sq + female + married + has_children + C(year_fe) + C(state)',
                   data=df_mex, weights=df_mex['PERWT'])
results_r4 = model_r4.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})
print(f"N = {len(df_mex):,}")
print(f"DiD coefficient: {results_r4.params['daca_post']:.4f} (SE: {results_r4.bse['daca_post']:.4f})")
print(f"p-value: {results_r4.pvalues['daca_post']:.4f}")

# =============================================================================
# STEP 11: EVENT STUDY / PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 11] EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year interactions with DACA eligibility
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_mex[f'year_{year}'] = (df_mex['YEAR'] == year).astype(int)
    df_mex[f'daca_year_{year}'] = df_mex['daca_eligible'] * df_mex[f'year_{year}']

# Event study regression (2011 as base year)
event_formula = 'fulltime ~ daca_eligible + AGE + age_sq + female + married + has_children + C(state)'
event_formula += ' + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016'
event_formula += ' + daca_year_2006 + daca_year_2007 + daca_year_2008 + daca_year_2009 + daca_year_2010'
event_formula += ' + daca_year_2013 + daca_year_2014 + daca_year_2015 + daca_year_2016'

model_event = smf.wls(event_formula, data=df_mex, weights=df_mex['PERWT'])
results_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': df_mex['state']})

print("\n--- Event Study Coefficients (DACA x Year interactions) ---")
print("(Base year: 2011)")
event_vars = ['daca_year_2006', 'daca_year_2007', 'daca_year_2008', 'daca_year_2009', 'daca_year_2010',
              'daca_year_2013', 'daca_year_2014', 'daca_year_2015', 'daca_year_2016']
for var in event_vars:
    print(f"{var}: {results_event.params[var]:.4f} (SE: {results_event.bse[var]:.4f}, p: {results_event.pvalues[var]:.4f})")

# =============================================================================
# STEP 12: SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 12] SUMMARY OF MAIN RESULTS")
print("=" * 80)

print("\n--- PREFERRED SPECIFICATION (Model 4) ---")
print(f"Sample size: {len(df_mex):,}")
print(f"DACA x Post coefficient: {results4.params['daca_post']:.4f}")
print(f"Standard error: {results4.bse['daca_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['daca_post', 0]:.4f}, {results4.conf_int().loc['daca_post', 1]:.4f}]")
print(f"t-statistic: {results4.tvalues['daca_post']:.4f}")
print(f"p-value: {results4.pvalues['daca_post']:.4f}")

# Pre-treatment mean for DACA-eligible
pre_mean = df_mex[(df_mex['daca_eligible'] == 1) & (df_mex['post'] == 0)]['fulltime'].mean()
print(f"\nPre-treatment full-time employment rate (DACA-eligible): {pre_mean:.4f}")
print(f"Effect as percentage of pre-treatment mean: {(results4.params['daca_post']/pre_mean)*100:.2f}%")

# =============================================================================
# STEP 13: EXPORT RESULTS FOR LATEX
# =============================================================================
print("\n" + "=" * 80)
print("[STEP 13] EXPORTING RESULTS")
print("=" * 80)

# Create results dataframe for export
results_summary = pd.DataFrame({
    'Model': ['Basic DiD', 'With Demographics', 'Year FE', 'Year + State FE (Preferred)', 'Adding Education'],
    'DiD Coefficient': [results1.params['daca_post'], results2.params['daca_post'],
                        results3.params['daca_post'], results4.params['daca_post'],
                        results5.params['daca_post']],
    'Std Error': [results1.bse['daca_post'], results2.bse['daca_post'],
                  results3.bse['daca_post'], results4.bse['daca_post'],
                  results5.bse['daca_post']],
    'p-value': [results1.pvalues['daca_post'], results2.pvalues['daca_post'],
                results3.pvalues['daca_post'], results4.pvalues['daca_post'],
                results5.pvalues['daca_post']],
    'N': [int(results1.nobs), int(results2.nobs), int(results3.nobs),
          int(results4.nobs), int(results5.nobs)]
})

print("\n--- Main Results Summary ---")
print(results_summary.to_string(index=False))

# Save to CSV
results_summary.to_csv('main_results.csv', index=False)

# Robustness results
robustness_summary = pd.DataFrame({
    'Specification': ['Ages 18-40', 'Males Only', 'Females Only', 'Any Employment'],
    'DiD Coefficient': [results_r1.params['daca_post'], results_r2.params['daca_post'],
                        results_r3.params['daca_post'], results_r4.params['daca_post']],
    'Std Error': [results_r1.bse['daca_post'], results_r2.bse['daca_post'],
                  results_r3.bse['daca_post'], results_r4.bse['daca_post']],
    'p-value': [results_r1.pvalues['daca_post'], results_r2.pvalues['daca_post'],
                results_r3.pvalues['daca_post'], results_r4.pvalues['daca_post']],
    'N': [int(results_r1.nobs), int(results_r2.nobs), int(results_r3.nobs), int(results_r4.nobs)]
})

print("\n--- Robustness Results Summary ---")
print(robustness_summary.to_string(index=False))

robustness_summary.to_csv('robustness_results.csv', index=False)

# Event study results
event_summary = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [results_event.params['daca_year_2006'], results_event.params['daca_year_2007'],
                    results_event.params['daca_year_2008'], results_event.params['daca_year_2009'],
                    results_event.params['daca_year_2010'], 0,
                    results_event.params['daca_year_2013'], results_event.params['daca_year_2014'],
                    results_event.params['daca_year_2015'], results_event.params['daca_year_2016']],
    'Std_Error': [results_event.bse['daca_year_2006'], results_event.bse['daca_year_2007'],
                  results_event.bse['daca_year_2008'], results_event.bse['daca_year_2009'],
                  results_event.bse['daca_year_2010'], 0,
                  results_event.bse['daca_year_2013'], results_event.bse['daca_year_2014'],
                  results_event.bse['daca_year_2015'], results_event.bse['daca_year_2016']]
})

print("\n--- Event Study Results ---")
print(event_summary.to_string(index=False))

event_summary.to_csv('event_study_results.csv', index=False)

# Summary statistics for the report
print("\n--- Descriptive Statistics ---")
desc_vars = ['fulltime', 'AGE', 'female', 'married', 'has_children', 'less_than_hs', 'some_college', 'college_plus']
desc_stats = df_mex[desc_vars].describe()
print(desc_stats)

# By treatment group
print("\n--- Means by Treatment Group ---")
means_by_group = df_mex.groupby('daca_eligible')[desc_vars].mean()
print(means_by_group)

means_by_group.to_csv('descriptive_stats.csv')

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

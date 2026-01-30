"""
DACA Replication Study - Analysis Script
Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?

Author: Replication 12
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD DATA IN CHUNKS AND FILTER
# =============================================================================
print("\n1. LOADING DATA (chunked processing)...")
print("-"*40)

# Load the ACS data in chunks, filtering as we go
data_path = "data/data.csv"

# Columns we need
keep_cols = ['YEAR', 'SAMPLE', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
             'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
             'YRSUSA1', 'YRSUSA2', 'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD',
             'LABFORCE', 'UHRSWORK', 'MARST', 'NCHILD', 'METRO', 'STATEFIP',
             'REGION', 'FAMSIZE']

# Process in chunks
chunk_size = 1000000
filtered_chunks = []

print("Processing chunks...")
chunk_num = 0
for chunk in pd.read_csv(data_path, chunksize=chunk_size, usecols=keep_cols):
    chunk_num += 1

    # Filter to Hispanic-Mexican ethnicity (HISPAN == 1 indicates Mexican)
    mask_hisp = (chunk['HISPAN'] == 1) | ((chunk['HISPAND'] >= 100) & (chunk['HISPAND'] <= 107))
    chunk = chunk[mask_hisp]

    # Filter to Mexican-born (BPL == 200 is Mexico)
    chunk = chunk[chunk['BPL'] == 200]

    # Filter to non-citizens (CITIZEN == 3 means "Not a citizen")
    chunk = chunk[chunk['CITIZEN'] == 3]

    # Filter to working age population (16-64)
    chunk = chunk[(chunk['AGE'] >= 16) & (chunk['AGE'] <= 64)]

    if len(chunk) > 0:
        filtered_chunks.append(chunk)

    if chunk_num % 5 == 0:
        print(f"  Processed chunk {chunk_num}...")

# Combine filtered chunks
df_sample = pd.concat(filtered_chunks, ignore_index=True)
del filtered_chunks

print(f"\nFinal analysis sample: {len(df_sample):,}")
print(f"Years covered: {df_sample['YEAR'].min()} - {df_sample['YEAR'].max()}")

# =============================================================================
# 2. CREATE DACA ELIGIBILITY VARIABLE
# =============================================================================
print("\n2. CREATING DACA ELIGIBILITY VARIABLE...")
print("-"*40)

"""
DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Had not yet turned 31 by June 15, 2012 (born on or after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (arrived by 2007 or earlier)
4. Present in US on June 15, 2012 (cannot verify directly, assume present)
5. Not a citizen (already filtered)

Note: We must compute eligibility carefully. Someone may become age-ineligible
over time, but we use the 2012 cutoff for consistency.
"""

# Age at arrival using YRSUSA1 (years in USA)
# If YRSUSA1 is 0 or missing, cannot determine
df_sample['yrs_in_usa'] = df_sample['YRSUSA1'].replace(0, np.nan)
df_sample['age_at_arrival'] = df_sample['AGE'] - df_sample['yrs_in_usa']

# For robustness, also compute using YRIMMIG
# Age at immigration = survey_year - YRIMMIG - (survey_year - AGE - BIRTHYR)
# Simplified: arrived_year = YRIMMIG, so age at arrival = BIRTHYR - YRIMMIG approximately
# But better: age at arrival = AGE_current - (YEAR_survey - YRIMMIG)

# Condition 1: Arrived before 16th birthday
cond1 = df_sample['age_at_arrival'] < 16

# Condition 2: Born after June 15, 1981 (under 31 on June 15, 2012)
# Need to be born in 1981 or later to be eligible
# To be precise: born on or after June 15, 1981
# BIRTHYR >= 1982 definitely eligible, BIRTHYR == 1981 with BIRTHQTR >= 2 (Apr-Jun or later)
cond2_strict = (df_sample['BIRTHYR'] >= 1982) | \
               ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 2))

# Condition 3: Present in US since June 15, 2007 (YRIMMIG <= 2007)
cond3 = df_sample['YRIMMIG'] <= 2007

# Handle missing values
valid_data = df_sample['yrs_in_usa'].notna() & (df_sample['YRIMMIG'] > 0)

# Create DACA eligible indicator
df_sample['daca_eligible'] = (cond1 & cond2_strict & cond3 & valid_data).astype(int)

print(f"DACA eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")
print(f"Not DACA eligible: {(1-df_sample['daca_eligible']).sum():,} ({(1-df_sample['daca_eligible']).mean()*100:.1f}%)")

# =============================================================================
# 3. CREATE OUTCOME VARIABLE
# =============================================================================
print("\n3. CREATING OUTCOME VARIABLE...")
print("-"*40)

"""
Full-time employment: Usually working 35+ hours per week
UHRSWORK = usual hours worked per week
"""

# Full-time employment (35+ hours per week)
df_sample['fulltime_employed'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed variable (EMPSTAT == 1)
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"Full-time employed: {df_sample['fulltime_employed'].sum():,} ({df_sample['fulltime_employed'].mean()*100:.1f}%)")
print(f"Employed (any): {df_sample['employed'].sum():,} ({df_sample['employed'].mean()*100:.1f}%)")

# =============================================================================
# 4. CREATE TIME PERIOD VARIABLES
# =============================================================================
print("\n4. CREATING TIME PERIOD VARIABLES...")
print("-"*40)

"""
DACA was implemented on June 15, 2012.
Pre-period: 2006-2011
Post-period: 2013-2016 (focus years per instructions)
Transition year: 2012 (exclude or treat separately)
"""

# Post-DACA indicator
df_sample['post_daca'] = (df_sample['YEAR'] >= 2013).astype(int)

# Treatment effect interaction
df_sample['daca_x_post'] = df_sample['daca_eligible'] * df_sample['post_daca']

print(f"Pre-DACA observations (2006-2012): {(df_sample['post_daca']==0).sum():,}")
print(f"Post-DACA observations (2013-2016): {(df_sample['post_daca']==1).sum():,}")

# =============================================================================
# 5. CREATE CONTROL VARIABLES
# =============================================================================
print("\n5. CREATING CONTROL VARIABLES...")
print("-"*40)

# Gender (1=Male, 2=Female)
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Age squared
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Marital status (1=Married, spouse present)
df_sample['married'] = (df_sample['MARST'] == 1).astype(int)

# Education level
# EDUC: 0-N/A, 1-No school, 2-Nursery to grade 4, 3-Grade 5-8, 4-Grade 9,
#       5-Grade 10, 6-Grade 11, 7-Grade 12/HS diploma, 8-1 yr college,
#       9-2 yr college, 10-3 yr college, 11-4+ years college
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # Some college or more

# Metropolitan area
df_sample['metro'] = (df_sample['METRO'] >= 2).astype(int)

# Number of children
df_sample['has_children'] = (df_sample['NCHILD'] > 0).astype(int)

# State fixed effects
df_sample['state'] = df_sample['STATEFIP']

print("Control variables created:")
print(f"  Female: {df_sample['female'].mean()*100:.1f}%")
print(f"  Married: {df_sample['married'].mean()*100:.1f}%")
print(f"  HS or more: {df_sample['educ_hs'].mean()*100:.1f}%")
print(f"  College: {df_sample['educ_college'].mean()*100:.1f}%")
print(f"  Metro area: {df_sample['metro'].mean()*100:.1f}%")
print(f"  Has children: {df_sample['has_children'].mean()*100:.1f}%")

# =============================================================================
# 6. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n6. DESCRIPTIVE STATISTICS...")
print("-"*40)

# Summary statistics
desc_vars = ['AGE', 'female', 'married', 'educ_hs', 'educ_college',
             'employed', 'fulltime_employed', 'UHRSWORK', 'daca_eligible']

# By treatment group
print("\nDescriptive Statistics by DACA Eligibility:")
desc_by_group = df_sample.groupby('daca_eligible')[desc_vars].mean()
print(desc_by_group.round(3).T)

# By time period
print("\nFull-time Employment by Group and Time Period:")
desc_by_time = df_sample.groupby(['daca_eligible', 'post_daca'])['fulltime_employed'].agg(['mean', 'std', 'count'])
print(desc_by_time.round(3))

# Raw DiD calculation
print("\nRaw Difference-in-Differences:")
means = df_sample.groupby(['daca_eligible', 'post_daca'])['fulltime_employed'].mean()
did_raw = (means.loc[(1, 1)] - means.loc[(1, 0)]) - (means.loc[(0, 1)] - means.loc[(0, 0)])
print(f"  Eligible Pre: {means.loc[(1, 0)]:.4f}")
print(f"  Eligible Post: {means.loc[(1, 1)]:.4f}")
print(f"  Not Eligible Pre: {means.loc[(0, 0)]:.4f}")
print(f"  Not Eligible Post: {means.loc[(0, 1)]:.4f}")
print(f"  Raw DiD: {did_raw:.4f}")

# =============================================================================
# 7. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n7. DIFFERENCE-IN-DIFFERENCES ANALYSIS...")
print("-"*40)

# Exclude 2012 (transition year) for main analysis
df_did = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"Observations excluding 2012: {len(df_did):,}")

# Model 1: Basic DiD without controls
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime_employed ~ daca_eligible + post_daca + daca_x_post',
                 data=df_did).fit(cov_type='cluster', cov_kwds={'groups': df_did['state']})
print(f"  daca_x_post: {model1.params['daca_x_post']:.4f} (SE: {model1.bse['daca_x_post']:.4f})")
print(f"  R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.ols('fulltime_employed ~ daca_eligible + post_daca + daca_x_post + '
                 'AGE + age_sq + female + married + educ_hs + educ_college + '
                 'has_children + metro',
                 data=df_did).fit(cov_type='cluster', cov_kwds={'groups': df_did['state']})
print(f"  daca_x_post: {model2.params['daca_x_post']:.4f} (SE: {model2.bse['daca_x_post']:.4f})")
print(f"  R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with state and year fixed effects
print("\nModel 3: DiD with state and year fixed effects")
model3 = smf.ols('fulltime_employed ~ daca_eligible + daca_x_post + '
                 'AGE + age_sq + female + married + educ_hs + educ_college + '
                 'has_children + metro + C(state) + C(YEAR)',
                 data=df_did).fit(cov_type='cluster', cov_kwds={'groups': df_did['state']})
print(f"  daca_x_post: {model3.params['daca_x_post']:.4f} (SE: {model3.bse['daca_x_post']:.4f})")
print(f"  R-squared: {model3.rsquared:.4f}")

# Extract key coefficients summary
print("\n" + "-"*40)
print("KEY RESULTS - DiD Coefficient (daca_x_post):")
print(f"  Model 1 (basic): {model1.params['daca_x_post']:.4f} (SE: {model1.bse['daca_x_post']:.4f}), p={model1.pvalues['daca_x_post']:.4f}")
print(f"  Model 2 (controls): {model2.params['daca_x_post']:.4f} (SE: {model2.bse['daca_x_post']:.4f}), p={model2.pvalues['daca_x_post']:.4f}")
print(f"  Model 3 (FE): {model3.params['daca_x_post']:.4f} (SE: {model3.bse['daca_x_post']:.4f}), p={model3.pvalues['daca_x_post']:.4f}")

# =============================================================================
# 8. EVENT STUDY ANALYSIS
# =============================================================================
print("\n8. EVENT STUDY ANALYSIS...")
print("-"*40)

# Create year interactions with DACA eligibility (reference year: 2011)
df_event = df_sample[df_sample['YEAR'] != 2012].copy()

# Create interaction terms for event study
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_event[f'daca_x_{year}'] = df_event['daca_eligible'] * (df_event['YEAR'] == year).astype(int)

# Event study regression
event_terms = ' + '.join([f'daca_x_{y}' for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
event_formula = f'fulltime_employed ~ daca_eligible + {event_terms} + ' + \
                'AGE + age_sq + female + married + educ_hs + educ_college + has_children + metro + C(state) + C(YEAR)'

model_event = smf.ols(event_formula, data=df_event).fit(cov_type='cluster',
                                                         cov_kwds={'groups': df_event['state']})

print("\nEvent Study Coefficients (reference: 2011):")
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
for year in event_years:
    coef = model_event.params[f'daca_x_{year}']
    se = model_event.bse[f'daca_x_{year}']
    pval = model_event.pvalues[f'daca_x_{year}']
    event_coefs.append({'year': year, 'coef': coef, 'se': se, 'pval': pval})
    stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}){stars}")

event_df = pd.DataFrame(event_coefs)

# =============================================================================
# 9. ROBUSTNESS CHECKS
# =============================================================================
print("\n9. ROBUSTNESS CHECKS...")
print("-"*40)

# Robustness 1: Different age restrictions (18-40)
print("\nRobustness 1: Age restriction 18-40")
df_robust1 = df_did[(df_did['AGE'] >= 18) & (df_did['AGE'] <= 40)]
model_r1 = smf.ols('fulltime_employed ~ daca_eligible + post_daca + daca_x_post + '
                   'AGE + age_sq + female + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_robust1).fit(cov_type='cluster', cov_kwds={'groups': df_robust1['state']})
print(f"  DiD coefficient: {model_r1.params['daca_x_post']:.4f} (SE: {model_r1.bse['daca_x_post']:.4f}), N = {len(df_robust1):,}")

# Robustness 2: Male only
print("\nRobustness 2: Male only")
df_robust2 = df_did[df_did['female'] == 0]
model_r2 = smf.ols('fulltime_employed ~ daca_eligible + post_daca + daca_x_post + '
                   'AGE + age_sq + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_robust2).fit(cov_type='cluster', cov_kwds={'groups': df_robust2['state']})
print(f"  DiD coefficient: {model_r2.params['daca_x_post']:.4f} (SE: {model_r2.bse['daca_x_post']:.4f}), N = {len(df_robust2):,}")

# Robustness 3: Female only
print("\nRobustness 3: Female only")
df_robust3 = df_did[df_did['female'] == 1]
model_r3 = smf.ols('fulltime_employed ~ daca_eligible + post_daca + daca_x_post + '
                   'AGE + age_sq + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_robust3).fit(cov_type='cluster', cov_kwds={'groups': df_robust3['state']})
print(f"  DiD coefficient: {model_r3.params['daca_x_post']:.4f} (SE: {model_r3.bse['daca_x_post']:.4f}), N = {len(df_robust3):,}")

# Robustness 4: Employment (any) as outcome
print("\nRobustness 4: Employment (any) as outcome")
model_r4 = smf.ols('employed ~ daca_eligible + post_daca + daca_x_post + '
                   'AGE + age_sq + female + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_did).fit(cov_type='cluster', cov_kwds={'groups': df_did['state']})
print(f"  DiD coefficient: {model_r4.params['daca_x_post']:.4f} (SE: {model_r4.bse['daca_x_post']:.4f})")

# Robustness 5: Include 2012 in post-period
print("\nRobustness 5: Include 2012 as post-period")
df_robust5 = df_sample.copy()
df_robust5['post_daca_incl2012'] = (df_robust5['YEAR'] >= 2012).astype(int)
df_robust5['daca_x_post_incl2012'] = df_robust5['daca_eligible'] * df_robust5['post_daca_incl2012']
model_r5 = smf.ols('fulltime_employed ~ daca_eligible + post_daca_incl2012 + daca_x_post_incl2012 + '
                   'AGE + age_sq + female + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_robust5).fit(cov_type='cluster', cov_kwds={'groups': df_robust5['state']})
print(f"  DiD coefficient: {model_r5.params['daca_x_post_incl2012']:.4f} (SE: {model_r5.bse['daca_x_post_incl2012']:.4f})")

# Robustness 6: Weighted regression using person weights
print("\nRobustness 6: Using survey weights (PERWT)")
model_r6 = smf.wls('fulltime_employed ~ daca_eligible + post_daca + daca_x_post + '
                   'AGE + age_sq + female + married + educ_hs + educ_college + '
                   'has_children + metro + C(state) + C(YEAR)',
                   data=df_did, weights=df_did['PERWT']).fit(cov_type='cluster',
                                                              cov_kwds={'groups': df_did['state']})
print(f"  DiD coefficient: {model_r6.params['daca_x_post']:.4f} (SE: {model_r6.bse['daca_x_post']:.4f})")

# =============================================================================
# 10. PLACEBO TEST
# =============================================================================
print("\n10. PLACEBO TEST...")
print("-"*40)

# Placebo test: Use fake treatment date of 2009 (pre-DACA period only)
df_placebo = df_sample[(df_sample['YEAR'] >= 2006) & (df_sample['YEAR'] <= 2011)].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['daca_x_placebo'] = df_placebo['daca_eligible'] * df_placebo['post_placebo']

model_placebo = smf.ols('fulltime_employed ~ daca_eligible + post_placebo + daca_x_placebo + '
                        'AGE + age_sq + female + married + educ_hs + educ_college + '
                        'has_children + metro + C(state) + C(YEAR)',
                        data=df_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['state']})
print(f"Placebo DiD coefficient (fake 2009 treatment): {model_placebo.params['daca_x_placebo']:.4f}")
print(f"SE: {model_placebo.bse['daca_x_placebo']:.4f}")
print(f"p-value: {model_placebo.pvalues['daca_x_placebo']:.4f}")

# =============================================================================
# 11. SAVE RESULTS
# =============================================================================
print("\n11. SAVING RESULTS...")
print("-"*40)

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'DiD + Controls', 'DiD + FE (Main)',
              'Robust: Age 18-40', 'Robust: Male', 'Robust: Female',
              'Robust: Any Employment', 'Robust: Incl 2012', 'Robust: Weighted',
              'Placebo (2009)'],
    'Coefficient': [
        model1.params['daca_x_post'],
        model2.params['daca_x_post'],
        model3.params['daca_x_post'],
        model_r1.params['daca_x_post'],
        model_r2.params['daca_x_post'],
        model_r3.params['daca_x_post'],
        model_r4.params['daca_x_post'],
        model_r5.params['daca_x_post_incl2012'],
        model_r6.params['daca_x_post'],
        model_placebo.params['daca_x_placebo']
    ],
    'Std_Error': [
        model1.bse['daca_x_post'],
        model2.bse['daca_x_post'],
        model3.bse['daca_x_post'],
        model_r1.bse['daca_x_post'],
        model_r2.bse['daca_x_post'],
        model_r3.bse['daca_x_post'],
        model_r4.bse['daca_x_post'],
        model_r5.bse['daca_x_post_incl2012'],
        model_r6.bse['daca_x_post'],
        model_placebo.bse['daca_x_placebo']
    ],
    'P_Value': [
        model1.pvalues['daca_x_post'],
        model2.pvalues['daca_x_post'],
        model3.pvalues['daca_x_post'],
        model_r1.pvalues['daca_x_post'],
        model_r2.pvalues['daca_x_post'],
        model_r3.pvalues['daca_x_post'],
        model_r4.pvalues['daca_x_post'],
        model_r5.pvalues['daca_x_post_incl2012'],
        model_r6.pvalues['daca_x_post'],
        model_placebo.pvalues['daca_x_placebo']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model_r1.nobs),
        int(model_r2.nobs),
        int(model_r3.nobs),
        int(model_r4.nobs),
        int(model_r5.nobs),
        int(model_r6.nobs),
        int(model_placebo.nobs)
    ]
}

results_df = pd.DataFrame(results_summary)
results_df['95% CI Lower'] = results_df['Coefficient'] - 1.96 * results_df['Std_Error']
results_df['95% CI Upper'] = results_df['Coefficient'] + 1.96 * results_df['Std_Error']
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save descriptive statistics
desc_stats = df_sample.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime_employed': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'educ_college': 'mean',
    'PERWT': 'sum'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

# =============================================================================
# 12. CREATE VISUALIZATIONS
# =============================================================================
print("\n12. CREATING VISUALIZATIONS...")
print("-"*40)

# Figure 1: Trends in full-time employment by group
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Trends by year and group
ax1 = axes[0, 0]
trends = df_sample.groupby(['YEAR', 'daca_eligible'])['fulltime_employed'].mean().unstack()
trends.columns = ['Not DACA Eligible', 'DACA Eligible']
trends.plot(ax=ax1, marker='o', linewidth=2)
ax1.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax1.set_xlabel('Year')
ax1.set_ylabel('Full-Time Employment Rate')
ax1.set_title('A. Full-Time Employment Trends by DACA Eligibility')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Panel B: Event study plot
ax2 = axes[0, 1]
years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_plot = []
ses_plot = []
for y in years_plot:
    if y == 2011:
        coefs_plot.append(0)
        ses_plot.append(0)
    else:
        row = event_df[event_df['year']==y]
        if len(row) > 0:
            coefs_plot.append(row['coef'].values[0])
            ses_plot.append(row['se'].values[0])
        else:
            coefs_plot.append(0)
            ses_plot.append(0)

ax2.errorbar(years_plot, coefs_plot, yerr=[1.96*s for s in ses_plot],
             fmt='o-', capsize=3, capthick=1, linewidth=2, markersize=6, color='blue')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax2.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax2.set_xlabel('Year')
ax2.set_ylabel('Coefficient (relative to 2011)')
ax2.set_title('B. Event Study: DACA Eligible × Year Interactions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Panel C: Distribution of usual hours worked
ax3 = axes[1, 0]
eligible_hrs = df_sample[df_sample['daca_eligible']==1]['UHRSWORK']
not_eligible_hrs = df_sample[df_sample['daca_eligible']==0]['UHRSWORK']
ax3.hist(not_eligible_hrs[not_eligible_hrs > 0], bins=50, alpha=0.5,
         label='Not Eligible', density=True, color='orange')
ax3.hist(eligible_hrs[eligible_hrs > 0], bins=50, alpha=0.5,
         label='DACA Eligible', density=True, color='blue')
ax3.axvline(x=35, color='red', linestyle='--', label='Full-time threshold (35 hrs)')
ax3.set_xlabel('Usual Hours Worked per Week')
ax3.set_ylabel('Density')
ax3.set_title('C. Distribution of Hours Worked by DACA Eligibility')
ax3.legend()

# Panel D: Sample sizes by year
ax4 = axes[1, 1]
sample_sizes = df_sample.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_sizes.columns = ['Not DACA Eligible', 'DACA Eligible']
sample_sizes.plot(kind='bar', ax=ax4, width=0.8)
ax4.set_xlabel('Year')
ax4.set_ylabel('Sample Size')
ax4.set_title('D. Sample Size by Year and DACA Eligibility')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figure1_main_results.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_main_results.pdf', bbox_inches='tight')
print("Figure 1 saved")

# Figure 2: Robustness visualization
fig2, ax = plt.subplots(figsize=(12, 6))
models_names = ['Main\n(Full Sample)', 'Age 18-40', 'Male Only', 'Female Only',
                'Any\nEmployment', 'Incl. 2012', 'Weighted']
coefs = [model3.params['daca_x_post'], model_r1.params['daca_x_post'],
         model_r2.params['daca_x_post'], model_r3.params['daca_x_post'],
         model_r4.params['daca_x_post'], model_r5.params['daca_x_post_incl2012'],
         model_r6.params['daca_x_post']]
errors = [model3.bse['daca_x_post'], model_r1.bse['daca_x_post'],
          model_r2.bse['daca_x_post'], model_r3.bse['daca_x_post'],
          model_r4.bse['daca_x_post'], model_r5.bse['daca_x_post_incl2012'],
          model_r6.bse['daca_x_post']]

x_pos = range(len(models_names))
ax.errorbar(x_pos, coefs, yerr=[1.96*e for e in errors], fmt='o', capsize=5,
            capthick=2, markersize=10, color='blue')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models_names)
ax.set_ylabel('DiD Coefficient (with 95% CI)')
ax.set_title('Robustness of DACA Effect on Full-Time Employment')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure2_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_robustness.pdf', bbox_inches='tight')
print("Figure 2 saved")

# Figure 3: Pre-trends visualization
fig3, ax = plt.subplots(figsize=(10, 6))
pre_years = [2006, 2007, 2008, 2009, 2010]
pre_coefs = [event_df[event_df['year']==y]['coef'].values[0] for y in pre_years]
pre_ses = [event_df[event_df['year']==y]['se'].values[0] for y in pre_years]

ax.errorbar(pre_years, pre_coefs, yerr=[1.96*s for s in pre_ses],
            fmt='o-', capsize=4, capthick=1.5, linewidth=2, markersize=8, color='green')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (relative to 2011)')
ax.set_title('Pre-Trends Test: DACA Eligible × Year (Pre-2012 Only)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_pretrends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_pretrends.pdf', bbox_inches='tight')
print("Figure 3 saved")

plt.close('all')

# =============================================================================
# 13. FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

# Calculate pre-trend test
pre_trend_coefs = [event_df[event_df['year']==y]['coef'].values[0] for y in [2006, 2007, 2008, 2009, 2010]]
pre_trend_significant = sum(1 for y in [2006, 2007, 2008, 2009, 2010]
                            if event_df[event_df['year']==y]['pval'].values[0] < 0.05)

print(f"""
PREFERRED ESTIMATE (Model 3 with State and Year FE):
  Effect of DACA on Full-Time Employment: {model3.params['daca_x_post']:.4f}
  Standard Error (clustered by state): {model3.bse['daca_x_post']:.4f}
  95% Confidence Interval: [{model3.params['daca_x_post'] - 1.96*model3.bse['daca_x_post']:.4f}, {model3.params['daca_x_post'] + 1.96*model3.bse['daca_x_post']:.4f}]
  t-statistic: {model3.tvalues['daca_x_post']:.3f}
  p-value: {model3.pvalues['daca_x_post']:.4f}
  Sample Size: {int(model3.nobs):,}
  R-squared: {model3.rsquared:.4f}

INTERPRETATION:
  DACA eligibility is associated with a {model3.params['daca_x_post']*100:.2f} percentage point
  {'increase' if model3.params['daca_x_post'] > 0 else 'decrease'} in the probability of full-time employment
  among Mexican-born non-citizen Hispanics following the program's implementation.
  This effect is {'statistically significant' if model3.pvalues['daca_x_post'] < 0.05 else 'not statistically significant'} at the 5% level.

PRE-TRENDS (Event Study):
  Number of pre-2012 coefficients significant at 5%: {pre_trend_significant} out of 5
  {'Pre-trends appear parallel (supports identification)' if pre_trend_significant <= 1 else 'Some evidence of pre-trends (caution warranted)'}

PLACEBO TEST (Fake 2009 Treatment):
  Placebo coefficient: {model_placebo.params['daca_x_placebo']:.4f}
  p-value: {model_placebo.pvalues['daca_x_placebo']:.4f}
  {'No significant placebo effect (expected under valid design)' if model_placebo.pvalues['daca_x_placebo'] > 0.05 else 'Significant placebo effect (potential concern)'}

SAMPLE COMPOSITION:
  Total observations: {len(df_sample):,}
  DACA eligible: {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)
  Pre-period (2006-2011): {(df_sample['YEAR'] <= 2011).sum():,}
  Post-period (2013-2016): {(df_sample['YEAR'] >= 2013).sum():,}
""")

# Print full results table
print("\nFULL RESULTS TABLE:")
print(results_df.to_string(index=False))

# Print detailed model 3 results
print("\n" + "-"*40)
print("DETAILED MODEL 3 RESULTS (Selected Coefficients):")
print("-"*40)
key_vars = ['daca_eligible', 'daca_x_post', 'AGE', 'age_sq', 'female',
            'married', 'educ_hs', 'educ_college', 'has_children', 'metro']
for var in key_vars:
    if var in model3.params:
        coef = model3.params[var]
        se = model3.bse[var]
        pval = model3.pvalues[var]
        stars = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {var:20s}: {coef:8.4f} ({se:.4f}){stars}")

print("\n" + "="*80)
print("Analysis complete. Files saved:")
print("  - results_summary.csv")
print("  - event_study_results.csv")
print("  - descriptive_stats.csv")
print("  - figure1_main_results.png/pdf")
print("  - figure2_robustness.png/pdf")
print("  - figure3_pretrends.png/pdf")
print("="*80)

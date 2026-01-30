"""
DACA Replication Study - Analysis Script
Research Question: Impact of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States.

Author: [Anonymous]
Date: 2026-01-25
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
print("DACA REPLICATION STUDY - FULL ANALYSIS")
print("="*80)

#------------------------------------------------------------------------------
# STEP 1: Load Data
#------------------------------------------------------------------------------
print("\n[1] Loading data...")
data_path = "data/data.csv"

# Read the data
df = pd.read_csv(data_path)
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

#------------------------------------------------------------------------------
# STEP 2: Apply Sample Restrictions
#------------------------------------------------------------------------------
print("\n[2] Applying sample restrictions...")

# Filter 1: Hispanic-Mexican ethnicity
# HISPAN = 1 (Mexican) or HISPAND in 100-107 (Mexican detailed codes)
df_sample = df[(df['HISPAN'] == 1) | ((df['HISPAND'] >= 100) & (df['HISPAND'] <= 107))].copy()
print(f"   After Hispanic-Mexican filter: {len(df_sample):,}")

# Filter 2: Born in Mexico
# BPL = 200 (Mexico) or BPLD = 20000 (Mexico detailed)
df_sample = df_sample[(df_sample['BPL'] == 200) | (df_sample['BPLD'] == 20000)].copy()
print(f"   After born in Mexico filter: {len(df_sample):,}")

# Filter 3: Non-citizen (CITIZEN = 3)
# Per instructions: assume anyone who is not a citizen and has not received
# immigration papers is undocumented for DACA purposes
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After non-citizen filter: {len(df_sample):,}")

# Filter 4: Has valid immigration year (YRIMMIG > 0)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"   After valid immigration year filter: {len(df_sample):,}")

# Filter 5: Exclude 2012 due to timing ambiguity (DACA implemented mid-year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_sample):,}")

# Filter 6: Working age population (16-64)
df_sample = df_sample[(df_sample['AGE'] >= 16) & (df_sample['AGE'] <= 64)].copy()
print(f"   After working age filter (16-64): {len(df_sample):,}")

#------------------------------------------------------------------------------
# STEP 3: Define DACA Eligibility
#------------------------------------------------------------------------------
print("\n[3] Defining DACA eligibility...")

# DACA Eligibility Criteria (as of June 15, 2012):
# 1. Arrived before 16th birthday
# 2. Not yet 31 as of June 15, 2012 (born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
# 4. Non-citizen (already filtered)

# Calculate age at immigration
# Age at immigration = Immigration year - Birth year
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Criterion 1: Arrived before 16th birthday
df_sample['arrived_before_16'] = df_sample['age_at_immig'] < 16

# Criterion 2: Born after June 15, 1981 (not yet 31 as of June 15, 2012)
# Conservative: use birth year >= 1982 (anyone born in 1981 with Q1-Q2 would qualify)
# More precise approach:
#   - Born 1982 or later: definitely qualifies
#   - Born 1981: qualifies if born after June 15 (Q3 or Q4)
df_sample['age_criterion'] = (df_sample['BIRTHYR'] >= 1982) | \
                              ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))

# Criterion 3: In US since June 15, 2007 (immigrated by 2007 or earlier)
df_sample['continuous_presence'] = df_sample['YRIMMIG'] <= 2007

# Combined DACA eligibility
df_sample['daca_eligible'] = (df_sample['arrived_before_16'] &
                               df_sample['age_criterion'] &
                               df_sample['continuous_presence'])

print(f"   DACA eligible observations: {df_sample['daca_eligible'].sum():,}")
print(f"   Non-eligible observations: {(~df_sample['daca_eligible']).sum():,}")

#------------------------------------------------------------------------------
# STEP 4: Define Treatment and Control Groups
#------------------------------------------------------------------------------
print("\n[4] Defining treatment and control groups...")

# Treatment group: DACA eligible
# Control group: Similar individuals who are NOT eligible
# For a clean comparison, use individuals who:
# - Arrived before age 16 (childhood arrivals)
# - Were too old to qualify (born before June 15, 1981)
# OR
# - Arrived before age 16 but after 2007 (didn't meet continuous presence)

# More restrictive control: focus on age-based control
# Control = arrived before 16, but born before 1981 (too old for DACA)
df_sample['control_too_old'] = (df_sample['arrived_before_16'] &
                                 ~df_sample['age_criterion'] &
                                 (df_sample['YRIMMIG'] <= 2007))

# Alternative control: arrived before 16 but after 2007
df_sample['control_late_arrival'] = (df_sample['arrived_before_16'] &
                                      df_sample['age_criterion'] &
                                      ~df_sample['continuous_presence'])

# Primary analysis: use age-based control (too old)
df_sample['treatment'] = df_sample['daca_eligible'].astype(int)
df_sample['in_analysis'] = df_sample['daca_eligible'] | df_sample['control_too_old']

print(f"   Treatment group (DACA eligible): {df_sample['daca_eligible'].sum():,}")
print(f"   Control group (too old): {df_sample['control_too_old'].sum():,}")
print(f"   Total in primary analysis: {df_sample['in_analysis'].sum():,}")

#------------------------------------------------------------------------------
# STEP 5: Define Outcome Variable
#------------------------------------------------------------------------------
print("\n[5] Defining outcome variable...")

# Full-time employment: UHRSWORK >= 35
# UHRSWORK = 0 means N/A (not in labor force, unemployed, etc.)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

print(f"   Full-time workers: {df_sample['fulltime'].sum():,}")
print(f"   Part-time/not working: {(~df_sample['fulltime'].astype(bool)).sum():,}")

#------------------------------------------------------------------------------
# STEP 6: Create Period Variables
#------------------------------------------------------------------------------
print("\n[6] Creating period variables...")

# Post-DACA period: 2013-2016
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Pre-period: 2006-2011
# Post-period: 2013-2016

print(f"   Pre-period (2006-2011) observations: {(df_sample['post'] == 0).sum():,}")
print(f"   Post-period (2013-2016) observations: {(df_sample['post'] == 1).sum():,}")

#------------------------------------------------------------------------------
# STEP 7: Restrict to Analysis Sample
#------------------------------------------------------------------------------
print("\n[7] Creating analysis sample...")

df_analysis = df_sample[df_sample['in_analysis']].copy()
print(f"   Analysis sample size: {len(df_analysis):,}")

# Interaction term for DiD
df_analysis['treat_post'] = df_analysis['treatment'] * df_analysis['post']

#------------------------------------------------------------------------------
# STEP 8: Summary Statistics
#------------------------------------------------------------------------------
print("\n[8] Summary Statistics")
print("-"*60)

# By treatment status and period
summary_stats = df_analysis.groupby(['treatment', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'UHRSWORK': ['mean', 'std'],
    'PERWT': 'sum'
}).round(3)

print("\nSummary by Treatment x Period:")
print(summary_stats)

# Weighted means
print("\n" + "="*60)
print("WEIGHTED SUMMARY STATISTICS")
print("="*60)

for treat in [0, 1]:
    for post in [0, 1]:
        subset = df_analysis[(df_analysis['treatment'] == treat) & (df_analysis['post'] == post)]
        if len(subset) > 0:
            treat_label = "Treatment" if treat == 1 else "Control"
            post_label = "Post" if post == 1 else "Pre"
            weighted_ft = np.average(subset['fulltime'], weights=subset['PERWT'])
            weighted_age = np.average(subset['AGE'], weights=subset['PERWT'])
            weighted_hrs = np.average(subset['UHRSWORK'], weights=subset['PERWT'])
            print(f"\n{treat_label} - {post_label}:")
            print(f"   N (unweighted): {len(subset):,}")
            print(f"   N (weighted): {subset['PERWT'].sum():,.0f}")
            print(f"   Full-time rate (weighted): {weighted_ft:.4f}")
            print(f"   Mean age (weighted): {weighted_age:.1f}")
            print(f"   Mean hours worked (weighted): {weighted_hrs:.1f}")

#------------------------------------------------------------------------------
# STEP 9: Simple Difference-in-Differences Calculation
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Calculate weighted means by group
treat_pre = df_analysis[(df_analysis['treatment'] == 1) & (df_analysis['post'] == 0)]
treat_post = df_analysis[(df_analysis['treatment'] == 1) & (df_analysis['post'] == 1)]
control_pre = df_analysis[(df_analysis['treatment'] == 0) & (df_analysis['post'] == 0)]
control_post = df_analysis[(df_analysis['treatment'] == 0) & (df_analysis['post'] == 1)]

mean_treat_pre = np.average(treat_pre['fulltime'], weights=treat_pre['PERWT'])
mean_treat_post = np.average(treat_post['fulltime'], weights=treat_post['PERWT'])
mean_control_pre = np.average(control_pre['fulltime'], weights=control_pre['PERWT'])
mean_control_post = np.average(control_post['fulltime'], weights=control_post['PERWT'])

diff_treat = mean_treat_post - mean_treat_pre
diff_control = mean_control_post - mean_control_pre
did_estimate = diff_treat - diff_control

print(f"\nTreatment Group:")
print(f"   Pre-period mean:  {mean_treat_pre:.4f}")
print(f"   Post-period mean: {mean_treat_post:.4f}")
print(f"   Difference:       {diff_treat:.4f}")

print(f"\nControl Group:")
print(f"   Pre-period mean:  {mean_control_pre:.4f}")
print(f"   Post-period mean: {mean_control_post:.4f}")
print(f"   Difference:       {diff_control:.4f}")

print(f"\nDifference-in-Differences Estimate: {did_estimate:.4f}")

#------------------------------------------------------------------------------
# STEP 10: DiD Regression Analysis
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df_analysis).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (Weighted) ---")
model2 = smf.wls('fulltime ~ treatment + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with controls (weighted)
print("\n--- Model 3: DiD with Controls (Weighted) ---")
df_analysis['age_sq'] = df_analysis['AGE'] ** 2
df_analysis['male'] = (df_analysis['SEX'] == 1).astype(int)

model3 = smf.wls('fulltime ~ treatment + post + treat_post + AGE + age_sq + male',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects (weighted)
print("\n--- Model 4: DiD with Year FE (Weighted) ---")
df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model4 = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + AGE + age_sq + male',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
# Only show key coefficients
print("Key Coefficients:")
print(f"   treat_post: {model4.params['treat_post']:.6f} (SE: {model4.bse['treat_post']:.6f})")
print(f"   treatment:  {model4.params['treatment']:.6f} (SE: {model4.bse['treatment']:.6f})")

# Model 5: DiD with state fixed effects (weighted)
print("\n--- Model 5: DiD with State FE (Weighted) ---")
model5 = smf.wls('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq + male',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit()
print("Key Coefficients:")
print(f"   treat_post: {model5.params['treat_post']:.6f} (SE: {model5.bse['treat_post']:.6f})")
print(f"   treatment:  {model5.params['treatment']:.6f} (SE: {model5.bse['treatment']:.6f})")

#------------------------------------------------------------------------------
# STEP 11: Robust Standard Errors
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("PREFERRED SPECIFICATION WITH ROBUST SE")
print("="*60)

# Model with robust (HC1) standard errors
model_robust = smf.wls('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq + male',
                        data=df_analysis,
                        weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nPreferred Model: DiD with Year and State FE, Controls, Robust SE")
print(f"\n   DiD Estimate (treat_post): {model_robust.params['treat_post']:.6f}")
print(f"   Robust Std Error:          {model_robust.bse['treat_post']:.6f}")
print(f"   t-statistic:               {model_robust.tvalues['treat_post']:.4f}")
print(f"   p-value:                   {model_robust.pvalues['treat_post']:.4f}")
ci = model_robust.conf_int().loc['treat_post']
print(f"   95% CI:                    [{ci[0]:.6f}, {ci[1]:.6f}]")
print(f"\n   N:                         {int(model_robust.nobs):,}")
print(f"   R-squared:                 {model_robust.rsquared:.4f}")

#------------------------------------------------------------------------------
# STEP 12: Event Study / Pre-Trends Analysis
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("EVENT STUDY ANALYSIS")
print("="*60)

# Create year interaction terms
df_analysis['treat_2006'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['treat_2007'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['treat_2008'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['treat_2009'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['treat_2010'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2010).astype(int)
df_analysis['treat_2011'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2011).astype(int)
# 2012 excluded
df_analysis['treat_2013'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['treat_2014'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['treat_2015'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['treat_2016'] = df_analysis['treatment'] * (df_analysis['YEAR'] == 2016).astype(int)

# Reference year: 2011 (last pre-treatment year)
event_model = smf.wls(
    'fulltime ~ treatment + C(YEAR) + C(STATEFIP) + '
    'treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + '
    'treat_2013 + treat_2014 + treat_2015 + treat_2016 + '
    'AGE + age_sq + male',
    data=df_analysis,
    weights=df_analysis['PERWT']
).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-"*50)
event_years = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
               'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for year_var in event_years:
    coef = event_model.params[year_var]
    se = event_model.bse[year_var]
    pval = event_model.pvalues[year_var]
    sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
    print(f"   {year_var}: {coef:8.4f} (SE: {se:.4f}) {sig}")

#------------------------------------------------------------------------------
# STEP 13: Robustness Checks
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# Robustness 1: Alternative control group (late arrivals)
print("\n--- Robustness 1: Alternative Control (Late Arrivals) ---")
df_robust1 = df_sample[(df_sample['daca_eligible'] | df_sample['control_late_arrival'])].copy()
df_robust1['treatment'] = df_robust1['daca_eligible'].astype(int)
df_robust1['treat_post'] = df_robust1['treatment'] * df_robust1['post']
df_robust1['age_sq'] = df_robust1['AGE'] ** 2
df_robust1['male'] = (df_robust1['SEX'] == 1).astype(int)

if len(df_robust1) > 100:
    rob1_model = smf.wls('fulltime ~ treatment + C(YEAR) + treat_post + AGE + age_sq + male',
                          data=df_robust1,
                          weights=df_robust1['PERWT']).fit(cov_type='HC1')
    print(f"   DiD Estimate: {rob1_model.params['treat_post']:.6f} (SE: {rob1_model.bse['treat_post']:.6f})")
    print(f"   N: {int(rob1_model.nobs):,}")
else:
    print("   Insufficient observations for this specification")

# Robustness 2: Probit model
print("\n--- Robustness 2: Probit Model ---")
try:
    probit_model = smf.probit('fulltime ~ treatment + post + treat_post + AGE + age_sq + male',
                               data=df_analysis).fit(disp=0)
    # Marginal effect at mean
    margeff = probit_model.get_margeff(at='mean')
    treat_post_idx = list(margeff.summary_frame().index).index('treat_post')
    me_coef = margeff.margeff[treat_post_idx]
    me_se = margeff.margeff_se[treat_post_idx]
    print(f"   Marginal Effect (treat_post): {me_coef:.6f} (SE: {me_se:.6f})")
except Exception as e:
    print(f"   Could not estimate probit model: {e}")

# Robustness 3: Restrict to ages 18-30 (prime DACA-eligible ages)
print("\n--- Robustness 3: Ages 18-30 Only ---")
df_robust3 = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 30)].copy()
if len(df_robust3) > 100:
    rob3_model = smf.wls('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq + male',
                          data=df_robust3,
                          weights=df_robust3['PERWT']).fit(cov_type='HC1')
    print(f"   DiD Estimate: {rob3_model.params['treat_post']:.6f} (SE: {rob3_model.bse['treat_post']:.6f})")
    print(f"   N: {int(rob3_model.nobs):,}")
else:
    print("   Insufficient observations for this specification")

# Robustness 4: Males only
print("\n--- Robustness 4: Males Only ---")
df_robust4 = df_analysis[df_analysis['male'] == 1].copy()
if len(df_robust4) > 100:
    rob4_model = smf.wls('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq',
                          data=df_robust4,
                          weights=df_robust4['PERWT']).fit(cov_type='HC1')
    print(f"   DiD Estimate: {rob4_model.params['treat_post']:.6f} (SE: {rob4_model.bse['treat_post']:.6f})")
    print(f"   N: {int(rob4_model.nobs):,}")
else:
    print("   Insufficient observations for this specification")

# Robustness 5: Females only
print("\n--- Robustness 5: Females Only ---")
df_robust5 = df_analysis[df_analysis['male'] == 0].copy()
if len(df_robust5) > 100:
    rob5_model = smf.wls('fulltime ~ treatment + C(YEAR) + C(STATEFIP) + treat_post + AGE + age_sq',
                          data=df_robust5,
                          weights=df_robust5['PERWT']).fit(cov_type='HC1')
    print(f"   DiD Estimate: {rob5_model.params['treat_post']:.6f} (SE: {rob5_model.bse['treat_post']:.6f})")
    print(f"   N: {int(rob5_model.nobs):,}")
else:
    print("   Insufficient observations for this specification")

#------------------------------------------------------------------------------
# STEP 14: Save Results for Report
#------------------------------------------------------------------------------
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Create results dictionary
results = {
    'did_simple': did_estimate,
    'did_regression': model_robust.params['treat_post'],
    'did_se': model_robust.bse['treat_post'],
    'did_pvalue': model_robust.pvalues['treat_post'],
    'did_ci_low': ci[0],
    'did_ci_high': ci[1],
    'n_obs': int(model_robust.nobs),
    'n_treatment': len(df_analysis[df_analysis['treatment'] == 1]),
    'n_control': len(df_analysis[df_analysis['treatment'] == 0]),
    'mean_fulltime_treat_pre': mean_treat_pre,
    'mean_fulltime_treat_post': mean_treat_post,
    'mean_fulltime_control_pre': mean_control_pre,
    'mean_fulltime_control_post': mean_control_post,
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("   Saved results_summary.csv")

# Save event study coefficients
event_results = []
for year_var in event_years:
    year = int(year_var.split('_')[1])
    event_results.append({
        'year': year,
        'coefficient': event_model.params[year_var],
        'std_error': event_model.bse[year_var],
        'p_value': event_model.pvalues[year_var],
        'ci_low': event_model.conf_int().loc[year_var, 0],
        'ci_high': event_model.conf_int().loc[year_var, 1]
    })
# Add reference year
event_results.append({
    'year': 2011,
    'coefficient': 0,
    'std_error': 0,
    'p_value': 1,
    'ci_low': 0,
    'ci_high': 0
})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("   Saved event_study_results.csv")

# Save summary statistics
summary_output = df_analysis.groupby(['treatment', 'post']).apply(
    lambda x: pd.Series({
        'n_unweighted': len(x),
        'n_weighted': x['PERWT'].sum(),
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'age_mean': np.average(x['AGE'], weights=x['PERWT']),
        'male_share': np.average(x['male'], weights=x['PERWT']),
        'hours_mean': np.average(x['UHRSWORK'], weights=x['PERWT'])
    })
).reset_index()
summary_output.to_csv('summary_statistics.csv', index=False)
print("   Saved summary_statistics.csv")

#------------------------------------------------------------------------------
# FINAL OUTPUT
#------------------------------------------------------------------------------
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"""
RESEARCH QUESTION:
    Impact of DACA eligibility on full-time employment (35+ hours/week)
    among Hispanic-Mexican Mexican-born non-citizens in the US.

IDENTIFICATION STRATEGY:
    Difference-in-Differences comparing DACA-eligible individuals to
    similar non-eligible individuals (childhood arrivals who were too old
    for DACA eligibility, born before June 1981).

PREFERRED ESTIMATE:
    DiD Coefficient:  {model_robust.params['treat_post']:.4f}
    Standard Error:   {model_robust.bse['treat_post']:.4f}
    95% CI:           [{ci[0]:.4f}, {ci[1]:.4f}]
    p-value:          {model_robust.pvalues['treat_post']:.4f}

SAMPLE SIZE:
    Total N:          {int(model_robust.nobs):,}
    Treatment:        {len(df_analysis[df_analysis['treatment']==1]):,}
    Control:          {len(df_analysis[df_analysis['treatment']==0]):,}

INTERPRETATION:
    DACA eligibility is associated with a {model_robust.params['treat_post']:.1%}
    {'increase' if model_robust.params['treat_post'] > 0 else 'decrease'} in the
    probability of full-time employment.
    This effect is {'statistically significant' if model_robust.pvalues['treat_post'] < 0.05 else 'not statistically significant'}
    at the 5% level.
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)

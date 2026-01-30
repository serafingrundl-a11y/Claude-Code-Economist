"""
DACA Replication Analysis - Participant 31
Effect of DACA eligibility on full-time employment among Hispanic-Mexican Mexican-born individuals

Methodology: Difference-in-Differences
Treatment group: Ages 26-30 as of June 15, 2012 (eligible for DACA)
Control group: Ages 31-35 as of June 15, 2012 (would be eligible if not for age cutoff)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("="*80)
print("DACA REPLICATION ANALYSIS - PARTICIPANT 31")
print("="*80)

# Load data
print("\n1. LOADING DATA...")
df = pd.read_csv('data/data.csv')
print(f"   Total observations loaded: {len(df):,}")
print(f"   Years in data: {sorted(df['YEAR'].unique())}")

# ==============================================================================
# STEP 1: Define sample restrictions
# ==============================================================================
print("\n2. DEFINING SAMPLE RESTRICTIONS...")

# Hispanic-Mexican ethnicity (HISPAN=1 for Mexican)
print(f"   HISPAN values: {df['HISPAN'].value_counts().to_dict()}")
df_sample = df[df['HISPAN'] == 1].copy()
print(f"   After Hispanic-Mexican restriction: {len(df_sample):,}")

# Born in Mexico (BPL=200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"   After Mexico birthplace restriction: {len(df_sample):,}")

# Non-citizens only (CITIZEN=3) - assumed undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After non-citizen restriction: {len(df_sample):,}")

# ==============================================================================
# STEP 2: Calculate age as of June 15, 2012 and define treatment/control groups
# ==============================================================================
print("\n3. CALCULATING AGE AS OF JUNE 15, 2012 AND DEFINING GROUPS...")

# Calculate age as of June 15, 2012
# Using BIRTHYR and BIRTHQTR to be precise
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If born in Q1-Q2 of year Y, they reached their birthday by June 15, 2012
# If born in Q3-Q4 of year Y, they had NOT reached their birthday by June 15, 2012

def age_as_of_june_2012(row):
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # Age is 2012 - birth_year, but subtract 1 if birthday hasn't occurred yet
    # June 15 is roughly in Q2, so:
    # Q1 (Jan-Mar): birthday passed by June 15
    # Q2 (Apr-Jun): assume roughly half passed, use middle of quarter logic
    #               For simplicity, Q2 birthdays assumed to have passed by June 15
    # Q3-Q4: birthday NOT yet reached by June 15

    age = 2012 - birth_year
    if birth_qtr in [3, 4]:  # Born Jul-Dec, birthday not reached by June 15
        age = age - 1
    return age

df_sample['age_june_2012'] = df_sample.apply(age_as_of_june_2012, axis=1)

print(f"   Age range in data: {df_sample['age_june_2012'].min()} to {df_sample['age_june_2012'].max()}")

# ==============================================================================
# STEP 3: Additional DACA eligibility criteria
# ==============================================================================
print("\n4. APPLYING ADDITIONAL DACA ELIGIBILITY CRITERIA...")

# Criterion: Arrived in US before 16th birthday
# YRIMMIG = year of immigration, BIRTHYR = birth year
# Age at arrival = YRIMMIG - BIRTHYR
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Filter for those who arrived before age 16
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['age_at_arrival'] < 16)].copy()
print(f"   After arrived-before-16 restriction: {len(df_sample):,}")

# Criterion: Continuous residence since June 15, 2007 (at least ~5 years by 2012)
# This is approximated by YRSUSA1 (years in USA) - we require at least 5 years
# But YRSUSA is calculated relative to survey year, so we need to adjust
# For now, we'll use YRIMMIG <= 2007 as proxy
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"   After continuous-residence restriction (YRIMMIG<=2007): {len(df_sample):,}")

# ==============================================================================
# STEP 4: Define treatment and control groups based on age as of June 2012
# ==============================================================================
print("\n5. DEFINING TREATMENT AND CONTROL GROUPS...")

# Treatment group: 26-30 as of June 15, 2012 (DACA eligible)
# Control group: 31-35 as of June 15, 2012 (too old for DACA)

df_sample['treat'] = ((df_sample['age_june_2012'] >= 26) &
                       (df_sample['age_june_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_june_2012'] >= 31) &
                         (df_sample['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treat'] == 1) | (df_sample['control'] == 1)].copy()
print(f"   Treatment group (26-30 in 2012): {df_analysis['treat'].sum():,}")
print(f"   Control group (31-35 in 2012): {(df_analysis['control']).sum():,}")
print(f"   Total analysis sample: {len(df_analysis):,}")

# ==============================================================================
# STEP 5: Define pre and post periods
# ==============================================================================
print("\n6. DEFINING PRE AND POST PERIODS...")

# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA, excluding 2012)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Exclude 2012 (DACA was implemented mid-year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()

print(f"   Pre-period years: {sorted(df_analysis[df_analysis['post']==0]['YEAR'].unique())}")
print(f"   Post-period years: {sorted(df_analysis[df_analysis['post']==1]['YEAR'].unique())}")
print(f"   Pre-period observations: {(df_analysis['post']==0).sum():,}")
print(f"   Post-period observations: {(df_analysis['post']==1).sum():,}")

# ==============================================================================
# STEP 6: Define outcome variable
# ==============================================================================
print("\n7. DEFINING OUTCOME VARIABLE...")

# Full-time employment: UHRSWORK >= 35
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"   Overall full-time employment rate: {df_analysis['fulltime'].mean():.3f}")
print(f"   Full-time employment by treatment status:")
print(f"      Treatment (26-30): {df_analysis[df_analysis['treat']==1]['fulltime'].mean():.3f}")
print(f"      Control (31-35): {df_analysis[df_analysis['control']==1]['fulltime'].mean():.3f}")

# ==============================================================================
# STEP 7: Summary statistics
# ==============================================================================
print("\n8. SUMMARY STATISTICS...")

# By group and period
summary = df_analysis.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'SEX': lambda x: (x==2).mean(),  # Proportion female
    'EDUC': 'mean'
}).round(4)
print(summary)

# Sample sizes by year and group
print("\n   Sample sizes by year and treatment status:")
print(df_analysis.groupby(['YEAR', 'treat']).size().unstack())

# ==============================================================================
# STEP 8: Simple DiD calculation
# ==============================================================================
print("\n9. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION...")

# Calculate means by group and period (unweighted)
means = df_analysis.groupby(['treat', 'post'])['fulltime'].mean()
print("   Mean full-time employment rates:")
print(means)

# DiD calculation
treat_pre = means[(1, 0)]
treat_post = means[(1, 1)]
control_pre = means[(0, 0)]
control_post = means[(0, 1)]

did_estimate = (treat_post - treat_pre) - (control_post - control_pre)

print(f"\n   Treatment group change: {treat_post - treat_pre:.4f}")
print(f"   Control group change: {control_post - control_pre:.4f}")
print(f"   DiD estimate (unweighted): {did_estimate:.4f}")

# Weighted DiD calculation
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

means_wt = df_analysis.groupby(['treat', 'post']).apply(weighted_mean)
print("\n   Weighted mean full-time employment rates:")
print(means_wt)

treat_pre_wt = means_wt[(1, 0)]
treat_post_wt = means_wt[(1, 1)]
control_pre_wt = means_wt[(0, 0)]
control_post_wt = means_wt[(0, 1)]

did_estimate_wt = (treat_post_wt - treat_pre_wt) - (control_post_wt - control_pre_wt)
print(f"\n   DiD estimate (weighted): {did_estimate_wt:.4f}")

# ==============================================================================
# STEP 9: Regression-based DiD estimation
# ==============================================================================
print("\n10. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES...")

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

# Model 1: Basic DiD without covariates
print("\n   Model 1: Basic DiD (unweighted)")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: Basic DiD with weights
print("\n   Model 2: Basic DiD (weighted)")
model2 = smf.wls('fulltime ~ treat + post + treat_post', data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model2.summary().tables[1])

# ==============================================================================
# STEP 10: DiD with covariates
# ==============================================================================
print("\n11. DIFFERENCE-IN-DIFFERENCES WITH COVARIATES...")

# Create additional control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 7).astype(int)  # Some college or more

# Model 3: DiD with demographic covariates (weighted)
print("\n   Model 3: DiD with demographics (weighted)")
model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects (weighted)
print("\n   Model 4: DiD with year fixed effects (weighted)")
model4 = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
# Print only key coefficients
print(f"   treat_post coefficient: {model4.params['treat_post']:.4f}")
print(f"   treat_post std error: {model4.bse['treat_post']:.4f}")
print(f"   treat_post t-stat: {model4.tvalues['treat_post']:.4f}")
print(f"   treat_post p-value: {model4.pvalues['treat_post']:.4f}")

# Model 5: DiD with state fixed effects (weighted)
print("\n   Model 5: DiD with state fixed effects (weighted)")
model5 = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + C(STATEFIP) + female + married + educ_hs',
                  data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"   treat_post coefficient: {model5.params['treat_post']:.4f}")
print(f"   treat_post std error: {model5.bse['treat_post']:.4f}")
print(f"   treat_post t-stat: {model5.tvalues['treat_post']:.4f}")
print(f"   treat_post p-value: {model5.pvalues['treat_post']:.4f}")

# ==============================================================================
# STEP 11: Additional specifications
# ==============================================================================
print("\n12. ADDITIONAL SPECIFICATIONS AND ROBUSTNESS...")

# Model 6: Triple difference by sex
print("\n   Model 6: By sex")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df_analysis[df_analysis['SEX'] == sex_val]
    model_sex = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"   {sex_name}: DiD = {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f})")

# Model 7: By education
print("\n   Model 7: By education level")
df_low_ed = df_analysis[df_analysis['educ_hs'] == 0]
df_high_ed = df_analysis[df_analysis['educ_hs'] == 1]

if len(df_low_ed) > 100:
    model_low = smf.wls('fulltime ~ treat + post + treat_post',
                         data=df_low_ed, weights=df_low_ed['PERWT']).fit(cov_type='HC1')
    print(f"   Less than HS: DiD = {model_low.params['treat_post']:.4f} (SE: {model_low.bse['treat_post']:.4f})")

if len(df_high_ed) > 100:
    model_high = smf.wls('fulltime ~ treat + post + treat_post',
                          data=df_high_ed, weights=df_high_ed['PERWT']).fit(cov_type='HC1')
    print(f"   HS or more: DiD = {model_high.params['treat_post']:.4f} (SE: {model_high.bse['treat_post']:.4f})")

# ==============================================================================
# STEP 12: Event study / pre-trends analysis
# ==============================================================================
print("\n13. EVENT STUDY / PRE-TRENDS ANALYSIS...")

# Create year-specific treatment interactions
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

# Interactions with treatment (omitting 2011 as reference)
df_analysis['treat_2006'] = df_analysis['treat'] * df_analysis['year_2006']
df_analysis['treat_2007'] = df_analysis['treat'] * df_analysis['year_2007']
df_analysis['treat_2008'] = df_analysis['treat'] * df_analysis['year_2008']
df_analysis['treat_2009'] = df_analysis['treat'] * df_analysis['year_2009']
df_analysis['treat_2010'] = df_analysis['treat'] * df_analysis['year_2010']
df_analysis['treat_2013'] = df_analysis['treat'] * df_analysis['year_2013']
df_analysis['treat_2014'] = df_analysis['treat'] * df_analysis['year_2014']
df_analysis['treat_2015'] = df_analysis['treat'] * df_analysis['year_2015']
df_analysis['treat_2016'] = df_analysis['treat'] * df_analysis['year_2016']

# Event study regression
event_formula = 'fulltime ~ treat + C(YEAR) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016'
model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("   Event study coefficients (2011 = reference year):")
event_years = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
               'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
for var in event_years:
    year = var.split('_')[1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"   {year}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# ==============================================================================
# STEP 13: Final preferred specification
# ==============================================================================
print("\n" + "="*80)
print("14. PREFERRED SPECIFICATION - FINAL RESULTS")
print("="*80)

# Preferred model: DiD with year and state FE, demographics
print("\n   Preferred Model: DiD with year FE, state FE, and demographics")
print(f"\n   Sample size: {len(df_analysis):,}")
print(f"   Treatment group (26-30 in 2012): {df_analysis['treat'].sum():,}")
print(f"   Control group (31-35 in 2012): {(df_analysis['treat']==0).sum():,}")

# Run preferred specification
preferred_model = smf.wls('fulltime ~ treat + treat_post + C(YEAR) + C(STATEFIP) + female + married + educ_hs',
                           data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

did_coef = preferred_model.params['treat_post']
did_se = preferred_model.bse['treat_post']
did_ci_low = did_coef - 1.96 * did_se
did_ci_high = did_coef + 1.96 * did_se
did_pval = preferred_model.pvalues['treat_post']

print(f"\n   PREFERRED ESTIMATE:")
print(f"   -------------------------------------------------")
print(f"   DiD Effect (treat x post): {did_coef:.4f}")
print(f"   Standard Error (robust):   {did_se:.4f}")
print(f"   95% Confidence Interval:   [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"   t-statistic:               {preferred_model.tvalues['treat_post']:.4f}")
print(f"   p-value:                   {did_pval:.4f}")
print(f"   R-squared:                 {preferred_model.rsquared:.4f}")
print(f"   -------------------------------------------------")

# ==============================================================================
# STEP 14: Save results for LaTeX report
# ==============================================================================
print("\n15. SAVING RESULTS FOR REPORT...")

# Create results dictionary
results = {
    'sample_size': len(df_analysis),
    'n_treatment': df_analysis['treat'].sum(),
    'n_control': (df_analysis['treat']==0).sum(),
    'did_estimate': did_coef,
    'did_se': did_se,
    'did_ci_low': did_ci_low,
    'did_ci_high': did_ci_high,
    'did_pval': did_pval,
    'r_squared': preferred_model.rsquared,
    'pre_treat_mean': treat_pre_wt,
    'pre_control_mean': control_pre_wt,
    'post_treat_mean': treat_post_wt,
    'post_control_mean': control_post_wt,
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)

# Save model comparison table
model_comparison = pd.DataFrame({
    'Model': ['Basic DiD (unweighted)', 'Basic DiD (weighted)', 'DiD + Demographics',
              'DiD + Year FE', 'DiD + Year & State FE (Preferred)'],
    'DiD_Estimate': [model1.params['treat_post'], model2.params['treat_post'],
                     model3.params['treat_post'], model4.params['treat_post'],
                     model5.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post'],
                  model5.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model5.pvalues['treat_post']],
})
model_comparison.to_csv('model_comparison.csv', index=False)
print("   Results saved to results_summary.csv and model_comparison.csv")

# Save event study coefficients
event_results = []
for var in event_years:
    year = int(var.split('_')[1])
    event_results.append({
        'year': year,
        'coefficient': model_event.params[var],
        'std_error': model_event.bse[var],
        'ci_low': model_event.params[var] - 1.96 * model_event.bse[var],
        'ci_high': model_event.params[var] + 1.96 * model_event.bse[var]
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Full-time employment rates by group and period
ft_rates = pd.DataFrame({
    'Group': ['Treatment (26-30)', 'Control (31-35)'],
    'Pre_Period': [treat_pre_wt, control_pre_wt],
    'Post_Period': [treat_post_wt, control_post_wt],
    'Change': [treat_post_wt - treat_pre_wt, control_post_wt - control_pre_wt]
})
ft_rates.to_csv('fulltime_rates.csv', index=False)

# Demographics summary
demo_summary = df_analysis.groupby('treat').agg({
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'AGE': 'mean',
    'age_june_2012': 'mean'
}).round(3)
demo_summary.to_csv('demographics_summary.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

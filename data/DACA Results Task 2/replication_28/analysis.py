"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. Load Data
# =============================================================================
print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

df = pd.read_csv('data/data.csv')
print(f"\nTotal observations in raw data: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. Define Target Population
# =============================================================================
print("\n" + "=" * 60)
print("DEFINING TARGET POPULATION")
print("=" * 60)

# Hispanic-Mexican ethnicity (HISPAN == 1)
df_target = df[df['HISPAN'] == 1].copy()
print(f"After filtering Hispanic-Mexican: {len(df_target):,}")

# Born in Mexico (BPL == 200)
df_target = df_target[df_target['BPL'] == 200]
print(f"After filtering born in Mexico: {len(df_target):,}")

# Not a citizen (CITIZEN == 3)
df_target = df_target[df_target['CITIZEN'] == 3]
print(f"After filtering non-citizens: {len(df_target):,}")

# =============================================================================
# 3. Calculate Age as of June 15, 2012
# =============================================================================
print("\n" + "=" * 60)
print("CALCULATING AGE AS OF JUNE 15, 2012")
print("=" * 60)

# Birth quarter mapping to approximate birth month midpoint
# Q1: Jan-Mar -> ~Feb 15 (month 2.5)
# Q2: Apr-Jun -> ~May 15 (month 5.5)
# Q3: Jul-Sep -> ~Aug 15 (month 8.5)
# Q4: Oct-Dec -> ~Nov 15 (month 11.5)

def calculate_age_june_2012(row):
    """Calculate age as of June 15, 2012"""
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']

    # Approximate birth month based on quarter
    if birth_qtr == 1:  # Jan-Mar
        birth_month = 2  # Approximate as Feb
    elif birth_qtr == 2:  # Apr-Jun
        birth_month = 5  # Approximate as May
    elif birth_qtr == 3:  # Jul-Sep
        birth_month = 8  # Approximate as Aug
    else:  # Oct-Dec
        birth_month = 11  # Approximate as Nov

    # Age as of June 15, 2012
    age = 2012 - birth_year
    if birth_month > 6:  # Birthday hasn't occurred yet in 2012
        age -= 1

    return age

df_target['age_june_2012'] = df_target.apply(calculate_age_june_2012, axis=1)

print(f"Age distribution as of June 15, 2012:")
print(df_target['age_june_2012'].describe())

# =============================================================================
# 4. Define Treatment and Control Groups
# =============================================================================
print("\n" + "=" * 60)
print("DEFINING TREATMENT AND CONTROL GROUPS")
print("=" * 60)

# Treatment: Ages 26-30 as of June 15, 2012
# Control: Ages 31-35 as of June 15, 2012

df_target['treat'] = ((df_target['age_june_2012'] >= 26) &
                       (df_target['age_june_2012'] <= 30)).astype(int)
df_target['control'] = ((df_target['age_june_2012'] >= 31) &
                         (df_target['age_june_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_sample = df_target[(df_target['treat'] == 1) | (df_target['control'] == 1)].copy()
print(f"Sample size (treatment + control groups): {len(df_sample):,}")
print(f"Treatment group (ages 26-30): {df_sample['treat'].sum():,}")
print(f"Control group (ages 31-35): {(df_sample['control'] == 1).sum():,}")

# =============================================================================
# 5. Apply Additional DACA Eligibility Criteria
# =============================================================================
print("\n" + "=" * 60)
print("APPLYING DACA ELIGIBILITY CRITERIA")
print("=" * 60)

# 5a. Must have arrived before their 16th birthday
# Calculate age at immigration
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immigration'] < 16]
print(f"After filtering arrived before age 16: {len(df_sample):,}")

# 5b. Must have been in US since June 15, 2007 (5 years continuous residence)
# YRIMMIG <= 2007
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
print(f"After filtering arrived by 2007: {len(df_sample):,}")

# 5c. Must have been present in US on June 15, 2012
# We approximate this by requiring they were in the US by the survey year
# This is implicit in the ACS sample

print(f"\nFinal sample size: {len(df_sample):,}")
print(f"Treatment group: {df_sample['treat'].sum():,}")
print(f"Control group: {(df_sample['treat'] == 0).sum():,}")

# =============================================================================
# 6. Define Outcome Variable and Time Periods
# =============================================================================
print("\n" + "=" * 60)
print("DEFINING OUTCOME AND TIME PERIODS")
print("=" * 60)

# Full-time employment: UHRSWORK >= 35
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Post period: 2013-2016 (after DACA)
# Pre period: 2006-2011 (before DACA)
# Exclude 2012 (implementation year)
df_sample = df_sample[df_sample['YEAR'] != 2012]
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

print(f"Sample after excluding 2012: {len(df_sample):,}")
print(f"Pre-period (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_sample['post'] == 1).sum():,}")

# Create interaction term
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

# =============================================================================
# 7. Summary Statistics
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Pre-period summary
pre = df_sample[df_sample['post'] == 0]
post = df_sample[df_sample['post'] == 1]

print("\n--- Pre-Period (2006-2011) ---")
print(f"Treatment group N: {pre[pre['treat']==1].shape[0]:,}")
print(f"Control group N: {pre[pre['treat']==0].shape[0]:,}")
print(f"Treatment FT rate: {pre[pre['treat']==1]['fulltime'].mean():.4f}")
print(f"Control FT rate: {pre[pre['treat']==0]['fulltime'].mean():.4f}")

print("\n--- Post-Period (2013-2016) ---")
print(f"Treatment group N: {post[post['treat']==1].shape[0]:,}")
print(f"Control group N: {post[post['treat']==0].shape[0]:,}")
print(f"Treatment FT rate: {post[post['treat']==1]['fulltime'].mean():.4f}")
print(f"Control FT rate: {post[post['treat']==0]['fulltime'].mean():.4f}")

# Weighted means
print("\n--- Weighted Statistics ---")
pre_treat_wt = np.average(pre[pre['treat']==1]['fulltime'],
                          weights=pre[pre['treat']==1]['PERWT'])
pre_ctrl_wt = np.average(pre[pre['treat']==0]['fulltime'],
                         weights=pre[pre['treat']==0]['PERWT'])
post_treat_wt = np.average(post[post['treat']==1]['fulltime'],
                           weights=post[post['treat']==1]['PERWT'])
post_ctrl_wt = np.average(post[post['treat']==0]['fulltime'],
                          weights=post[post['treat']==0]['PERWT'])

print(f"Pre-Treatment FT rate (weighted): {pre_treat_wt:.4f}")
print(f"Pre-Control FT rate (weighted): {pre_ctrl_wt:.4f}")
print(f"Post-Treatment FT rate (weighted): {post_treat_wt:.4f}")
print(f"Post-Control FT rate (weighted): {post_ctrl_wt:.4f}")

# Simple DiD calculation
did_simple = (post_treat_wt - pre_treat_wt) - (post_ctrl_wt - pre_ctrl_wt)
print(f"\nSimple DiD estimate (weighted): {did_simple:.4f}")

# =============================================================================
# 8. Regression Analysis
# =============================================================================
print("\n" + "=" * 60)
print("REGRESSION ANALYSIS")
print("=" * 60)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Controls ---")
# Create control variables
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'] == 1).astype(int)

# Education categories
df_sample['educ_hs'] = (df_sample['EDUC'] >= 6).astype(int)  # HS or more
df_sample['educ_college'] = (df_sample['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
model3 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + C(YEAR)',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD coefficient (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard error: {model3.bse['treat_post']:.4f}")
print(f"t-statistic: {model3.tvalues['treat_post']:.4f}")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: Full model with state fixed effects
print("\n--- Model 4: Full Model with State Fixed Effects ---")
model4 = smf.wls('fulltime ~ treat + treat_post + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                  data=df_sample, weights=df_sample['PERWT']).fit()
print(f"DiD coefficient (treat_post): {model4.params['treat_post']:.4f}")
print(f"Standard error: {model4.bse['treat_post']:.4f}")
print(f"t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")

# =============================================================================
# 9. Heterogeneity Analysis by Sex
# =============================================================================
print("\n" + "=" * 60)
print("HETEROGENEITY ANALYSIS BY SEX")
print("=" * 60)

# Males
df_male = df_sample[df_sample['SEX'] == 1]
model_male = smf.wls('fulltime ~ treat + post + treat_post + married + educ_hs',
                      data=df_male, weights=df_male['PERWT']).fit()
print(f"\nMales - DiD coefficient: {model_male.params['treat_post']:.4f}")
print(f"Males - SE: {model_male.bse['treat_post']:.4f}")
print(f"Males - p-value: {model_male.pvalues['treat_post']:.4f}")
print(f"Males - N: {len(df_male):,}")

# Females
df_female = df_sample[df_sample['SEX'] == 2]
model_female = smf.wls('fulltime ~ treat + post + treat_post + married + educ_hs',
                        data=df_female, weights=df_female['PERWT']).fit()
print(f"\nFemales - DiD coefficient: {model_female.params['treat_post']:.4f}")
print(f"Females - SE: {model_female.bse['treat_post']:.4f}")
print(f"Females - p-value: {model_female.pvalues['treat_post']:.4f}")
print(f"Females - N: {len(df_female):,}")

# =============================================================================
# 10. Robustness Checks
# =============================================================================
print("\n" + "=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)

# Robustness 1: Alternative age windows
print("\n--- Robustness: Narrower Age Window (27-29 vs 32-34) ---")
df_narrow = df_target.copy()
df_narrow = df_narrow[df_narrow['YRIMMIG'] - df_narrow['BIRTHYR'] < 16]
df_narrow = df_narrow[df_narrow['YRIMMIG'] <= 2007]
df_narrow = df_narrow[df_narrow['YEAR'] != 2012]

df_narrow['treat_narrow'] = ((df_narrow['age_june_2012'] >= 27) &
                              (df_narrow['age_june_2012'] <= 29)).astype(int)
df_narrow['ctrl_narrow'] = ((df_narrow['age_june_2012'] >= 32) &
                             (df_narrow['age_june_2012'] <= 34)).astype(int)
df_narrow = df_narrow[(df_narrow['treat_narrow'] == 1) | (df_narrow['ctrl_narrow'] == 1)]
df_narrow['treat'] = df_narrow['treat_narrow']
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = (df_narrow['MARST'] == 1).astype(int)
df_narrow['educ_hs'] = (df_narrow['EDUC'] >= 6).astype(int)

model_narrow = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs',
                        data=df_narrow, weights=df_narrow['PERWT']).fit()
print(f"DiD coefficient: {model_narrow.params['treat_post']:.4f}")
print(f"SE: {model_narrow.bse['treat_post']:.4f}")
print(f"p-value: {model_narrow.pvalues['treat_post']:.4f}")
print(f"N: {len(df_narrow):,}")

# Robustness 2: Placebo test (pre-period only)
print("\n--- Placebo Test: Pre-Period (2006-2008 vs 2009-2011) ---")
df_placebo = df_sample[df_sample['post'] == 0].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['fake_treat_post'] = df_placebo['treat'] * df_placebo['fake_post']

model_placebo = smf.wls('fulltime ~ treat + fake_post + fake_treat_post + female + married + educ_hs',
                         data=df_placebo, weights=df_placebo['PERWT']).fit()
print(f"Placebo DiD coefficient: {model_placebo.params['fake_treat_post']:.4f}")
print(f"SE: {model_placebo.bse['fake_treat_post']:.4f}")
print(f"p-value: {model_placebo.pvalues['fake_treat_post']:.4f}")

# =============================================================================
# 11. Event Study Analysis
# =============================================================================
print("\n" + "=" * 60)
print("EVENT STUDY ANALYSIS")
print("=" * 60)

# Create year dummies interacted with treatment
df_event = df_sample.copy()
df_event['year_2006'] = (df_event['YEAR'] == 2006).astype(int)
df_event['year_2007'] = (df_event['YEAR'] == 2007).astype(int)
df_event['year_2008'] = (df_event['YEAR'] == 2008).astype(int)
df_event['year_2009'] = (df_event['YEAR'] == 2009).astype(int)
df_event['year_2010'] = (df_event['YEAR'] == 2010).astype(int)
df_event['year_2011'] = (df_event['YEAR'] == 2011).astype(int)  # Reference year
df_event['year_2013'] = (df_event['YEAR'] == 2013).astype(int)
df_event['year_2014'] = (df_event['YEAR'] == 2014).astype(int)
df_event['year_2015'] = (df_event['YEAR'] == 2015).astype(int)
df_event['year_2016'] = (df_event['YEAR'] == 2016).astype(int)

# Interactions (excluding 2011 as reference)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_event[f'treat_x_{yr}'] = df_event['treat'] * df_event[f'year_{yr}']

model_event = smf.wls('fulltime ~ treat + C(YEAR) + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016 + female + married + educ_hs',
                       data=df_event, weights=df_event['PERWT']).fit()

print("\nEvent Study Coefficients (relative to 2011):")
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{yr}']
    se = model_event.bse[f'treat_x_{yr}']
    pval = model_event.pvalues[f'treat_x_{yr}']
    print(f"  {yr}: coef = {coef:.4f}, SE = {se:.4f}, p = {pval:.4f}")

# =============================================================================
# 12. Final Results Summary
# =============================================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)

print(f"""
PREFERRED ESTIMATE (Model 3: DiD with Year Fixed Effects):
- DiD Coefficient: {model3.params['treat_post']:.4f}
- Standard Error: {model3.bse['treat_post']:.4f}
- 95% CI: [{model3.params['treat_post'] - 1.96*model3.bse['treat_post']:.4f}, {model3.params['treat_post'] + 1.96*model3.bse['treat_post']:.4f}]
- t-statistic: {model3.tvalues['treat_post']:.4f}
- p-value: {model3.pvalues['treat_post']:.4f}
- Sample Size: {len(df_sample):,}

INTERPRETATION:
DACA eligibility is associated with a {model3.params['treat_post']*100:.2f} percentage point
change in the probability of full-time employment for eligible Hispanic-Mexican,
Mexican-born individuals aged 26-30, compared to similar individuals aged 31-35.
""")

# =============================================================================
# 13. Export Results for LaTeX
# =============================================================================
print("\n" + "=" * 60)
print("EXPORTING RESULTS")
print("=" * 60)

# Create results dictionary
results = {
    'model1_coef': model1.params['treat_post'],
    'model1_se': model1.bse['treat_post'],
    'model1_pval': model1.pvalues['treat_post'],
    'model1_n': int(model1.nobs),

    'model2_coef': model2.params['treat_post'],
    'model2_se': model2.bse['treat_post'],
    'model2_pval': model2.pvalues['treat_post'],
    'model2_n': int(model2.nobs),

    'model3_coef': model3.params['treat_post'],
    'model3_se': model3.bse['treat_post'],
    'model3_pval': model3.pvalues['treat_post'],
    'model3_n': int(model3.nobs),

    'model4_coef': model4.params['treat_post'],
    'model4_se': model4.bse['treat_post'],
    'model4_pval': model4.pvalues['treat_post'],
    'model4_n': int(model4.nobs),

    'pre_treat_ft': pre_treat_wt,
    'pre_ctrl_ft': pre_ctrl_wt,
    'post_treat_ft': post_treat_wt,
    'post_ctrl_ft': post_ctrl_wt,

    'male_coef': model_male.params['treat_post'],
    'male_se': model_male.bse['treat_post'],
    'female_coef': model_female.params['treat_post'],
    'female_se': model_female.bse['treat_post'],

    'placebo_coef': model_placebo.params['fake_treat_post'],
    'placebo_se': model_placebo.bse['fake_treat_post'],
}

# Save to CSV for reference
pd.DataFrame([results]).to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save sample statistics
sample_stats = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs': 'mean',
    'AGE': 'mean'
}).round(4)
sample_stats.to_csv('sample_statistics.csv')
print("Sample statistics saved to sample_statistics.csv")

print("\nAnalysis complete!")

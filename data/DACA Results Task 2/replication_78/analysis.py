"""
DACA Replication Analysis
Research Question: Impact of DACA eligibility on full-time employment (35+ hrs/week)
for Hispanic-Mexican Mexican-born individuals in the US.

Treatment Group: Ages 26-30 at DACA implementation (June 15, 2012)
Control Group: Ages 31-35 at DACA implementation (June 15, 2012)
Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 due to policy timing ambiguity)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_78")

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\n[1] Loading data...")
df = pd.read_csv("data/data.csv")
print(f"Total observations in raw data: {len(df):,}")
print(f"Years available: {sorted(df['YEAR'].unique())}")

# ============================================================================
# STEP 2: Define Sample Restrictions
# ============================================================================
print("\n[2] Applying sample restrictions...")

# Store sample sizes at each step
sample_counts = {}
sample_counts['raw'] = len(df)

# 2a. Keep only years 2006-2016 (excluding 2012)
df = df[df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])]
sample_counts['years_2006_2016_excl_2012'] = len(df)
print(f"After restricting to 2006-2011, 2013-2016: {len(df):,}")

# 2b. Hispanic-Mexican (HISPAN == 1 means Mexican)
df = df[df['HISPAN'] == 1]
sample_counts['hispanic_mexican'] = len(df)
print(f"After restricting to Hispanic-Mexican: {len(df):,}")

# 2c. Born in Mexico (BPL == 200)
df = df[df['BPL'] == 200]
sample_counts['born_mexico'] = len(df)
print(f"After restricting to born in Mexico: {len(df):,}")

# 2d. Not a citizen (CITIZEN == 3 means "Not a citizen")
# This is a proxy for undocumented status as per instructions
df = df[df['CITIZEN'] == 3]
sample_counts['non_citizen'] = len(df)
print(f"After restricting to non-citizens: {len(df):,}")

# ============================================================================
# STEP 3: Define DACA Eligibility Criteria
# ============================================================================
print("\n[3] Applying DACA eligibility criteria...")

# DACA eligibility requirements:
# 1. Arrived in US before 16th birthday
# 2. Under 31 as of June 15, 2012
# 3. Lived continuously in US since June 15, 2007
# 4. Present in US on June 15, 2012

# For age-based eligibility:
# Treatment: Born 1982-1986 (ages 26-30 as of June 15, 2012)
# Control: Born 1977-1981 (ages 31-35 as of June 15, 2012)

# Calculate age as of June 15, 2012
# BIRTHYR is the birth year
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Arrived before 16th birthday: YRIMMIG - BIRTHYR < 16
# We need YRIMMIG (year of immigration)
df = df[df['YRIMMIG'] > 0]  # Valid immigration year
sample_counts['valid_yrimmig'] = len(df)
print(f"After requiring valid immigration year: {len(df):,}")

df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_arrival'] < 16]
sample_counts['arrived_before_16'] = len(df)
print(f"After restricting to arrived before age 16: {len(df):,}")

# Lived continuously in US since June 15, 2007 (at least 5 years by 2012)
# YRIMMIG <= 2007
df = df[df['YRIMMIG'] <= 2007]
sample_counts['continuous_residence'] = len(df)
print(f"After restricting to continuous residence since 2007: {len(df):,}")

# ============================================================================
# STEP 4: Define Treatment and Control Groups
# ============================================================================
print("\n[4] Defining treatment and control groups...")

# Treatment: Ages 26-30 as of June 15, 2012 (birth years 1982-1986)
# Control: Ages 31-35 as of June 15, 2012 (birth years 1977-1981)

# For the control group (ages 31-35), they would NOT have been eligible
# for DACA due to age, but otherwise meet eligibility criteria

df_analysis = df[(df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1986)].copy()
sample_counts['age_eligible'] = len(df_analysis)
print(f"After restricting to birth years 1977-1986: {len(df_analysis):,}")

# Create treatment indicator (1 if born 1982-1986, i.e., ages 26-30 in 2012)
df_analysis['treated'] = ((df_analysis['BIRTHYR'] >= 1982) & (df_analysis['BIRTHYR'] <= 1986)).astype(int)

# Create post-treatment indicator (1 if year >= 2013)
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

print(f"\nTreatment group (born 1982-1986): {df_analysis['treated'].sum():,}")
print(f"Control group (born 1977-1981): {(df_analysis['treated'] == 0).sum():,}")

# ============================================================================
# STEP 5: Define Outcome Variable
# ============================================================================
print("\n[5] Defining outcome variable...")

# Full-time employment: UHRSWORK >= 35
# UHRSWORK = 0 means N/A (not working)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"Full-time employment rate (overall): {df_analysis['fulltime'].mean()*100:.2f}%")

# ============================================================================
# STEP 6: Descriptive Statistics
# ============================================================================
print("\n[6] Descriptive Statistics...")
print("\n--- Sample Sizes by Year and Group ---")
sample_by_year_group = df_analysis.groupby(['YEAR', 'treated']).size().unstack(fill_value=0)
sample_by_year_group.columns = ['Control (31-35)', 'Treatment (26-30)']
print(sample_by_year_group)

print("\n--- Full-Time Employment Rates by Year and Group ---")
ft_by_year_group = df_analysis.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack(fill_value=0) * 100
ft_by_year_group.columns = ['Control (31-35)', 'Treatment (26-30)']
print(ft_by_year_group.round(2))

# Pre-post comparison
print("\n--- Pre-Post Comparison ---")
pre_post_table = df_analysis.groupby(['post', 'treated'])['fulltime'].agg(['mean', 'count']).unstack()
print(pre_post_table)

# ============================================================================
# STEP 7: Difference-in-Differences Analysis
# ============================================================================
print("\n[7] Difference-in-Differences Analysis...")

# Basic DiD without covariates
print("\n--- Model 1: Basic DiD (No Covariates) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post', data=df_analysis).fit(cov_type='HC1')
print(model1.summary().tables[1])

# DiD with year fixed effects
print("\n--- Model 2: DiD with Year Fixed Effects ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model2 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_analysis).fit(cov_type='HC1')
print(f"Treated x Post coefficient: {model2.params['treated_post']:.4f}")
print(f"Std Error: {model2.bse['treated_post']:.4f}")
print(f"t-stat: {model2.tvalues['treated_post']:.4f}")
print(f"p-value: {model2.pvalues['treated_post']:.4f}")

# DiD with covariates
print("\n--- Model 3: DiD with Covariates ---")
# Create covariate indicators
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = df_analysis['MARST'].isin([1, 2]).astype(int)

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or more
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)  # Some college or more

# Age in survey year
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

model3 = smf.ols('fulltime ~ treated + C(year_factor) + treated_post + female + married + educ_hs + age + age_sq',
                 data=df_analysis).fit(cov_type='HC1')
print(f"Treated x Post coefficient: {model3.params['treated_post']:.4f}")
print(f"Std Error: {model3.bse['treated_post']:.4f}")
print(f"t-stat: {model3.tvalues['treated_post']:.4f}")
print(f"p-value: {model3.pvalues['treated_post']:.4f}")

# Model 4: Full model with state FE
print("\n--- Model 4: DiD with Covariates and State Fixed Effects ---")
df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)
model4 = smf.ols('fulltime ~ treated + C(year_factor) + C(state_factor) + treated_post + female + married + educ_hs + age + age_sq',
                 data=df_analysis).fit(cov_type='HC1')
print(f"Treated x Post coefficient: {model4.params['treated_post']:.4f}")
print(f"Std Error: {model4.bse['treated_post']:.4f}")
print(f"t-stat: {model4.tvalues['treated_post']:.4f}")
print(f"p-value: {model4.pvalues['treated_post']:.4f}")

# ============================================================================
# STEP 8: Calculate DiD Estimate Manually (for verification)
# ============================================================================
print("\n[8] Manual DiD Calculation (Verification)...")

# Calculate group means
treat_pre = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 0)]['fulltime'].mean()
treat_post = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 1)]['fulltime'].mean()
control_pre = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 0)]['fulltime'].mean()
control_post = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 1)]['fulltime'].mean()

print(f"Treatment group (26-30), Pre-DACA: {treat_pre*100:.2f}%")
print(f"Treatment group (26-30), Post-DACA: {treat_post*100:.2f}%")
print(f"Control group (31-35), Pre-DACA: {control_pre*100:.2f}%")
print(f"Control group (31-35), Post-DACA: {control_post*100:.2f}%")

treat_diff = treat_post - treat_pre
control_diff = control_post - control_pre
did_estimate = treat_diff - control_diff

print(f"\nTreatment group change: {treat_diff*100:.2f} pp")
print(f"Control group change: {control_diff*100:.2f} pp")
print(f"DiD estimate: {did_estimate*100:.2f} pp")

# ============================================================================
# STEP 9: Robustness Checks
# ============================================================================
print("\n[9] Robustness Checks...")

# 9a. Weighted regression using person weights
print("\n--- Robustness 1: Weighted Regression ---")
model_weighted = smf.wls('fulltime ~ treated + C(year_factor) + treated_post + female + married + educ_hs + age + age_sq',
                         data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')
print(f"Treated x Post coefficient (weighted): {model_weighted.params['treated_post']:.4f}")
print(f"Std Error: {model_weighted.bse['treated_post']:.4f}")

# 9b. By gender
print("\n--- Robustness 2: By Gender ---")
df_male = df_analysis[df_analysis['female'] == 0]
df_female = df_analysis[df_analysis['female'] == 1]

model_male = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_male).fit(cov_type='HC1')
model_female = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_female).fit(cov_type='HC1')

print(f"Males - DiD estimate: {model_male.params['treated_post']:.4f} (SE: {model_male.bse['treated_post']:.4f})")
print(f"Females - DiD estimate: {model_female.params['treated_post']:.4f} (SE: {model_female.bse['treated_post']:.4f})")

# 9c. Different age bandwidth
print("\n--- Robustness 3: Narrower Age Bandwidth (28-30 vs 31-33) ---")
df_narrow = df[(df['BIRTHYR'] >= 1979) & (df['BIRTHYR'] <= 1984)].copy()
df_narrow['treated'] = ((df_narrow['BIRTHYR'] >= 1982) & (df_narrow['BIRTHYR'] <= 1984)).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treated_post'] = df_narrow['treated'] * df_narrow['post']
df_narrow['year_factor'] = df_narrow['YEAR'].astype(str)
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)

model_narrow = smf.ols('fulltime ~ treated + C(year_factor) + treated_post', data=df_narrow).fit(cov_type='HC1')
print(f"Narrow bandwidth - DiD estimate: {model_narrow.params['treated_post']:.4f} (SE: {model_narrow.bse['treated_post']:.4f})")
print(f"Sample size: {len(df_narrow):,}")

# ============================================================================
# STEP 10: Event Study / Pre-Trend Analysis
# ============================================================================
print("\n[10] Event Study Analysis (Pre-Trends Check)...")

# Create year interactions with treatment
df_analysis['year_2006'] = (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['year_2007'] = (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['year_2008'] = (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['year_2009'] = (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['year_2010'] = (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_analysis['year_2013'] = (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['year_2014'] = (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['year_2015'] = (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['year_2016'] = (df_analysis['YEAR'] == 2016).astype(int)

# Create interactions
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{yr}'] = df_analysis['treated'] * df_analysis[f'year_{yr}']

event_study_formula = 'fulltime ~ treated + C(year_factor) + ' + ' + '.join([f'treat_x_{yr}' for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]])
model_event = smf.ols(event_study_formula, data=df_analysis).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params[f'treat_x_{yr}']
    se = model_event.bse[f'treat_x_{yr}']
    pval = model_event.pvalues[f'treat_x_{yr}']
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {yr}: {coef:7.4f} (SE: {se:.4f}) {sig}")

# ============================================================================
# STEP 11: Save Results for Report
# ============================================================================
print("\n[11] Saving Results...")

# Create results dictionary
results = {
    'sample_counts': sample_counts,
    'n_analysis': len(df_analysis),
    'n_treatment': df_analysis['treated'].sum(),
    'n_control': (df_analysis['treated'] == 0).sum(),
    'model1_coef': model1.params['treated_post'],
    'model1_se': model1.bse['treated_post'],
    'model1_pval': model1.pvalues['treated_post'],
    'model2_coef': model2.params['treated_post'],
    'model2_se': model2.bse['treated_post'],
    'model2_pval': model2.pvalues['treated_post'],
    'model3_coef': model3.params['treated_post'],
    'model3_se': model3.bse['treated_post'],
    'model3_pval': model3.pvalues['treated_post'],
    'model4_coef': model4.params['treated_post'],
    'model4_se': model4.bse['treated_post'],
    'model4_pval': model4.pvalues['treated_post'],
    'model_weighted_coef': model_weighted.params['treated_post'],
    'model_weighted_se': model_weighted.bse['treated_post'],
    'treat_pre': treat_pre,
    'treat_post': treat_post,
    'control_pre': control_pre,
    'control_post': control_post,
    'did_manual': did_estimate
}

# Save sample by year
sample_by_year_group.to_csv('output_sample_by_year.csv')
ft_by_year_group.to_csv('output_ft_rates_by_year.csv')

# Save event study coefficients
event_coeffs = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coef': [model_event.params.get(f'treat_x_{yr}', 0) for yr in [2006, 2007, 2008, 2009, 2010]] + [0] +
            [model_event.params.get(f'treat_x_{yr}', 0) for yr in [2013, 2014, 2015, 2016]],
    'se': [model_event.bse.get(f'treat_x_{yr}', 0) for yr in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [model_event.bse.get(f'treat_x_{yr}', 0) for yr in [2013, 2014, 2015, 2016]]
})
event_coeffs.to_csv('output_event_study.csv', index=False)

# ============================================================================
# STEP 12: Summary Statistics Table
# ============================================================================
print("\n[12] Summary Statistics...")

summary_vars = ['age', 'female', 'married', 'educ_hs', 'educ_college', 'fulltime', 'UHRSWORK']
summary_pre = df_analysis[df_analysis['post'] == 0].groupby('treated')[summary_vars].mean()
summary_post = df_analysis[df_analysis['post'] == 1].groupby('treated')[summary_vars].mean()

print("\n--- Pre-Period Summary Statistics ---")
print(summary_pre.round(3))
print("\n--- Post-Period Summary Statistics ---")
print(summary_post.round(3))

# Save summary stats
summary_pre.to_csv('output_summary_pre.csv')
summary_post.to_csv('output_summary_post.csv')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"\nPreferred Specification: Model 3 (DiD with Year FE and Covariates)")
print(f"  DiD Estimate (Treated x Post): {model3.params['treated_post']:.4f}")
print(f"  Standard Error: {model3.bse['treated_post']:.4f}")
print(f"  95% Confidence Interval: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]")
print(f"  p-value: {model3.pvalues['treated_post']:.4f}")
print(f"  Sample Size: {int(model3.nobs):,}")

print(f"\nInterpretation: DACA eligibility is associated with a {model3.params['treated_post']*100:.2f} percentage point")
if model3.params['treated_post'] > 0:
    print(f"increase in the probability of full-time employment.")
else:
    print(f"change in the probability of full-time employment.")

if model3.pvalues['treated_post'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")

print("\n" + "="*80)
print("Analysis complete. Output files saved.")
print("="*80)

# Save final results to file
with open('output_results_summary.txt', 'w') as f:
    f.write("DACA REPLICATION - FINAL RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Sample Size: {int(model3.nobs):,}\n")
    f.write(f"Treatment Group (26-30): {df_analysis['treated'].sum():,}\n")
    f.write(f"Control Group (31-35): {(df_analysis['treated']==0).sum():,}\n\n")
    f.write("PREFERRED ESTIMATE (Model 3):\n")
    f.write(f"  DiD Coefficient: {model3.params['treated_post']:.4f}\n")
    f.write(f"  Standard Error: {model3.bse['treated_post']:.4f}\n")
    f.write(f"  95% CI: [{model3.conf_int().loc['treated_post', 0]:.4f}, {model3.conf_int().loc['treated_post', 1]:.4f}]\n")
    f.write(f"  p-value: {model3.pvalues['treated_post']:.4f}\n\n")
    f.write("ROBUSTNESS CHECKS:\n")
    f.write(f"  Model 1 (Basic): {model1.params['treated_post']:.4f} (SE: {model1.bse['treated_post']:.4f})\n")
    f.write(f"  Model 2 (Year FE): {model2.params['treated_post']:.4f} (SE: {model2.bse['treated_post']:.4f})\n")
    f.write(f"  Model 4 (State FE): {model4.params['treated_post']:.4f} (SE: {model4.bse['treated_post']:.4f})\n")
    f.write(f"  Weighted: {model_weighted.params['treated_post']:.4f} (SE: {model_weighted.bse['treated_post']:.4f})\n")

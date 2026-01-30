"""
DACA Replication Study: Effect on Full-Time Employment
========================================================
Research Question: Among ethnically Hispanic-Mexican, Mexican-born individuals,
what was the causal impact of DACA eligibility on full-time employment?

Treatment: Ages 26-30 on June 15, 2012 (birth years 1982-1986)
Control: Ages 31-35 on June 15, 2012 (birth years 1977-1981)
Pre-period: 2006-2011
Post-period: 2013-2016 (excluding 2012 due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Step 1: Define DACA eligibility criteria
print("\n[2] Applying sample restrictions...")

# Restriction 1: Hispanic-Mexican ethnicity (HISPAN == 1)
# From data dict: HISPAN 1 = Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"After Hispanic-Mexican restriction: {len(df_sample):,}")

# Restriction 2: Born in Mexico (BPL == 200)
# From data dict: BPL 200 = Mexico
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After Mexican birthplace restriction: {len(df_sample):,}")

# Restriction 3: Non-citizen (proxy for undocumented)
# From data dict: CITIZEN 3 = Not a citizen
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen restriction: {len(df_sample):,}")

# Restriction 4: Arrived before age 16
# Calculate age at immigration
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
# Keep only those who arrived before age 16 and have valid immigration year
df_sample = df_sample[(df_sample['YRIMMIG'] > 0) & (df_sample['age_at_immig'] < 16)].copy()
print(f"After arrived-before-16 restriction: {len(df_sample):,}")

# Restriction 5: Arrived by June 15, 2007 (use 2007 as cutoff)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"After arrived-by-2007 restriction: {len(df_sample):,}")

# Define treatment and control groups based on age on June 15, 2012
# Treatment: Ages 26-30 on 6/15/2012 -> birth years 1982-1986
# Control: Ages 31-35 on 6/15/2012 -> birth years 1977-1981

# Create age_in_2012 variable
df_sample['age_in_2012'] = 2012 - df_sample['BIRTHYR']

# Treatment group: ages 26-30 in 2012
df_sample['treat'] = ((df_sample['age_in_2012'] >= 26) &
                       (df_sample['age_in_2012'] <= 30)).astype(int)

# Control group: ages 31-35 in 2012
df_sample['control'] = ((df_sample['age_in_2012'] >= 31) &
                         (df_sample['age_in_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[df_sample['treat'] + df_sample['control'] == 1].copy()
print(f"After treatment/control group restriction: {len(df_analysis):,}")

# Define pre and post periods
# Pre: 2006-2011, Post: 2013-2016 (exclude 2012)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
print(f"After excluding 2012: {len(df_analysis):,}")

# Create full-time employment outcome (UHRSWORK >= 35)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

print("\n[3] Sample summary statistics...")
print(f"\nSample size by group and period:")
print(df_analysis.groupby(['treat', 'post']).size().unstack())

print(f"\nFull-time employment rates by group and period:")
ft_rates = df_analysis.groupby(['treat', 'post'])['fulltime'].mean().unstack()
print(ft_rates)

# Calculate simple DiD
pre_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)]['fulltime'].mean()
pre_control = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)]['fulltime'].mean()
post_control = df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)]['fulltime'].mean()

did_simple = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\n[4] Simple Difference-in-Differences:")
print(f"Treatment pre-period mean: {pre_treat:.4f}")
print(f"Treatment post-period mean: {post_treat:.4f}")
print(f"Treatment change: {post_treat - pre_treat:.4f}")
print(f"Control pre-period mean: {pre_control:.4f}")
print(f"Control post-period mean: {post_control:.4f}")
print(f"Control change: {post_control - pre_control:.4f}")
print(f"DiD estimate: {did_simple:.4f}")

# Basic DiD regression (unweighted)
print("\n[5] Regression Analysis (Unweighted)...")
model1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
print("\nModel 1: Basic DiD")
print(model1.summary().tables[1])

# DiD with robust standard errors
model1_robust = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit(cov_type='HC1')
print("\nModel 1 with Robust SE:")
print(f"DiD Estimate: {model1_robust.params['treat_post']:.4f}")
print(f"Robust SE: {model1_robust.bse['treat_post']:.4f}")
print(f"95% CI: [{model1_robust.conf_int().loc['treat_post', 0]:.4f}, {model1_robust.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model1_robust.pvalues['treat_post']:.4f}")

# Weighted regression using PERWT
print("\n[6] Weighted Regression Analysis...")
model2 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print("\nModel 2: Weighted DiD (using PERWT)")
print(f"DiD Estimate: {model2.params['treat_post']:.4f}")
print(f"Robust SE: {model2.bse['treat_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model2.pvalues['treat_post']:.4f}")

# Add covariates
print("\n[7] With Covariates...")
# Create covariate dummies
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] <= 2).astype(int)  # 1 or 2 = married

# Education categories
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # HS or higher
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 7).astype(int)  # Some college or higher

model3 = smf.wls('fulltime ~ treat + post + treat_post + female + married + educ_hs + C(STATEFIP) + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')
print("\nModel 3: With covariates and state/year FE")
print(f"DiD Estimate: {model3.params['treat_post']:.4f}")
print(f"Robust SE: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Year-by-year effects (event study)
print("\n[8] Event Study Analysis...")
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
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'treat_x_{year}'] = df_analysis['treat'] * df_analysis[f'year_{year}']

event_formula = 'fulltime ~ treat + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + treat_x_2006 + treat_x_2007 + treat_x_2008 + treat_x_2009 + treat_x_2010 + treat_x_2013 + treat_x_2014 + treat_x_2015 + treat_x_2016'

model_event = smf.wls(event_formula, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (treatment group x year interactions):")
print("Reference year: 2011")
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_x_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    print(f"  {year}: {coef:7.4f} (SE: {se:.4f}, p={pval:.4f})")

# Subgroup analysis by sex
print("\n[9] Subgroup Analysis...")
print("\nBy Sex:")
for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex]
    model_sub = smf.wls('fulltime ~ treat + post + treat_post',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"  {label}: DiD = {model_sub.params['treat_post']:.4f} (SE: {model_sub.bse['treat_post']:.4f}), N = {len(df_sub):,}")

# Robustness: Different age bands
print("\n[10] Robustness Checks...")
print("\nUsing narrower age bands (27-29 vs 32-34):")
df_narrow = df_sample[
    ((df_sample['age_in_2012'] >= 27) & (df_sample['age_in_2012'] <= 29)) |
    ((df_sample['age_in_2012'] >= 32) & (df_sample['age_in_2012'] <= 34))
].copy()
df_narrow = df_narrow[df_narrow['YEAR'] != 2012]
df_narrow['treat'] = ((df_narrow['age_in_2012'] >= 27) & (df_narrow['age_in_2012'] <= 29)).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post'] = df_narrow['treat'] * df_narrow['post']
df_narrow['fulltime'] = (df_narrow['UHRSWORK'] >= 35).astype(int)

model_narrow = smf.wls('fulltime ~ treat + post + treat_post',
                        data=df_narrow, weights=df_narrow['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model_narrow.params['treat_post']:.4f} (SE: {model_narrow.bse['treat_post']:.4f}), N = {len(df_narrow):,}")

# Placebo test using 2008-2009 as fake treatment
print("\nPlacebo test (using 2008-2009 as fake post-period on pre-DACA data):")
df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_fake'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_fake'] = df_placebo['treat'] * df_placebo['post_fake']

model_placebo = smf.wls('fulltime ~ treat + post_fake + treat_post_fake',
                         data=df_placebo, weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD: {model_placebo.params['treat_post_fake']:.4f} (SE: {model_placebo.bse['treat_post_fake']:.4f}, p={model_placebo.pvalues['treat_post_fake']:.4f})")

# Summary statistics table
print("\n" + "="*70)
print("SUMMARY OF KEY RESULTS")
print("="*70)
print(f"\nPreferred Estimate (Weighted DiD with robust SE):")
print(f"  Effect of DACA eligibility on full-time employment: {model2.params['treat_post']:.4f}")
print(f"  Standard Error: {model2.bse['treat_post']:.4f}")
print(f"  95% Confidence Interval: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"  p-value: {model2.pvalues['treat_post']:.4f}")
print(f"  Sample Size: {len(df_analysis):,}")

print(f"\nInterpretation:")
effect_pct = model2.params['treat_post'] * 100
if model2.params['treat_post'] > 0:
    print(f"  DACA eligibility is associated with a {abs(effect_pct):.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(effect_pct):.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model2.pvalues['treat_post'] < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

# Save key results for the report
results = {
    'n_total': len(df_analysis),
    'n_treat_pre': len(df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==0)]),
    'n_treat_post': len(df_analysis[(df_analysis['treat']==1) & (df_analysis['post']==1)]),
    'n_control_pre': len(df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==0)]),
    'n_control_post': len(df_analysis[(df_analysis['treat']==0) & (df_analysis['post']==1)]),
    'ft_treat_pre': pre_treat,
    'ft_treat_post': post_treat,
    'ft_control_pre': pre_control,
    'ft_control_post': post_control,
    'did_simple': did_simple,
    'did_weighted': model2.params['treat_post'],
    'se_weighted': model2.bse['treat_post'],
    'ci_low': model2.conf_int().loc['treat_post', 0],
    'ci_high': model2.conf_int().loc['treat_post', 1],
    'pval_weighted': model2.pvalues['treat_post'],
    'did_covariates': model3.params['treat_post'],
    'se_covariates': model3.bse['treat_post'],
}

# Export results
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[Results saved to results.json]")

# Additional descriptive statistics
print("\n[11] Additional Descriptive Statistics...")
print("\nSample characteristics by treatment status:")
desc_vars = ['AGE', 'female', 'married', 'educ_hs', 'UHRSWORK']
for var in desc_vars:
    treat_mean = df_analysis[df_analysis['treat']==1][var].mean()
    control_mean = df_analysis[df_analysis['treat']==0][var].mean()
    print(f"  {var}: Treat={treat_mean:.3f}, Control={control_mean:.3f}")

print("\nAnalysis complete!")

"""
DACA Replication Study - Analysis Script
Examining the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States.

This script implements a difference-in-differences estimation strategy.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

print("=" * 80)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)
print("\n1. Loading ACS data (chunked to manage memory)...")

# Define columns we need
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'MARST',
           'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
           'UHRSWORK', 'NCHILD', 'STATEFIP']

# Define dtypes to reduce memory
dtypes = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'NCHILD': 'int8',
    'STATEFIP': 'int8'
}

# Load data in chunks and filter
chunks = []
chunksize = 500000
total_rows = 0

for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes, chunksize=chunksize):
    # Apply initial filters to reduce data size
    # Hispanic-Mexican (HISPAN == 1) and Born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    total_rows += len(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"   Total rows read: {total_rows:,}")
print(f"   Hispanic-Mexican, Mexico-born: {len(df):,}")

# ============================================================================
# PART 2: SAMPLE CONSTRUCTION
# ============================================================================

print("\n2. Applying remaining eligibility criteria...")

# Criterion: Not a citizen (CITIZEN == 3)
df_sample = df[df['CITIZEN'] == 3].copy()
print(f"   a) Non-citizen: {len(df_sample):,} observations")

# Criterion: Arrived before age 16
df_sample['age_at_immig'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
df_sample = df_sample[df_sample['age_at_immig'] < 16].copy()
print(f"   b) Arrived before age 16: {len(df_sample):,} observations")

# Criterion: Lived in US since 2007 or earlier
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"   c) In US since 2007 or earlier: {len(df_sample):,} observations")

# ============================================================================
# PART 3: DEFINE TREATMENT AND CONTROL GROUPS
# ============================================================================

print("\n3. Defining treatment and control groups based on age in June 2012...")

# Calculate age in 2012
df_sample['age_in_2012'] = 2012 - df_sample['BIRTHYR']

# Treatment (26-30 in 2012): BIRTHYR 1982-1986
# Control (31-35 in 2012): BIRTHYR 1977-1981
df_sample['treatment'] = ((df_sample['age_in_2012'] >= 26) &
                          (df_sample['age_in_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_in_2012'] >= 31) &
                        (df_sample['age_in_2012'] <= 35)).astype(int)

# Keep only treatment and control groups
df_analysis = df_sample[(df_sample['treatment'] == 1) | (df_sample['control'] == 1)].copy()

print(f"   Treatment group (ages 26-30 in 2012): {df_analysis['treatment'].sum():,} observations")
print(f"   Control group (ages 31-35 in 2012): {(df_analysis['control'] == 1).sum():,} observations")

# Free memory
del df, df_sample, chunks

# ============================================================================
# PART 4: DEFINE PRE AND POST PERIODS
# ============================================================================

print("\n4. Defining pre and post treatment periods...")

# DACA was implemented on June 15, 2012
# Pre-period: 2006-2011
# Post-period: 2013-2016
# Exclude 2012 for main analysis

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# For main analysis, exclude 2012
df_main = df_analysis[df_analysis['YEAR'] != 2012].copy()

print(f"   Pre-period (2006-2011): {df_main[df_main['post'] == 0].shape[0]:,} observations")
print(f"   Post-period (2013-2016): {df_main[df_main['post'] == 1].shape[0]:,} observations")

# ============================================================================
# PART 5: DEFINE OUTCOME AND COVARIATES
# ============================================================================

print("\n5. Defining outcome and covariates...")

# Full-time employment: EMPSTAT == 1 AND UHRSWORK >= 35
df_main['fulltime_emp'] = ((df_main['EMPSTAT'] == 1) &
                           (df_main['UHRSWORK'] >= 35)).astype(int)

# Covariates
df_main['female'] = (df_main['SEX'] == 2).astype(int)
df_main['married'] = df_main['MARST'].isin([1, 2]).astype(int)
df_main['educ_hs_plus'] = (df_main['EDUC'] >= 6).astype(int)
df_main['age'] = df_main['AGE']
df_main['age_sq'] = df_main['age'] ** 2
df_main['has_children'] = (df_main['NCHILD'] > 0).astype(int)

# Interaction term
df_main['treat_post'] = df_main['treatment'] * df_main['post']

print(f"   Overall full-time employment rate: {df_main['fulltime_emp'].mean()*100:.2f}%")

# ============================================================================
# PART 6: DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

# Full-time employment rates by group and period
print("\n6. Full-time employment rates by group and period:")

summary_table = df_main.groupby(['treatment', 'post'])['fulltime_emp'].agg(['mean', 'count', 'std'])
summary_table.columns = ['FT_Emp_Rate', 'N', 'Std_Dev']
summary_table['SE'] = summary_table['Std_Dev'] / np.sqrt(summary_table['N'])

print("\n   Group definitions:")
print("   treatment=0: Control group (ages 31-35 in 2012)")
print("   treatment=1: Treatment group (ages 26-30 in 2012)")
print("   post=0: Pre-period (2006-2011)")
print("   post=1: Post-period (2013-2016)")
print()
print(summary_table.round(4))

# Simple DiD calculation
pre_treatment = df_main[(df_main['treatment'] == 1) & (df_main['post'] == 0)]['fulltime_emp'].mean()
post_treatment = df_main[(df_main['treatment'] == 1) & (df_main['post'] == 1)]['fulltime_emp'].mean()
pre_control = df_main[(df_main['treatment'] == 0) & (df_main['post'] == 0)]['fulltime_emp'].mean()
post_control = df_main[(df_main['treatment'] == 0) & (df_main['post'] == 1)]['fulltime_emp'].mean()

simple_did = (post_treatment - pre_treatment) - (post_control - pre_control)

print(f"\n   Simple DiD calculation:")
print(f"   Treatment: {pre_treatment:.4f} -> {post_treatment:.4f} (change: {post_treatment - pre_treatment:.4f})")
print(f"   Control:   {pre_control:.4f} -> {post_control:.4f} (change: {post_control - pre_control:.4f})")
print(f"   DiD estimate: {simple_did:.4f}")

# Covariate balance
print("\n   Covariate means by treatment group (pre-period):")
cov_balance = df_main[df_main['post'] == 0].groupby('treatment')[['female', 'married', 'educ_hs_plus', 'age']].mean()
print(cov_balance.round(4))

# ============================================================================
# PART 7: REGRESSION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("REGRESSION ANALYSIS")
print("=" * 80)

print("\n7. Main DiD Regressions:")

# Model 1: Basic DiD
print("\n   Model 1: Basic DiD")
model1 = smf.ols('fulltime_emp ~ treatment + post + treat_post', data=df_main).fit(cov_type='HC1')
print(f"   DiD coefficient: {model1.params['treat_post']:.4f}")
print(f"   Robust SE: {model1.bse['treat_post']:.4f}")
print(f"   t-stat: {model1.tvalues['treat_post']:.4f}")
print(f"   p-value: {model1.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['treat_post', 0]:.4f}, {model1.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   N: {int(model1.nobs):,}")

results = {
    'model1': {
        'coef': model1.params['treat_post'],
        'se': model1.bse['treat_post'],
        'pvalue': model1.pvalues['treat_post'],
        'ci_low': model1.conf_int().loc['treat_post', 0],
        'ci_high': model1.conf_int().loc['treat_post', 1],
        'n': int(model1.nobs),
        'r2': model1.rsquared
    }
}

# Model 2: DiD with covariates
print("\n   Model 2: DiD with demographic covariates")
model2 = smf.ols('fulltime_emp ~ treatment + post + treat_post + female + married + educ_hs_plus + age + age_sq + has_children',
                 data=df_main).fit(cov_type='HC1')
print(f"   DiD coefficient: {model2.params['treat_post']:.4f}")
print(f"   Robust SE: {model2.bse['treat_post']:.4f}")
print(f"   p-value: {model2.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['treat_post', 0]:.4f}, {model2.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   N: {int(model2.nobs):,}")

results['model2'] = {
    'coef': model2.params['treat_post'],
    'se': model2.bse['treat_post'],
    'pvalue': model2.pvalues['treat_post'],
    'ci_low': model2.conf_int().loc['treat_post', 0],
    'ci_high': model2.conf_int().loc['treat_post', 1],
    'n': int(model2.nobs),
    'r2': model2.rsquared
}

# Model 3: DiD with year and state FE
print("\n   Model 3: DiD with year and state fixed effects")
df_main['year_str'] = df_main['YEAR'].astype(str)
df_main['state_str'] = df_main['STATEFIP'].astype(str)

model3 = smf.ols('fulltime_emp ~ treatment + treat_post + female + married + educ_hs_plus + age + age_sq + has_children + C(year_str) + C(state_str)',
                 data=df_main).fit(cov_type='HC1')
print(f"   DiD coefficient: {model3.params['treat_post']:.4f}")
print(f"   Robust SE: {model3.bse['treat_post']:.4f}")
print(f"   p-value: {model3.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")
print(f"   N: {int(model3.nobs):,}")

results['model3'] = {
    'coef': model3.params['treat_post'],
    'se': model3.bse['treat_post'],
    'pvalue': model3.pvalues['treat_post'],
    'ci_low': model3.conf_int().loc['treat_post', 0],
    'ci_high': model3.conf_int().loc['treat_post', 1],
    'n': int(model3.nobs),
    'r2': model3.rsquared
}

# Model 4: Weighted regression
print("\n   Model 4: Weighted DiD with fixed effects")
model4 = smf.wls('fulltime_emp ~ treatment + treat_post + female + married + educ_hs_plus + age + age_sq + has_children + C(year_str) + C(state_str)',
                 data=df_main, weights=df_main['PERWT']).fit(cov_type='HC1')
print(f"   DiD coefficient: {model4.params['treat_post']:.4f}")
print(f"   Robust SE: {model4.bse['treat_post']:.4f}")
print(f"   p-value: {model4.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

results['model4'] = {
    'coef': model4.params['treat_post'],
    'se': model4.bse['treat_post'],
    'pvalue': model4.pvalues['treat_post'],
    'ci_low': model4.conf_int().loc['treat_post', 0],
    'ci_high': model4.conf_int().loc['treat_post', 1],
    'n': int(model4.nobs),
    'r2': model4.rsquared
}

# ============================================================================
# PART 8: ROBUSTNESS CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 8a. Placebo test
print("\n8a. Placebo test (fake treatment in 2009, pre-DACA only):")
df_placebo = df_main[df_main['YEAR'] <= 2011].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_fake_post'] = df_placebo['treatment'] * df_placebo['fake_post']

model_placebo = smf.ols('fulltime_emp ~ treatment + fake_post + treat_fake_post + female + married + educ_hs_plus',
                        data=df_placebo).fit(cov_type='HC1')
print(f"   Placebo DiD coefficient: {model_placebo.params['treat_fake_post']:.4f}")
print(f"   Robust SE: {model_placebo.bse['treat_fake_post']:.4f}")
print(f"   p-value: {model_placebo.pvalues['treat_fake_post']:.4f}")

# 8b. Narrower age bandwidth
print("\n8b. Narrower age bandwidth (27-29 vs 32-34):")
df_narrow = df_analysis[
    ((df_analysis['age_in_2012'] >= 27) & (df_analysis['age_in_2012'] <= 29)) |
    ((df_analysis['age_in_2012'] >= 32) & (df_analysis['age_in_2012'] <= 34))
].copy()
df_narrow = df_narrow[df_narrow['YEAR'] != 2012].copy()
df_narrow['treatment_narrow'] = ((df_narrow['age_in_2012'] >= 27) &
                                  (df_narrow['age_in_2012'] <= 29)).astype(int)
df_narrow['post'] = (df_narrow['YEAR'] >= 2013).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treatment_narrow'] * df_narrow['post']
df_narrow['fulltime_emp'] = ((df_narrow['EMPSTAT'] == 1) &
                              (df_narrow['UHRSWORK'] >= 35)).astype(int)
df_narrow['female'] = (df_narrow['SEX'] == 2).astype(int)
df_narrow['married'] = df_narrow['MARST'].isin([1, 2]).astype(int)
df_narrow['educ_hs_plus'] = (df_narrow['EDUC'] >= 6).astype(int)

model_narrow = smf.ols('fulltime_emp ~ treatment_narrow + post + treat_post_narrow + female + married + educ_hs_plus',
                       data=df_narrow).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_narrow.params['treat_post_narrow']:.4f}")
print(f"   Robust SE: {model_narrow.bse['treat_post_narrow']:.4f}")
print(f"   p-value: {model_narrow.pvalues['treat_post_narrow']:.4f}")
print(f"   N: {int(model_narrow.nobs):,}")

# 8c. Including 2012
print("\n8c. Including 2012 in post-period:")
df_with_2012 = df_analysis.copy()
df_with_2012['post_2012'] = (df_with_2012['YEAR'] >= 2012).astype(int)
df_with_2012['treat_post_2012'] = df_with_2012['treatment'] * df_with_2012['post_2012']
df_with_2012['fulltime_emp'] = ((df_with_2012['EMPSTAT'] == 1) &
                                 (df_with_2012['UHRSWORK'] >= 35)).astype(int)
df_with_2012['female'] = (df_with_2012['SEX'] == 2).astype(int)
df_with_2012['married'] = df_with_2012['MARST'].isin([1, 2]).astype(int)
df_with_2012['educ_hs_plus'] = (df_with_2012['EDUC'] >= 6).astype(int)

model_2012 = smf.ols('fulltime_emp ~ treatment + post_2012 + treat_post_2012 + female + married + educ_hs_plus',
                     data=df_with_2012).fit(cov_type='HC1')
print(f"   DiD coefficient: {model_2012.params['treat_post_2012']:.4f}")
print(f"   Robust SE: {model_2012.bse['treat_post_2012']:.4f}")
print(f"   p-value: {model_2012.pvalues['treat_post_2012']:.4f}")

# ============================================================================
# PART 9: HETEROGENEITY ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("HETEROGENEITY ANALYSIS")
print("=" * 80)

print("\n9a. By sex:")
df_male = df_main[df_main['SEX'] == 1].copy()
model_male = smf.ols('fulltime_emp ~ treatment + post + treat_post + married + educ_hs_plus',
                     data=df_male).fit(cov_type='HC1')
print(f"   Males: DiD = {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f}, p={model_male.pvalues['treat_post']:.4f})")

df_female = df_main[df_main['SEX'] == 2].copy()
model_female = smf.ols('fulltime_emp ~ treatment + post + treat_post + married + educ_hs_plus',
                       data=df_female).fit(cov_type='HC1')
print(f"   Females: DiD = {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f}, p={model_female.pvalues['treat_post']:.4f})")

print("\n9b. By education:")
df_loweduc = df_main[df_main['educ_hs_plus'] == 0].copy()
model_loweduc = smf.ols('fulltime_emp ~ treatment + post + treat_post + female + married',
                        data=df_loweduc).fit(cov_type='HC1')
print(f"   Less than HS: DiD = {model_loweduc.params['treat_post']:.4f} (SE: {model_loweduc.bse['treat_post']:.4f})")

df_higheduc = df_main[df_main['educ_hs_plus'] == 1].copy()
model_higheduc = smf.ols('fulltime_emp ~ treatment + post + treat_post + female + married',
                         data=df_higheduc).fit(cov_type='HC1')
print(f"   HS or more: DiD = {model_higheduc.params['treat_post']:.4f} (SE: {model_higheduc.bse['treat_post']:.4f})")

# ============================================================================
# PART 10: EVENT STUDY
# ============================================================================

print("\n" + "=" * 80)
print("EVENT STUDY ANALYSIS")
print("=" * 80)

print("\n10. Year-by-year treatment effects (reference: 2011):")

# Year dummies
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_main[f'year_{yr}'] = (df_main['YEAR'] == yr).astype(int)
    df_main[f'treat_yr_{yr}'] = df_main['treatment'] * df_main[f'year_{yr}']

model_event = smf.ols('''fulltime_emp ~ treatment +
                      year_2006 + year_2007 + year_2008 + year_2009 + year_2010 +
                      year_2013 + year_2014 + year_2015 + year_2016 +
                      treat_yr_2006 + treat_yr_2007 + treat_yr_2008 + treat_yr_2009 + treat_yr_2010 +
                      treat_yr_2013 + treat_yr_2014 + treat_yr_2015 + treat_yr_2016 +
                      female + married + educ_hs_plus''',
                      data=df_main).fit(cov_type='HC1')

print("\n   Year    Coef       SE        p-value")
print("   " + "-" * 45)
for yr in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    var = f'treat_yr_{yr}'
    print(f"   {yr}    {model_event.params[var]:8.4f}   {model_event.bse[var]:.4f}    {model_event.pvalues[var]:.4f}")

# ============================================================================
# PART 11: SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY OF MAIN RESULTS")
print("=" * 80)

print("\n   PREFERRED SPECIFICATION: Model 3 (DiD with covariates + year & state FE)")
print(f"\n   Effect of DACA eligibility on full-time employment:")
print(f"   Point estimate: {results['model3']['coef']:.4f}")
print(f"   Robust standard error: {results['model3']['se']:.4f}")
print(f"   95% CI: [{results['model3']['ci_low']:.4f}, {results['model3']['ci_high']:.4f}]")
print(f"   p-value: {results['model3']['pvalue']:.4f}")
print(f"   Sample size: {results['model3']['n']:,}")

if results['model3']['pvalue'] < 0.05:
    direction = "increased" if results['model3']['coef'] > 0 else "decreased"
    print(f"\n   Interpretation: DACA eligibility {direction} the probability of full-time")
    print(f"   employment by {abs(results['model3']['coef'])*100:.2f} percentage points (p<0.05).")
else:
    print(f"\n   Interpretation: No statistically significant effect of DACA eligibility")
    print(f"   on full-time employment probability (p={results['model3']['pvalue']:.3f}).")

# ============================================================================
# PART 12: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save summary statistics
summary_stats = df_main.groupby(['treatment', 'post']).agg({
    'fulltime_emp': ['mean', 'std', 'count'],
    'female': 'mean',
    'married': 'mean',
    'educ_hs_plus': 'mean',
    'age': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("\n   Saved summary_statistics.csv")

# Save regression results
reg_results = pd.DataFrame({
    'Model': ['Basic DiD', 'DiD + Covariates', 'DiD + FE', 'Weighted DiD + FE'],
    'Coefficient': [results['model1']['coef'], results['model2']['coef'],
                   results['model3']['coef'], results['model4']['coef']],
    'Std_Error': [results['model1']['se'], results['model2']['se'],
                 results['model3']['se'], results['model4']['se']],
    'P_Value': [results['model1']['pvalue'], results['model2']['pvalue'],
               results['model3']['pvalue'], results['model4']['pvalue']],
    'CI_Lower': [results['model1']['ci_low'], results['model2']['ci_low'],
                results['model3']['ci_low'], results['model4']['ci_low']],
    'CI_Upper': [results['model1']['ci_high'], results['model2']['ci_high'],
                results['model3']['ci_high'], results['model4']['ci_high']],
    'N': [results['model1']['n'], results['model2']['n'],
          results['model3']['n'], results['model4']['n']],
    'R2': [results['model1']['r2'], results['model2']['r2'],
           results['model3']['r2'], results['model4']['r2']]
})
reg_results.to_csv('regression_results.csv', index=False)
print("   Saved regression_results.csv")

# Save event study results
event_study_results = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params.get(f'treat_yr_{yr}', 0) if yr != 2011 else 0
                   for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'Std_Error': [model_event.bse.get(f'treat_yr_{yr}', 0) if yr != 2011 else 0
                 for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]],
    'P_Value': [model_event.pvalues.get(f'treat_yr_{yr}', np.nan) if yr != 2011 else np.nan
               for yr in [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]]
})
event_study_results.to_csv('event_study_results.csv', index=False)
print("   Saved event_study_results.csv")

# Save sample by year
sample_by_year = df_main.groupby(['YEAR', 'treatment']).size().unstack(fill_value=0)
sample_by_year.columns = ['Control', 'Treatment']
sample_by_year.to_csv('sample_by_year.csv')
print("   Saved sample_by_year.csv")

# Save heterogeneity results
hetero_results = pd.DataFrame({
    'Subgroup': ['Males', 'Females', 'Less than HS', 'HS or more'],
    'Coefficient': [model_male.params['treat_post'], model_female.params['treat_post'],
                   model_loweduc.params['treat_post'], model_higheduc.params['treat_post']],
    'Std_Error': [model_male.bse['treat_post'], model_female.bse['treat_post'],
                 model_loweduc.bse['treat_post'], model_higheduc.bse['treat_post']],
    'P_Value': [model_male.pvalues['treat_post'], model_female.pvalues['treat_post'],
               model_loweduc.pvalues['treat_post'], model_higheduc.pvalues['treat_post']],
    'N': [int(model_male.nobs), int(model_female.nobs),
          int(model_loweduc.nobs), int(model_higheduc.nobs)]
})
hetero_results.to_csv('heterogeneity_results.csv', index=False)
print("   Saved heterogeneity_results.csv")

# Save robustness results
robust_results = pd.DataFrame({
    'Specification': ['Placebo (fake 2009)', 'Narrow bandwidth', 'Include 2012'],
    'Coefficient': [model_placebo.params['treat_fake_post'],
                   model_narrow.params['treat_post_narrow'],
                   model_2012.params['treat_post_2012']],
    'Std_Error': [model_placebo.bse['treat_fake_post'],
                 model_narrow.bse['treat_post_narrow'],
                 model_2012.bse['treat_post_2012']],
    'P_Value': [model_placebo.pvalues['treat_fake_post'],
               model_narrow.pvalues['treat_post_narrow'],
               model_2012.pvalues['treat_post_2012']]
})
robust_results.to_csv('robustness_results.csv', index=False)
print("   Saved robustness_results.csv")

# Save covariate coefficients from model 3
cov_results = pd.DataFrame({
    'Variable': ['treatment', 'treat_post', 'female', 'married', 'educ_hs_plus', 'age', 'age_sq', 'has_children'],
    'Coefficient': [model3.params.get(v, np.nan) for v in ['treatment', 'treat_post', 'female', 'married', 'educ_hs_plus', 'age', 'age_sq', 'has_children']],
    'Std_Error': [model3.bse.get(v, np.nan) for v in ['treatment', 'treat_post', 'female', 'married', 'educ_hs_plus', 'age', 'age_sq', 'has_children']],
    'P_Value': [model3.pvalues.get(v, np.nan) for v in ['treatment', 'treat_post', 'female', 'married', 'educ_hs_plus', 'age', 'age_sq', 'has_children']]
})
cov_results.to_csv('covariate_results.csv', index=False)
print("   Saved covariate_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

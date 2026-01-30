"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals

Research Design: Difference-in-Differences
- Treatment: Ages 26-30 at DACA implementation (June 15, 2012)
- Control: Ages 31-35 at DACA implementation
- Outcome: Full-time employment (35+ hours/week)
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

# -----------------------------------------------------------------------------
# STEP 1: Load and preprocess data
# -----------------------------------------------------------------------------
print("\n[1] Loading data...")

# Read data in chunks due to large file size
chunks = []
chunksize = 500000

# Define dtypes to reduce memory
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'UHRSWORK': 'int8',
    'PERWT': 'float32',
    'INCTOT': 'float32',
    'INCWAGE': 'float32'
}

# Filter conditions while reading to reduce memory
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, dtype=dtypes,
                         usecols=['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR',
                                  'MARST', 'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL',
                                  'BPLD', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EDUCD',
                                  'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK',
                                  'PERWT', 'INCTOT', 'INCWAGE']):
    # Filter for Hispanic-Mexican, Mexican-born, non-citizens
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 3) & # Not a citizen
        (chunk['YEAR'] != 2012)   # Exclude 2012 (ambiguous timing)
    ]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} observations (Hispanic-Mexican, Mexican-born, non-citizens)")

# -----------------------------------------------------------------------------
# STEP 2: Define treatment and control groups based on age at DACA
# -----------------------------------------------------------------------------
print("\n[2] Defining treatment and control groups...")

# DACA implemented June 15, 2012
# Treatment: Ages 26-30 at implementation (born 1982-1986)
# Control: Ages 31-35 at implementation (born 1977-1981)

# Calculate approximate age at June 15, 2012 based on birth year
df['age_at_daca'] = 2012 - df['BIRTHYR']

# Define treatment and control groups
df['treatment'] = ((df['BIRTHYR'] >= 1982) & (df['BIRTHYR'] <= 1986)).astype(int)
df['control'] = ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1981)).astype(int)

# Keep only treatment and control groups
df = df[(df['treatment'] == 1) | (df['control'] == 1)].copy()
print(f"After age restriction: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 3: Apply additional DACA eligibility criteria
# -----------------------------------------------------------------------------
print("\n[3] Applying DACA eligibility criteria...")

# DACA requirements:
# 1. Arrived in US before 16th birthday
# 2. Present in US since June 15, 2007 (so YRIMMIG <= 2007)
# 3. Not yet 31 as of June 15, 2012 (already handled by birth year)

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Filter for those who arrived before age 16 and by 2007
df = df[
    (df['YRIMMIG'] > 0) &           # Has immigration year
    (df['age_at_immig'] < 16) &     # Arrived before 16
    (df['YRIMMIG'] <= 2007)         # Present since June 2007
].copy()

print(f"After DACA eligibility criteria: {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 4: Define outcome and time variables
# -----------------------------------------------------------------------------
print("\n[4] Creating outcome and time variables...")

# Full-time employment: 35+ hours per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['treat_post'] = df['treatment'] * df['post']

# Working-age restriction (reasonable labor force participation)
# Keep ages 18-65 in each survey year
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 65)].copy()
print(f"After working-age restriction (18-65): {len(df):,} observations")

# -----------------------------------------------------------------------------
# STEP 5: Create control variables
# -----------------------------------------------------------------------------
print("\n[5] Creating control variables...")

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'] <= 2).astype(int)  # Married spouse present/absent

# Education categories
df['educ_lesshs'] = (df['EDUC'] < 6).astype(int)     # Less than HS
df['educ_hs'] = (df['EDUC'] == 6).astype(int)         # High school
df['educ_somecoll'] = ((df['EDUC'] >= 7) & (df['EDUC'] <= 9)).astype(int)  # Some college
df['educ_college'] = (df['EDUC'] >= 10).astype(int)   # College+

# Age squared for non-linear age effects
df['age_sq'] = df['AGE'] ** 2

# Years in US
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

print("Control variables created successfully")

# -----------------------------------------------------------------------------
# STEP 6: Summary Statistics
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# By treatment/control and pre/post
groups = df.groupby(['treatment', 'post'])

print("\n--- Sample Sizes by Group ---")
print(groups.size().unstack())

print("\n--- Full-Time Employment Rate by Group ---")
print(groups['fulltime'].mean().unstack().round(4))

print("\n--- Weighted Full-Time Employment Rate by Group ---")
def weighted_mean(group, var, weight):
    return np.average(group[var], weights=group[weight])

for t in [0, 1]:
    for p in [0, 1]:
        subset = df[(df['treatment'] == t) & (df['post'] == p)]
        wt_mean = weighted_mean(subset, 'fulltime', 'PERWT')
        label = f"{'Treatment' if t else 'Control'} - {'Post' if p else 'Pre'}"
        print(f"{label}: {wt_mean:.4f}")

# Descriptive statistics
print("\n--- Descriptive Statistics (Full Sample) ---")
desc_vars = ['fulltime', 'AGE', 'female', 'married', 'educ_lesshs', 'educ_hs',
             'educ_somecoll', 'educ_college', 'years_in_us']
print(df[desc_vars].describe().round(3))

# By treatment status
print("\n--- Pre-Treatment Characteristics (Balance Check) ---")
pre_df = df[df['post'] == 0]
balance = pre_df.groupby('treatment')[desc_vars].mean()
balance.index = ['Control', 'Treatment']
print(balance.round(4).T)

# -----------------------------------------------------------------------------
# STEP 7: Main DiD Regression Analysis
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('fulltime ~ treatment + post + treat_post', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.ols('fulltime ~ treatment + post + treat_post + female + married + AGE + age_sq',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n--- Model 3: DiD with Demographic + Education Controls ---")
model3 = smf.ols('fulltime ~ treatment + post + treat_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with Year Fixed Effects ---")
df['year_fe'] = df['YEAR'].astype('category')
model4 = smf.ols('fulltime ~ treatment + treat_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college + C(YEAR)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Print only key coefficients
print("\nKey coefficient (treat_post):")
print(f"Coefficient: {model4.params['treat_post']:.4f}")
print(f"Std Error: {model4.bse['treat_post']:.4f}")
print(f"t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with State and Year Fixed Effects ---")
model5 = smf.ols('fulltime ~ treatment + treat_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college + C(YEAR) + C(STATEFIP)',
                 data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nKey coefficient (treat_post):")
print(f"Coefficient: {model5.params['treat_post']:.4f}")
print(f"Std Error: {model5.bse['treat_post']:.4f}")
print(f"t-statistic: {model5.tvalues['treat_post']:.4f}")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# STEP 8: Weighted Regression
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("WEIGHTED REGRESSION ANALYSIS")
print("="*80)

print("\n--- Model 6: Weighted DiD with Full Controls ---")

# Use formula API for weighted regression
model6 = smf.wls('fulltime ~ treatment + post + treat_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nKey coefficient (treat_post):")
print(f"Coefficient: {model6.params['treat_post']:.4f}")
print(f"Std Error: {model6.bse['treat_post']:.4f}")
print(f"t-statistic: {model6.tvalues['treat_post']:.4f}")
print(f"p-value: {model6.pvalues['treat_post']:.4f}")
ci = model6.conf_int().loc['treat_post']
print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

# -----------------------------------------------------------------------------
# STEP 9: Pre-Trends Analysis
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("PRE-TRENDS ANALYSIS (PARALLEL TRENDS CHECK)")
print("="*80)

# Calculate yearly means by treatment group
yearly_means = df.groupby(['YEAR', 'treatment']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control', 'Treatment']
print("\n--- Weighted Full-Time Employment by Year and Group ---")
print(yearly_means.round(4))

# Difference between groups by year
yearly_means['Difference'] = yearly_means['Treatment'] - yearly_means['Control']
print("\n--- Treatment - Control Difference by Year ---")
print(yearly_means['Difference'].round(4))

# Event study specification
print("\n--- Event Study Regression ---")
# Create year interactions with treatment (relative to 2011)
df['year_2006'] = ((df['YEAR'] == 2006) & (df['treatment'] == 1)).astype(int)
df['year_2007'] = ((df['YEAR'] == 2007) & (df['treatment'] == 1)).astype(int)
df['year_2008'] = ((df['YEAR'] == 2008) & (df['treatment'] == 1)).astype(int)
df['year_2009'] = ((df['YEAR'] == 2009) & (df['treatment'] == 1)).astype(int)
df['year_2010'] = ((df['YEAR'] == 2010) & (df['treatment'] == 1)).astype(int)
# 2011 is reference year (omitted)
df['year_2013'] = ((df['YEAR'] == 2013) & (df['treatment'] == 1)).astype(int)
df['year_2014'] = ((df['YEAR'] == 2014) & (df['treatment'] == 1)).astype(int)
df['year_2015'] = ((df['YEAR'] == 2015) & (df['treatment'] == 1)).astype(int)
df['year_2016'] = ((df['YEAR'] == 2016) & (df['treatment'] == 1)).astype(int)

event_model = smf.ols('fulltime ~ treatment + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college + C(YEAR)',
                      data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

event_vars = ['year_2006', 'year_2007', 'year_2008', 'year_2009', 'year_2010',
              'year_2013', 'year_2014', 'year_2015', 'year_2016']
print("\nEvent Study Coefficients (Treatment x Year, relative to 2011):")
for var in event_vars:
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    print(f"{var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# -----------------------------------------------------------------------------
# STEP 10: Robustness Checks
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative age bandwidth (24-32 vs 33-37)
print("\n--- Robustness 1: Narrower Age Bandwidth ---")
df_narrow = df[
    ((df['BIRTHYR'] >= 1980) & (df['BIRTHYR'] <= 1986)) |  # Treatment-ish
    ((df['BIRTHYR'] >= 1977) & (df['BIRTHYR'] <= 1983))    # Control-ish
].copy()
df_narrow['treatment_narrow'] = ((df_narrow['BIRTHYR'] >= 1983) & (df_narrow['BIRTHYR'] <= 1986)).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treatment_narrow'] * df_narrow['post']

model_narrow = smf.ols('fulltime ~ treatment_narrow + post + treat_post_narrow + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college',
                       data=df_narrow).fit(cov_type='cluster', cov_kwds={'groups': df_narrow['STATEFIP']})
print(f"Coefficient: {model_narrow.params['treat_post_narrow']:.4f}")
print(f"Std Error: {model_narrow.bse['treat_post_narrow']:.4f}")
print(f"p-value: {model_narrow.pvalues['treat_post_narrow']:.4f}")

# Robustness 2: By gender
print("\n--- Robustness 2: By Gender ---")
for sex, label in [(0, 'Male'), (1, 'Female')]:
    df_sex = df[df['female'] == sex]
    model_sex = smf.ols('fulltime ~ treatment + post + treat_post + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college',
                        data=df_sex).fit(cov_type='cluster', cov_kwds={'groups': df_sex['STATEFIP']})
    print(f"{label}: Coef={model_sex.params['treat_post']:.4f}, SE={model_sex.bse['treat_post']:.4f}, p={model_sex.pvalues['treat_post']:.4f}")

# Robustness 3: Placebo test using citizens
print("\n--- Robustness 3: Placebo Test (Citizens) ---")
# Re-load citizen data for placebo
chunks_citizen = []
for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, dtype=dtypes,
                         usecols=['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR',
                                  'MARST', 'BIRTHYR', 'HISPAN', 'BPL',
                                  'CITIZEN', 'YRIMMIG', 'EDUC',
                                  'UHRSWORK', 'PERWT']):
    filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200) &   # Born in Mexico
        (chunk['CITIZEN'] == 2) & # Naturalized citizen (should not be affected)
        (chunk['YEAR'] != 2012) &
        ((chunk['BIRTHYR'] >= 1977) & (chunk['BIRTHYR'] <= 1986))
    ]
    chunks_citizen.append(filtered)

df_placebo = pd.concat(chunks_citizen, ignore_index=True)
df_placebo['treatment'] = ((df_placebo['BIRTHYR'] >= 1982) & (df_placebo['BIRTHYR'] <= 1986)).astype(int)
df_placebo['post'] = (df_placebo['YEAR'] >= 2013).astype(int)
df_placebo['treat_post'] = df_placebo['treatment'] * df_placebo['post']
df_placebo['fulltime'] = (df_placebo['UHRSWORK'] >= 35).astype(int)
df_placebo['female'] = (df_placebo['SEX'] == 2).astype(int)
df_placebo['married'] = (df_placebo['MARST'] <= 2).astype(int)
df_placebo['age_sq'] = df_placebo['AGE'] ** 2
df_placebo['educ_hs'] = (df_placebo['EDUC'] == 6).astype(int)
df_placebo['educ_somecoll'] = ((df_placebo['EDUC'] >= 7) & (df_placebo['EDUC'] <= 9)).astype(int)
df_placebo['educ_college'] = (df_placebo['EDUC'] >= 10).astype(int)
df_placebo = df_placebo[(df_placebo['AGE'] >= 18) & (df_placebo['AGE'] <= 65)]

if len(df_placebo) > 100:
    model_placebo = smf.ols('fulltime ~ treatment + post + treat_post + female + married + AGE + age_sq + educ_hs + educ_somecoll + educ_college',
                            data=df_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['STATEFIP']})
    print(f"N = {len(df_placebo)}")
    print(f"Coefficient: {model_placebo.params['treat_post']:.4f}")
    print(f"Std Error: {model_placebo.bse['treat_post']:.4f}")
    print(f"p-value: {model_placebo.pvalues['treat_post']:.4f}")
else:
    print("Insufficient sample size for placebo test")

# -----------------------------------------------------------------------------
# STEP 11: Save Results for Report
# -----------------------------------------------------------------------------
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)

# Calculate simple DiD estimate manually
pre_treat = df[(df['treatment'] == 1) & (df['post'] == 0)]['fulltime'].mean()
post_treat = df[(df['treatment'] == 1) & (df['post'] == 1)]['fulltime'].mean()
pre_control = df[(df['treatment'] == 0) & (df['post'] == 0)]['fulltime'].mean()
post_control = df[(df['treatment'] == 0) & (df['post'] == 1)]['fulltime'].mean()

did_estimate = (post_treat - pre_treat) - (post_control - pre_control)

print(f"\n--- Simple DiD Calculation ---")
print(f"Treatment Pre:  {pre_treat:.4f}")
print(f"Treatment Post: {post_treat:.4f}")
print(f"Treatment Change: {post_treat - pre_treat:.4f}")
print(f"Control Pre:    {pre_control:.4f}")
print(f"Control Post:   {post_control:.4f}")
print(f"Control Change: {post_control - pre_control:.4f}")
print(f"\nDiD Estimate: {did_estimate:.4f}")

print(f"\n--- Preferred Specification (Model 5 with State and Year FE) ---")
print(f"Effect Size: {model5.params['treat_post']:.4f}")
print(f"Standard Error: {model5.bse['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.4f}, {model5.conf_int().loc['treat_post', 1]:.4f}]")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")
print(f"Sample Size: {len(df)}")

# Save key results to file
results_dict = {
    'preferred_estimate': model5.params['treat_post'],
    'se': model5.bse['treat_post'],
    'ci_lower': model5.conf_int().loc['treat_post', 0],
    'ci_upper': model5.conf_int().loc['treat_post', 1],
    'pvalue': model5.pvalues['treat_post'],
    'n_obs': len(df),
    'n_treatment': len(df[df['treatment'] == 1]),
    'n_control': len(df[df['treatment'] == 0]),
    'pre_treat_mean': pre_treat,
    'post_treat_mean': post_treat,
    'pre_control_mean': pre_control,
    'post_control_mean': post_control
}

# Save to CSV for LaTeX
results_df = pd.DataFrame([results_dict])
results_df.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save yearly means for plotting
yearly_means.to_csv('yearly_means.csv')
print("Yearly means saved to yearly_means.csv")

# Save model comparison table
model_comparison = pd.DataFrame({
    'Model': ['Basic DiD', 'Demographics', 'Demographics + Education', 'Year FE', 'State + Year FE', 'Weighted'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                   model3.params['treat_post'], model4.params['treat_post'],
                   model5.params['treat_post'], model6.params['treat_post']],
    'SE': [model1.bse['treat_post'], model2.bse['treat_post'],
           model3.bse['treat_post'], model4.bse['treat_post'],
           model5.bse['treat_post'], model6.bse['treat_post']],
    'p-value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
                model3.pvalues['treat_post'], model4.pvalues['treat_post'],
                model5.pvalues['treat_post'], model6.pvalues['treat_post']]
})
model_comparison.to_csv('model_comparison.csv', index=False)
print("Model comparison saved to model_comparison.csv")

# Save descriptive statistics
desc_by_group = df.groupby(['treatment', 'post'])[desc_vars].mean()
desc_by_group.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

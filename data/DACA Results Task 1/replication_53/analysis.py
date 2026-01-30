"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on full-time employment (35+ hours/week)?

Identification Strategy: Difference-in-Differences
- Treatment group: DACA-eligible Hispanic-Mexican Mexican-born non-citizens
- Control group: Similar population but ineligible due to age criteria
- Pre-period: 2006-2011 (before DACA implementation on June 15, 2012)
- Post-period: 2013-2016 (after DACA implementation)
- 2012 is excluded as it's a transition year

DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Under 31 years old on June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (at least 5 years by 2012)
4. Present in US on June 15, 2012 and not a citizen
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("DACA REPLICATION ANALYSIS")
print("=" * 70)

# Load data
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Check column names
print(f"\nColumns: {list(df.columns)}")

# Filter to Hispanic-Mexican (HISPAN==1) and born in Mexico (BPL==200)
print("\n[2] Filtering to Hispanic-Mexican Mexican-born population...")
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
print(f"Hispanic-Mexican Mexican-born observations: {len(df_mex):,}")

# Filter to non-citizens (CITIZEN == 3 means "Not a citizen")
print("\n[3] Filtering to non-citizens...")
df_mex_nc = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"Non-citizen observations: {len(df_mex_nc):,}")

# Examine year distribution
print("\nYear distribution:")
print(df_mex_nc['YEAR'].value_counts().sort_index())

# Define the outcome variable: Full-time employment (35+ hours per week)
print("\n[4] Creating outcome variable: Full-time employment (UHRSWORK >= 35)...")
# UHRSWORK is usual hours worked per week
# 00 = N/A, meaning the person doesn't work
df_mex_nc['fulltime'] = (df_mex_nc['UHRSWORK'] >= 35).astype(int)
print(f"Full-time employment rate: {df_mex_nc['fulltime'].mean():.3f}")

# Create age variable for each survey year
# Age is reported at time of survey
print("\n[5] Constructing DACA eligibility criteria...")

# For DACA eligibility, we need to determine:
# 1. Age on June 15, 2012: person must be under 31
# 2. Age at arrival: must have arrived before 16th birthday
# 3. Continuous residence since June 15, 2007 (5 years before 2012)

# Calculate age as of June 15, 2012
# BIRTHYR gives birth year; we approximate age on June 15, 2012
df_mex_nc['age_2012'] = 2012 - df_mex_nc['BIRTHYR']

# Adjust for birth quarter (BIRTHQTR)
# 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec
# If born after June (Q3, Q4), they would be one year younger on June 15
df_mex_nc['age_2012_adjusted'] = df_mex_nc['age_2012'].copy()
# For those born in Q3 or Q4, they haven't had their birthday yet by June 15
df_mex_nc.loc[df_mex_nc['BIRTHQTR'].isin([3, 4]), 'age_2012_adjusted'] = df_mex_nc['age_2012'] - 1

# DACA age requirement: under 31 on June 15, 2012 (born after June 15, 1981)
# Using birthyear: born 1982 or later definitely qualifies
# Born 1981: depends on birth month - Q1/Q2 would be 31+, Q3/Q4 would be 30
df_mex_nc['under31_2012'] = (df_mex_nc['age_2012_adjusted'] < 31).astype(int)

# Age at arrival: must have arrived before 16th birthday
# YRIMMIG gives year of immigration
# Age at immigration = YRIMMIG - BIRTHYR
df_mex_nc['age_at_arrival'] = df_mex_nc['YRIMMIG'] - df_mex_nc['BIRTHYR']
df_mex_nc['arrived_before_16'] = (df_mex_nc['age_at_arrival'] < 16).astype(int)

# Handle cases where YRIMMIG is 0 (missing/N/A)
df_mex_nc.loc[df_mex_nc['YRIMMIG'] == 0, 'arrived_before_16'] = np.nan

# Continuous presence requirement: in US since June 15, 2007 (5+ years by 2012)
# YRSUSA1 gives years in the USA (at time of survey)
# We need to check if they were in US by 2007
# For each survey year, calculate approximate year of arrival
df_mex_nc['year_arrived'] = df_mex_nc['YRIMMIG']
df_mex_nc['in_us_since_2007'] = (df_mex_nc['year_arrived'] <= 2007).astype(int)
df_mex_nc.loc[df_mex_nc['YRIMMIG'] == 0, 'in_us_since_2007'] = np.nan

print(f"\nEligibility component distributions:")
print(f"Under 31 on June 15, 2012: {df_mex_nc['under31_2012'].mean():.3f}")
print(f"Arrived before age 16: {df_mex_nc['arrived_before_16'].mean():.3f}")
print(f"In US since 2007: {df_mex_nc['in_us_since_2007'].mean():.3f}")

# Define DACA eligible: meets all criteria
# Note: We cannot verify continuous physical presence, so we proxy with year of immigration
df_mex_nc['daca_eligible'] = (
    (df_mex_nc['under31_2012'] == 1) &
    (df_mex_nc['arrived_before_16'] == 1) &
    (df_mex_nc['in_us_since_2007'] == 1)
).astype(int)

# For those with missing immigration data, set eligibility to missing
df_mex_nc.loc[df_mex_nc['YRIMMIG'] == 0, 'daca_eligible'] = np.nan

print(f"DACA eligible: {df_mex_nc['daca_eligible'].mean():.3f} (of those with valid data)")

# Define post-treatment period
# DACA announced June 15, 2012, applications started August 15, 2012
# 2012 is transition year - exclude it
# Pre: 2006-2011, Post: 2013-2016
print("\n[6] Defining treatment periods...")
df_mex_nc['post'] = (df_mex_nc['YEAR'] >= 2013).astype(int)

# Exclude 2012 for cleaner identification
df_analysis = df_mex_nc[df_mex_nc['YEAR'] != 2012].copy()
print(f"Observations after excluding 2012: {len(df_analysis):,}")

# Drop observations with missing eligibility
df_analysis = df_analysis.dropna(subset=['daca_eligible'])
print(f"Observations with valid eligibility data: {len(df_analysis):,}")

# Create working-age sample (16-64 years old at time of survey)
print("\n[7] Creating working-age sample (16-64)...")
df_working = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)].copy()
print(f"Working-age observations: {len(df_working):,}")

# Summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print("\n[A] Sample sizes by group and period:")
crosstab = pd.crosstab(df_working['daca_eligible'], df_working['post'], margins=True)
crosstab.index = ['Ineligible', 'Eligible', 'Total']
crosstab.columns = ['Pre (2006-11)', 'Post (2013-16)', 'Total']
print(crosstab)

print("\n[B] Full-time employment rates by group and period:")
ft_rates = df_working.groupby(['daca_eligible', 'post'])['fulltime'].agg(['mean', 'std', 'count'])
ft_rates = ft_rates.reset_index()
ft_rates.columns = ['DACA Eligible', 'Post', 'Mean', 'Std', 'N']
print(ft_rates.to_string(index=False))

# Calculate simple DiD estimate
print("\n[C] Simple Difference-in-Differences:")
rate_elig_pre = df_working[(df_working['daca_eligible']==1) & (df_working['post']==0)]['fulltime'].mean()
rate_elig_post = df_working[(df_working['daca_eligible']==1) & (df_working['post']==1)]['fulltime'].mean()
rate_inelig_pre = df_working[(df_working['daca_eligible']==0) & (df_working['post']==0)]['fulltime'].mean()
rate_inelig_post = df_working[(df_working['daca_eligible']==0) & (df_working['post']==1)]['fulltime'].mean()

print(f"Eligible Pre:    {rate_elig_pre:.4f}")
print(f"Eligible Post:   {rate_elig_post:.4f}")
print(f"Ineligible Pre:  {rate_inelig_pre:.4f}")
print(f"Ineligible Post: {rate_inelig_post:.4f}")

diff_eligible = rate_elig_post - rate_elig_pre
diff_ineligible = rate_inelig_post - rate_inelig_pre
did_estimate = diff_eligible - diff_ineligible

print(f"\nChange for Eligible:   {diff_eligible:.4f}")
print(f"Change for Ineligible: {diff_ineligible:.4f}")
print(f"DiD Estimate:          {did_estimate:.4f}")

# Regression Analysis
print("\n" + "=" * 70)
print("REGRESSION ANALYSIS")
print("=" * 70)

# Create interaction term
df_working['treat_post'] = df_working['daca_eligible'] * df_working['post']

# Model 1: Basic DiD without controls
print("\n[Model 1] Basic DiD without controls:")
model1 = smf.ols('fulltime ~ daca_eligible + post + treat_post', data=df_working).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n[Model 2] DiD with demographic controls (age, sex, education):")
# Create age polynomial
df_working['age_sq'] = df_working['AGE'] ** 2

# Education categories (from EDUCD)
# 0-61: Less than high school
# 62-64: High school diploma
# 65-100: Some college
# 101+: Bachelor's or higher
df_working['educ_hs'] = ((df_working['EDUCD'] >= 62) & (df_working['EDUCD'] <= 64)).astype(int)
df_working['educ_somecoll'] = ((df_working['EDUCD'] >= 65) & (df_working['EDUCD'] <= 100)).astype(int)
df_working['educ_ba'] = (df_working['EDUCD'] >= 101).astype(int)

model2 = smf.ols('fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba',
                  data=df_working).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls and year fixed effects
print("\n[Model 3] DiD with demographics and year fixed effects:")
df_working['year_factor'] = df_working['YEAR'].astype(str)
model3 = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba + C(YEAR)',
                  data=df_working).fit(cov_type='HC1')
# Print only the key coefficients
print(f"treat_post coefficient: {model3.params['treat_post']:.5f}")
print(f"treat_post std error:   {model3.bse['treat_post']:.5f}")
print(f"treat_post t-stat:      {model3.tvalues['treat_post']:.3f}")
print(f"treat_post p-value:     {model3.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.5f}, {model3.conf_int().loc['treat_post', 1]:.5f}]")

# Model 4: DiD with demographics, year and state fixed effects
print("\n[Model 4] DiD with demographics, year and state fixed effects:")
model4 = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba + C(YEAR) + C(STATEFIP)',
                  data=df_working).fit(cov_type='HC1')
print(f"treat_post coefficient: {model4.params['treat_post']:.5f}")
print(f"treat_post std error:   {model4.bse['treat_post']:.5f}")
print(f"treat_post t-stat:      {model4.tvalues['treat_post']:.3f}")
print(f"treat_post p-value:     {model4.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.5f}, {model4.conf_int().loc['treat_post', 1]:.5f}]")

# Model 5: Full model with marital status and family controls
print("\n[Model 5] Full model with marital status and family controls:")
df_working['married'] = (df_working['MARST'].isin([1, 2])).astype(int)
df_working['has_children'] = (df_working['NCHILD'] > 0).astype(int)

model5 = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba + married + has_children + C(YEAR) + C(STATEFIP)',
                  data=df_working).fit(cov_type='HC1')
print(f"treat_post coefficient: {model5.params['treat_post']:.5f}")
print(f"treat_post std error:   {model5.bse['treat_post']:.5f}")
print(f"treat_post t-stat:      {model5.tvalues['treat_post']:.3f}")
print(f"treat_post p-value:     {model5.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['treat_post', 0]:.5f}, {model5.conf_int().loc['treat_post', 1]:.5f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs):,}")

# Robustness: Weighted regression using person weights
print("\n[Model 6] Weighted regression (using PERWT):")
model6 = smf.wls('fulltime ~ daca_eligible + treat_post + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba + married + has_children + C(YEAR) + C(STATEFIP)',
                  data=df_working, weights=df_working['PERWT']).fit(cov_type='HC1')
print(f"treat_post coefficient: {model6.params['treat_post']:.5f}")
print(f"treat_post std error:   {model6.bse['treat_post']:.5f}")
print(f"treat_post t-stat:      {model6.tvalues['treat_post']:.3f}")
print(f"treat_post p-value:     {model6.pvalues['treat_post']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['treat_post', 0]:.5f}, {model6.conf_int().loc['treat_post', 1]:.5f}]")

# Additional analysis by gender
print("\n" + "=" * 70)
print("HETEROGENEITY ANALYSIS BY GENDER")
print("=" * 70)

print("\n[Gender Analysis] Male subsample:")
df_male = df_working[df_working['SEX'] == 1].copy()
model_male = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + educ_hs + educ_somecoll + educ_ba + married + has_children + C(YEAR) + C(STATEFIP)',
                      data=df_male).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_male.params['treat_post']:.5f}")
print(f"treat_post std error:   {model_male.bse['treat_post']:.5f}")
print(f"95% CI: [{model_male.conf_int().loc['treat_post', 0]:.5f}, {model_male.conf_int().loc['treat_post', 1]:.5f}]")
print(f"N: {int(model_male.nobs):,}")

print("\n[Gender Analysis] Female subsample:")
df_female = df_working[df_working['SEX'] == 2].copy()
model_female = smf.ols('fulltime ~ daca_eligible + treat_post + AGE + age_sq + educ_hs + educ_somecoll + educ_ba + married + has_children + C(YEAR) + C(STATEFIP)',
                        data=df_female).fit(cov_type='HC1')
print(f"treat_post coefficient: {model_female.params['treat_post']:.5f}")
print(f"treat_post std error:   {model_female.bse['treat_post']:.5f}")
print(f"95% CI: [{model_female.conf_int().loc['treat_post', 0]:.5f}, {model_female.conf_int().loc['treat_post', 1]:.5f}]")
print(f"N: {int(model_female.nobs):,}")

# Event study / Pre-trends check
print("\n" + "=" * 70)
print("EVENT STUDY / PRE-TRENDS CHECK")
print("=" * 70)

# Create year interactions with treatment
df_working['year_2006'] = (df_working['YEAR'] == 2006).astype(int)
df_working['year_2007'] = (df_working['YEAR'] == 2007).astype(int)
df_working['year_2008'] = (df_working['YEAR'] == 2008).astype(int)
df_working['year_2009'] = (df_working['YEAR'] == 2009).astype(int)
df_working['year_2010'] = (df_working['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_working['year_2013'] = (df_working['YEAR'] == 2013).astype(int)
df_working['year_2014'] = (df_working['YEAR'] == 2014).astype(int)
df_working['year_2015'] = (df_working['YEAR'] == 2015).astype(int)
df_working['year_2016'] = (df_working['YEAR'] == 2016).astype(int)

# Interactions
df_working['elig_2006'] = df_working['daca_eligible'] * df_working['year_2006']
df_working['elig_2007'] = df_working['daca_eligible'] * df_working['year_2007']
df_working['elig_2008'] = df_working['daca_eligible'] * df_working['year_2008']
df_working['elig_2009'] = df_working['daca_eligible'] * df_working['year_2009']
df_working['elig_2010'] = df_working['daca_eligible'] * df_working['year_2010']
df_working['elig_2013'] = df_working['daca_eligible'] * df_working['year_2013']
df_working['elig_2014'] = df_working['daca_eligible'] * df_working['year_2014']
df_working['elig_2015'] = df_working['daca_eligible'] * df_working['year_2015']
df_working['elig_2016'] = df_working['daca_eligible'] * df_working['year_2016']

model_event = smf.ols('fulltime ~ daca_eligible + elig_2006 + elig_2007 + elig_2008 + elig_2009 + elig_2010 + elig_2013 + elig_2014 + elig_2015 + elig_2016 + AGE + age_sq + SEX + educ_hs + educ_somecoll + educ_ba + married + has_children + C(YEAR) + C(STATEFIP)',
                       data=df_working).fit(cov_type='HC1')

print("\nEvent Study Coefficients (2011 is reference year):")
event_vars = ['elig_2006', 'elig_2007', 'elig_2008', 'elig_2009', 'elig_2010',
              'elig_2013', 'elig_2014', 'elig_2015', 'elig_2016']
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_lo, ci_hi = model_event.conf_int().loc[var]
    print(f"{var}: {coef:8.5f} (SE: {se:.5f}) [{ci_lo:.5f}, {ci_hi:.5f}]")

# Test for pre-trends (joint F-test on pre-period interactions)
print("\nTest for pre-trends (H0: all pre-period interactions = 0):")
pre_vars = ['elig_2006', 'elig_2007', 'elig_2008', 'elig_2009', 'elig_2010']
r_matrix = np.zeros((len(pre_vars), len(model_event.params)))
for i, var in enumerate(pre_vars):
    r_matrix[i, model_event.params.index.get_loc(var)] = 1
f_test = model_event.f_test(r_matrix)
try:
    fval = f_test.fvalue[0][0] if hasattr(f_test.fvalue, '__getitem__') else f_test.fvalue
except:
    fval = float(f_test.fvalue)
print(f"F-statistic: {fval:.3f}")
print(f"p-value: {float(f_test.pvalue):.4f}")

# Save key results
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE (Model 5 - Full model with OLS)")
print("=" * 70)
print(f"\nEffect of DACA eligibility on full-time employment:")
print(f"Coefficient: {model5.params['treat_post']:.5f}")
print(f"Standard Error: {model5.bse['treat_post']:.5f}")
print(f"95% Confidence Interval: [{model5.conf_int().loc['treat_post', 0]:.5f}, {model5.conf_int().loc['treat_post', 1]:.5f}]")
print(f"t-statistic: {model5.tvalues['treat_post']:.3f}")
print(f"p-value: {model5.pvalues['treat_post']:.4f}")
print(f"Sample size: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

# Save results to file
results_dict = {
    'Model': ['Model 1 (Basic)', 'Model 2 (+ Demographics)', 'Model 3 (+ Year FE)',
              'Model 4 (+ State FE)', 'Model 5 (+ Family)', 'Model 6 (Weighted)'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                    model3.params['treat_post'], model4.params['treat_post'],
                    model5.params['treat_post'], model6.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                  model3.bse['treat_post'], model4.bse['treat_post'],
                  model5.bse['treat_post'], model6.bse['treat_post']],
    'CI_Lower': [model1.conf_int().loc['treat_post', 0], model2.conf_int().loc['treat_post', 0],
                 model3.conf_int().loc['treat_post', 0], model4.conf_int().loc['treat_post', 0],
                 model5.conf_int().loc['treat_post', 0], model6.conf_int().loc['treat_post', 0]],
    'CI_Upper': [model1.conf_int().loc['treat_post', 1], model2.conf_int().loc['treat_post', 1],
                 model3.conf_int().loc['treat_post', 1], model4.conf_int().loc['treat_post', 1],
                 model5.conf_int().loc['treat_post', 1], model6.conf_int().loc['treat_post', 1]],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model6.nobs)]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# Save event study results
event_results = []
for var in event_vars:
    year = var.replace('elig_', '')
    event_results.append({
        'Year': int(year),
        'Coefficient': model_event.params[var],
        'Std_Error': model_event.bse[var],
        'CI_Lower': model_event.conf_int().loc[var, 0],
        'CI_Upper': model_event.conf_int().loc[var, 1]
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Create summary statistics table
print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS TABLE")
print("=" * 70)

desc_vars = ['fulltime', 'AGE', 'SEX', 'married', 'has_children', 'educ_hs', 'educ_somecoll', 'educ_ba']

def desc_stats(data, name):
    stats = {}
    for var in desc_vars:
        stats[var] = {'mean': data[var].mean(), 'std': data[var].std()}
    return pd.DataFrame(stats).T

# By treatment group and period
groups = [
    ('Eligible, Pre', df_working[(df_working['daca_eligible']==1) & (df_working['post']==0)]),
    ('Eligible, Post', df_working[(df_working['daca_eligible']==1) & (df_working['post']==1)]),
    ('Ineligible, Pre', df_working[(df_working['daca_eligible']==0) & (df_working['post']==0)]),
    ('Ineligible, Post', df_working[(df_working['daca_eligible']==0) & (df_working['post']==1)])
]

desc_table = pd.DataFrame()
for name, data in groups:
    means = data[desc_vars].mean()
    means.name = name
    desc_table = pd.concat([desc_table, means.to_frame().T])

print(desc_table.round(3).to_string())
desc_table.to_csv('descriptive_stats.csv')
print("\nDescriptive statistics saved to descriptive_stats.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

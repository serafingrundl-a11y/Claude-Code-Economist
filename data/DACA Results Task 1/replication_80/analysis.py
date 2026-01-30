"""
DACA Replication Study: Effect on Full-Time Employment
Analysis of Hispanic-Mexican, Mexican-born individuals
Using American Community Survey (ACS) data 2006-2016

This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment (>=35 hours/week).
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

print("=" * 80)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
print("\n[1] Loading ACS data...")

# Define data types for efficient memory usage
dtypes = {
    'YEAR': 'int16',
    'SAMPLE': 'int32',
    'SERIAL': 'int32',
    'HHWT': 'float32',
    'STATEFIP': 'int8',
    'PUMA': 'int32',
    'METRO': 'int8',
    'GQ': 'int8',
    'PERNUM': 'int16',
    'PERWT': 'float32',
    'FAMSIZE': 'int8',
    'NCHILD': 'int8',
    'RELATE': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'MARST': 'int8',
    'BIRTHYR': 'int16',
    'RACE': 'int8',
    'HISPAN': 'int8',
    'HISPAND': 'int16',
    'BPL': 'int16',
    'BPLD': 'int32',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'YRSUSA1': 'int8',
    'EDUC': 'int8',
    'EDUCD': 'int16',
    'EMPSTAT': 'int8',
    'EMPSTATD': 'int8',
    'LABFORCE': 'int8',
    'CLASSWKR': 'int8',
    'UHRSWORK': 'int8',
    'INCTOT': 'int32',
    'INCWAGE': 'int32',
    'POVERTY': 'int16'
}

# Load only necessary columns
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
           'BIRTHYR', 'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
           'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK', 'INCWAGE', 'METRO']

print("Loading data file (this may take a few minutes)...")
df = pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes)
print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# -----------------------------------------------------------------------------
# 2. FILTER TO SAMPLE OF INTEREST
# -----------------------------------------------------------------------------
print("\n[2] Filtering to sample of interest...")

# Filter to Hispanic-Mexican ethnicity (HISPAN == 1 indicates Mexican)
# HISPAND provides detailed codes: 100-107 are Mexican categories
print(f"  Observations before filtering: {len(df):,}")

# Filter 1: Hispanic-Mexican (HISPAN == 1)
df = df[df['HISPAN'] == 1].copy()
print(f"  After filtering to Hispanic-Mexican (HISPAN=1): {len(df):,}")

# Filter 2: Born in Mexico (BPL == 200 for Mexico)
df = df[df['BPL'] == 200].copy()
print(f"  After filtering to Mexico-born (BPL=200): {len(df):,}")

# Filter 3: Non-citizens (CITIZEN == 3 means not a citizen)
# Per instructions: assume non-citizens without papers are undocumented
df = df[df['CITIZEN'] == 3].copy()
print(f"  After filtering to non-citizens (CITIZEN=3): {len(df):,}")

# Filter 4: Working-age population (16-64 years old)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"  After filtering to ages 16-64: {len(df):,}")

# -----------------------------------------------------------------------------
# 3. CREATE KEY VARIABLES
# -----------------------------------------------------------------------------
print("\n[3] Creating analysis variables...")

# 3a. DACA Eligibility Criteria (as of June 15, 2012):
# - Arrived before 16th birthday
# - Under 31 on June 15, 2012 (birth year >= 1982, or born in 1981 Q3-Q4)
# - In US continuously since June 15, 2007 (immigration year <= 2007)
# - Present in US on June 15, 2012

# Calculate age at immigration
# YRIMMIG gives year of immigration
# We need: immigration year + 16 > birth year (arrived before turning 16)
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# Create DACA eligibility indicator
# Criterion 1: Arrived before 16th birthday
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)

# Criterion 2: Under 31 on June 15, 2012 (born after June 15, 1981)
# Born in 1982 or later definitely eligible
# Born in 1981: only if born in Q3 or Q4 (July-Dec)
# Being conservative: use birthyr > 1981 or (birthyr == 1981 & birthqtr >= 3)
df['under_31_jun2012'] = ((df['BIRTHYR'] > 1981) |
                          ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)

# Criterion 3: Continuous residence since June 15, 2007 (immigration <= 2007)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)

# Combined DACA eligibility
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                       (df['under_31_jun2012'] == 1) &
                       (df['in_us_since_2007'] == 1)).astype(int)

print(f"  DACA eligible individuals: {df['daca_eligible'].sum():,} ({df['daca_eligible'].mean()*100:.1f}%)")

# 3b. Post-DACA indicator (2013-2016)
# Note: 2012 is excluded as program implemented mid-year (June 15, 2012)
df['post_daca'] = (df['YEAR'] >= 2013).astype(int)

# 3c. Treatment variable (DiD interaction)
df['daca_x_post'] = df['daca_eligible'] * df['post_daca']

# 3d. Outcome: Full-time employment (usually works >= 35 hours/week)
# UHRSWORK: usual hours worked per week (0 = N/A or not working)
df['fulltime_emp'] = (df['UHRSWORK'] >= 35).astype(int)

# 3e. Alternative outcome: Employed (any employment)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# 3f. Alternative outcome: In labor force
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

print(f"  Full-time employment rate: {df['fulltime_emp'].mean()*100:.1f}%")
print(f"  Employment rate: {df['employed'].mean()*100:.1f}%")
print(f"  Labor force participation: {df['in_labor_force'].mean()*100:.1f}%")

# 3g. Control variables
# Education categories
df['educ_less_hs'] = (df['EDUCD'] < 62).astype(int)  # Less than high school
df['educ_hs'] = ((df['EDUCD'] >= 62) & (df['EDUCD'] < 65)).astype(int)  # HS diploma
df['educ_some_college'] = ((df['EDUCD'] >= 65) & (df['EDUCD'] < 101)).astype(int)
df['educ_ba_plus'] = (df['EDUCD'] >= 101).astype(int)

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Male indicator
df['male'] = (df['SEX'] == 1).astype(int)

# Metropolitan area
df['metro'] = (df['METRO'].isin([2, 3, 4])).astype(int)

# Age squared (for age polynomial)
df['age_sq'] = df['AGE'] ** 2

# Years in US (proxy using YRIMMIG)
df['years_in_us'] = df['YEAR'] - df['YRIMMIG']

# -----------------------------------------------------------------------------
# 4. DESCRIPTIVE STATISTICS
# -----------------------------------------------------------------------------
print("\n[4] Descriptive Statistics")
print("-" * 60)

# By DACA eligibility status
desc_vars = ['fulltime_emp', 'employed', 'in_labor_force', 'AGE', 'male',
             'married', 'educ_less_hs', 'educ_hs', 'educ_some_college',
             'educ_ba_plus', 'years_in_us']

print("\nMeans by DACA Eligibility Status:")
print("-" * 60)
desc_table = df.groupby('daca_eligible')[desc_vars].mean()
desc_table = desc_table.T
desc_table.columns = ['Non-Eligible', 'DACA-Eligible']
desc_table['Difference'] = desc_table['DACA-Eligible'] - desc_table['Non-Eligible']
print(desc_table.round(3))

# Sample sizes by year and eligibility
print("\n\nSample sizes by Year and DACA Eligibility:")
print("-" * 60)
sample_table = df.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_table.columns = ['Non-Eligible', 'DACA-Eligible']
sample_table['Total'] = sample_table.sum(axis=1)
print(sample_table)
print(f"\nTotal observations: {len(df):,}")

# Full-time employment by year and eligibility
print("\n\nFull-Time Employment Rate by Year and DACA Eligibility:")
print("-" * 60)
ft_table = df.groupby(['YEAR', 'daca_eligible'])['fulltime_emp'].mean().unstack()
ft_table.columns = ['Non-Eligible', 'DACA-Eligible']
ft_table['Difference'] = ft_table['DACA-Eligible'] - ft_table['Non-Eligible']
print(ft_table.round(4))

# -----------------------------------------------------------------------------
# 5. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[5] DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# 5a. Simple DiD (no controls)
print("\n5a. Basic Difference-in-Differences (No Controls)")
print("-" * 60)

# Manual 2x2 DiD calculation
pre_treat = df[(df['daca_eligible'] == 1) & (df['post_daca'] == 0)]['fulltime_emp'].mean()
post_treat = df[(df['daca_eligible'] == 1) & (df['post_daca'] == 1)]['fulltime_emp'].mean()
pre_control = df[(df['daca_eligible'] == 0) & (df['post_daca'] == 0)]['fulltime_emp'].mean()
post_control = df[(df['daca_eligible'] == 0) & (df['post_daca'] == 1)]['fulltime_emp'].mean()

did_manual = (post_treat - pre_treat) - (post_control - pre_control)
print(f"Pre-DACA, Eligible:     {pre_treat:.4f}")
print(f"Post-DACA, Eligible:    {post_treat:.4f}")
print(f"Pre-DACA, Non-Eligible: {pre_control:.4f}")
print(f"Post-DACA, Non-Eligible:{post_control:.4f}")
print(f"\nDiD Estimate (manual):  {did_manual:.4f}")

# Regression-based DiD
model1 = smf.ols('fulltime_emp ~ daca_eligible + post_daca + daca_x_post',
                  data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"\nRegression DiD Estimate: {model1.params['daca_x_post']:.4f}")
print(f"Standard Error (clustered by state): {model1.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model1.tvalues['daca_x_post']:.3f}")
print(f"p-value: {model1.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['daca_x_post', 0]:.4f}, {model1.conf_int().loc['daca_x_post', 1]:.4f}]")

# 5b. DiD with demographic controls
print("\n\n5b. Difference-in-Differences with Demographic Controls")
print("-" * 60)

model2 = smf.ols('fulltime_emp ~ daca_eligible + post_daca + daca_x_post + '
                  'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                  'years_in_us + metro',
                  data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"DiD Estimate: {model2.params['daca_x_post']:.4f}")
print(f"Standard Error (clustered): {model2.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model2.tvalues['daca_x_post']:.3f}")
print(f"p-value: {model2.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['daca_x_post', 0]:.4f}, {model2.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {int(model2.nobs):,}")

# 5c. DiD with year and state fixed effects
print("\n\n5c. Difference-in-Differences with Year and State Fixed Effects")
print("-" * 60)

# Create year dummies (excluding 2006 as reference)
df['year_fe'] = pd.Categorical(df['YEAR'])
df['state_fe'] = pd.Categorical(df['STATEFIP'])

model3 = smf.ols('fulltime_emp ~ daca_eligible + daca_x_post + '
                  'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                  'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                  data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"DiD Estimate: {model3.params['daca_x_post']:.4f}")
print(f"Standard Error (clustered): {model3.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model3.tvalues['daca_x_post']:.3f}")
print(f"p-value: {model3.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['daca_x_post', 0]:.4f}, {model3.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {int(model3.nobs):,}")

# -----------------------------------------------------------------------------
# 6. ROBUSTNESS CHECKS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[6] ROBUSTNESS CHECKS")
print("=" * 80)

# 6a. Alternative outcome: Employment (any)
print("\n6a. Alternative Outcome: Employment (any)")
print("-" * 60)

model_emp = smf.ols('employed ~ daca_eligible + daca_x_post + '
                     'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                     'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                     data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Estimate: {model_emp.params['daca_x_post']:.4f}")
print(f"SE: {model_emp.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model_emp.conf_int().loc['daca_x_post', 0]:.4f}, {model_emp.conf_int().loc['daca_x_post', 1]:.4f}]")

# 6b. Alternative outcome: Labor force participation
print("\n6b. Alternative Outcome: Labor Force Participation")
print("-" * 60)

model_lf = smf.ols('in_labor_force ~ daca_eligible + daca_x_post + '
                    'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                    'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                    data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Estimate: {model_lf.params['daca_x_post']:.4f}")
print(f"SE: {model_lf.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model_lf.conf_int().loc['daca_x_post', 0]:.4f}, {model_lf.conf_int().loc['daca_x_post', 1]:.4f}]")

# 6c. By gender subgroups
print("\n6c. Heterogeneity by Gender")
print("-" * 60)

# Males
df_male = df[df['male'] == 1]
model_male = smf.ols('fulltime_emp ~ daca_eligible + daca_x_post + '
                      'AGE + age_sq + married + educ_hs + educ_some_college + educ_ba_plus + '
                      'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                      data=df_male).fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"Males - DiD Estimate: {model_male.params['daca_x_post']:.4f} (SE: {model_male.bse['daca_x_post']:.4f})")

# Females
df_female = df[df['male'] == 0]
model_female = smf.ols('fulltime_emp ~ daca_eligible + daca_x_post + '
                        'AGE + age_sq + married + educ_hs + educ_some_college + educ_ba_plus + '
                        'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                        data=df_female).fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"Females - DiD Estimate: {model_female.params['daca_x_post']:.4f} (SE: {model_female.bse['daca_x_post']:.4f})")

# 6d. Pre-trend check using 2010 as placebo
print("\n6d. Placebo Test: Pre-Trend Check (2010 as fake treatment year)")
print("-" * 60)

df_pretrend = df[df['YEAR'] <= 2011].copy()
df_pretrend['placebo_post'] = (df_pretrend['YEAR'] >= 2010).astype(int)
df_pretrend['placebo_x_post'] = df_pretrend['daca_eligible'] * df_pretrend['placebo_post']

model_placebo = smf.ols('fulltime_emp ~ daca_eligible + placebo_post + placebo_x_post + '
                         'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                         'years_in_us + metro + C(YEAR) + C(STATEFIP)',
                         data=df_pretrend).fit(cov_type='cluster', cov_kwds={'groups': df_pretrend['STATEFIP']})
print(f"Placebo DiD Estimate: {model_placebo.params['placebo_x_post']:.4f}")
print(f"SE: {model_placebo.bse['placebo_x_post']:.4f}")
print(f"p-value: {model_placebo.pvalues['placebo_x_post']:.4f}")
print(f"(Expect: not significant if parallel trends hold)")

# 6e. Event study specification
print("\n6e. Event Study (Year-by-Year Effects)")
print("-" * 60)

# Create year dummies interacted with eligibility (2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'eligible_x_{year}'] = (df['daca_eligible'] * (df['YEAR'] == year)).astype(int)

event_vars = ['eligible_x_2006', 'eligible_x_2007', 'eligible_x_2008',
              'eligible_x_2009', 'eligible_x_2010', 'eligible_x_2013',
              'eligible_x_2014', 'eligible_x_2015', 'eligible_x_2016']

event_formula = ('fulltime_emp ~ daca_eligible + ' + ' + '.join(event_vars) + ' + '
                 'AGE + age_sq + male + married + educ_hs + educ_some_college + educ_ba_plus + '
                 'years_in_us + metro + C(YEAR) + C(STATEFIP)')

model_event = smf.ols(event_formula, data=df).fit(cov_type='cluster',
                                                    cov_kwds={'groups': df['STATEFIP']})

print("Year-by-Year Effects (relative to 2011):")
print("-" * 40)
for var in event_vars:
    year = var.split('_')[-1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    pval = model_event.pvalues[var]
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"  {year}: {coef:8.4f} ({se:.4f}) {sig}")

# -----------------------------------------------------------------------------
# 7. WEIGHTED ANALYSIS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[7] WEIGHTED ANALYSIS (using PERWT)")
print("=" * 80)

import statsmodels.api as sm

# Prepare data for weighted regression
df_weighted = df.dropna(subset=['fulltime_emp', 'daca_eligible', 'post_daca', 'daca_x_post',
                                 'AGE', 'age_sq', 'male', 'married', 'educ_hs',
                                 'educ_some_college', 'educ_ba_plus', 'years_in_us', 'metro',
                                 'PERWT']).copy()

# Create design matrix
y = df_weighted['fulltime_emp']
X = df_weighted[['daca_eligible', 'post_daca', 'daca_x_post', 'AGE', 'age_sq',
                  'male', 'married', 'educ_hs', 'educ_some_college', 'educ_ba_plus',
                  'years_in_us', 'metro']]
X = sm.add_constant(X)
weights = df_weighted['PERWT']

model_weighted = sm.WLS(y, X, weights=weights).fit(cov_type='cluster',
                                                     cov_kwds={'groups': df_weighted['STATEFIP']})
print(f"Weighted DiD Estimate: {model_weighted.params['daca_x_post']:.4f}")
print(f"SE (clustered): {model_weighted.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['daca_x_post', 0]:.4f}, {model_weighted.conf_int().loc['daca_x_post', 1]:.4f}]")

# -----------------------------------------------------------------------------
# 8. PREFERRED SPECIFICATION SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[8] PREFERRED SPECIFICATION SUMMARY")
print("=" * 80)

# Preferred model: Model 3 (with year and state FE, demographic controls, clustered SE)
print("\nPreferred Model: DiD with Year and State Fixed Effects")
print("Outcome: Full-time employment (usually works >= 35 hours/week)")
print("-" * 60)
print(f"DiD Estimate (DACA effect): {model3.params['daca_x_post']:.4f}")
print(f"Standard Error (clustered by state): {model3.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model3.tvalues['daca_x_post']:.3f}")
print(f"p-value: {model3.pvalues['daca_x_post']:.4f}")
ci_low = model3.conf_int().loc['daca_x_post', 0]
ci_high = model3.conf_int().loc['daca_x_post', 1]
print(f"95% Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"Sample Size: {int(model3.nobs):,}")
print(f"R-squared: {model3.rsquared:.4f}")

# Interpretation
print("\nInterpretation:")
print("-" * 60)
effect_pp = model3.params['daca_x_post'] * 100
print(f"DACA eligibility is associated with a {effect_pp:.2f} percentage point")
if model3.params['daca_x_post'] > 0:
    print("INCREASE in the probability of full-time employment.")
else:
    print("DECREASE in the probability of full-time employment.")

baseline_rate = df[(df['daca_eligible'] == 1) & (df['post_daca'] == 0)]['fulltime_emp'].mean()
relative_effect = (model3.params['daca_x_post'] / baseline_rate) * 100
print(f"Relative to pre-DACA baseline for eligible individuals ({baseline_rate*100:.1f}%),")
print(f"this represents a {relative_effect:.1f}% relative change.")

# -----------------------------------------------------------------------------
# 9. SAVE RESULTS FOR REPORT
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[9] Saving Results")
print("=" * 80)

# Create results dictionary for export
results = {
    'Preferred Model': {
        'Estimate': model3.params['daca_x_post'],
        'SE': model3.bse['daca_x_post'],
        'CI_low': ci_low,
        'CI_high': ci_high,
        't_stat': model3.tvalues['daca_x_post'],
        'p_value': model3.pvalues['daca_x_post'],
        'N': int(model3.nobs),
        'R2': model3.rsquared
    },
    'Basic Model': {
        'Estimate': model1.params['daca_x_post'],
        'SE': model1.bse['daca_x_post'],
        'N': int(model1.nobs)
    },
    'Controls Only Model': {
        'Estimate': model2.params['daca_x_post'],
        'SE': model2.bse['daca_x_post'],
        'N': int(model2.nobs)
    },
    'Employment Outcome': {
        'Estimate': model_emp.params['daca_x_post'],
        'SE': model_emp.bse['daca_x_post']
    },
    'LFP Outcome': {
        'Estimate': model_lf.params['daca_x_post'],
        'SE': model_lf.bse['daca_x_post']
    },
    'Male Subgroup': {
        'Estimate': model_male.params['daca_x_post'],
        'SE': model_male.bse['daca_x_post']
    },
    'Female Subgroup': {
        'Estimate': model_female.params['daca_x_post'],
        'SE': model_female.bse['daca_x_post']
    },
    'Placebo Test': {
        'Estimate': model_placebo.params['placebo_x_post'],
        'SE': model_placebo.bse['placebo_x_post'],
        'p_value': model_placebo.pvalues['placebo_x_post']
    },
    'Weighted Model': {
        'Estimate': model_weighted.params['daca_x_post'],
        'SE': model_weighted.bse['daca_x_post']
    }
}

# Save results to pickle
import pickle
with open('analysis_results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save summary stats
summary_stats = {
    'total_observations': len(df),
    'daca_eligible_count': df['daca_eligible'].sum(),
    'daca_eligible_pct': df['daca_eligible'].mean() * 100,
    'pre_daca_obs': (df['post_daca'] == 0).sum(),
    'post_daca_obs': (df['post_daca'] == 1).sum(),
    'fulltime_rate_overall': df['fulltime_emp'].mean() * 100,
    'fulltime_rate_eligible_pre': pre_treat * 100,
    'fulltime_rate_eligible_post': post_treat * 100,
    'fulltime_rate_ineligible_pre': pre_control * 100,
    'fulltime_rate_ineligible_post': post_control * 100,
    'sample_table': sample_table,
    'ft_table': ft_table,
    'desc_table': desc_table
}

with open('summary_stats.pkl', 'wb') as f:
    pickle.dump(summary_stats, f)

# Save event study results
event_results = {}
for var in event_vars:
    year = var.split('_')[-1]
    event_results[year] = {
        'coef': model_event.params[var],
        'se': model_event.bse[var],
        'pval': model_event.pvalues[var]
    }
# Add 2011 as reference (0 by construction)
event_results['2011'] = {'coef': 0, 'se': 0, 'pval': 1}
# Add 2012 (excluded due to mid-year implementation)
event_results['2012'] = {'coef': np.nan, 'se': np.nan, 'pval': np.nan}

with open('event_study_results.pkl', 'wb') as f:
    pickle.dump(event_results, f)

print("Results saved to:")
print("  - analysis_results.pkl")
print("  - summary_stats.pkl")
print("  - event_study_results.pkl")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

"""
DACA Replication Study - Analysis Script
Replication 44

Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# ==============================================================================
# STEP 1: Load Data
# ==============================================================================
print("\n[1] Loading data...")

# Load in chunks due to large file size
chunks = []
dtypes = {
    'YEAR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'BIRTHYR': 'int16',
    'BIRTHQTR': 'int8',
    'AGE': 'int8',
    'SEX': 'int8',
    'UHRSWORK': 'int8',
    'EMPSTAT': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'PERWT': 'float64',
    'STATEFIP': 'int8',
}

# Read only needed columns
usecols = ['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'BIRTHYR', 'BIRTHQTR',
           'AGE', 'SEX', 'UHRSWORK', 'EMPSTAT', 'EDUC', 'MARST', 'PERWT', 'STATEFIP']

# First, filter for relevant population while reading
chunksize = 1_000_000
for chunk in pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunksize):
    # Filter for Hispanic-Mexican AND Mexican-born
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Sample size after filtering for Hispanic-Mexican, Mexican-born: {len(df):,}")

# ==============================================================================
# STEP 2: Sample Restrictions
# ==============================================================================
print("\n[2] Applying sample restrictions...")

# Exclude 2012 (DACA implementation year - cannot distinguish pre/post)
df = df[df['YEAR'] != 2012]
print(f"After excluding 2012: {len(df):,}")

# Focus on working-age population (16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]
print(f"After restricting to ages 16-64: {len(df):,}")

# Require valid immigration year for non-citizens
# Keep citizens and non-citizens with valid YRIMMIG
df = df[(df['CITIZEN'] != 3) | ((df['CITIZEN'] == 3) & (df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= df['YEAR']))]
print(f"After requiring valid YRIMMIG for non-citizens: {len(df):,}")

# ==============================================================================
# STEP 3: Construct Variables
# ==============================================================================
print("\n[3] Constructing analysis variables...")

# Create post-DACA indicator (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Age at arrival for non-citizens
df['age_at_arrival'] = np.where(df['CITIZEN'] == 3,
                                 df['YRIMMIG'] - df['BIRTHYR'],
                                 np.nan)

# DACA Eligibility criteria (for non-citizens only):
# 1. Arrived before age 16
# 2. Under 31 on June 15, 2012 (born after June 15, 1981)
# 3. Continuous presence since June 2007 (YRIMMIG <= 2007)
# 4. Non-citizen

# Criterion 1: Arrived before 16th birthday
crit1 = (df['age_at_arrival'] < 16) & (df['age_at_arrival'] >= 0)

# Criterion 2: Under 31 on June 15, 2012
# Born after June 15, 1981 means:
# - BIRTHYR > 1981, OR
# - BIRTHYR == 1981 AND BIRTHQTR >= 3 (July-Sept or later, to be conservative)
crit2 = (df['BIRTHYR'] > 1981) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))

# Criterion 3: Continuous presence since June 2007
crit3 = df['YRIMMIG'] <= 2007

# Criterion 4: Non-citizen (not naturalized)
crit4 = df['CITIZEN'] == 3

# DACA eligible
df['daca_eligible'] = (crit1 & crit2 & crit3 & crit4).astype(int)

# Create outcome: Full-time employment (>=35 hours/week)
# UHRSWORK of 0 means N/A or not employed
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Education categories
df['educ_cat'] = pd.cut(df['EDUC'],
                        bins=[-1, 5, 6, 9, 11],
                        labels=['less_hs', 'hs', 'some_college', 'college_plus'])

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status
df['married'] = (df['MARST'] <= 2).astype(int)

print(f"\nDACA-eligible individuals: {df['daca_eligible'].sum():,}")
print(f"Non-DACA-eligible individuals: {(df['daca_eligible'] == 0).sum():,}")

# ==============================================================================
# STEP 4: Define Analysis Sample
# ==============================================================================
print("\n[4] Defining analysis sample...")

# For main DiD: Compare DACA-eligible to similar non-eligible non-citizens
# Control group: Non-citizen Mexican-born individuals who are NOT DACA eligible
# (e.g., arrived too old, or arrived too late, or too old on June 15, 2012)

# Analysis sample: Non-citizens only (both eligible and ineligible)
analysis_df = df[df['CITIZEN'] == 3].copy()
print(f"Analysis sample (non-citizens only): {len(analysis_df):,}")

# Further restrict to reasonable age range for comparison
# Focus on ages where we have both eligible and ineligible individuals
# DACA eligible must be under 31 in 2012, so in 2016 they'd be under 35
# But we also need a comparison group of similar ages who are ineligible
# Let's use ages 16-45 to have sufficient comparison

analysis_df = analysis_df[(analysis_df['AGE'] >= 16) & (analysis_df['AGE'] <= 45)]
print(f"After age restriction (16-45): {len(analysis_df):,}")

# ==============================================================================
# STEP 5: Descriptive Statistics
# ==============================================================================
print("\n[5] Descriptive Statistics")
print("=" * 80)

# Summary by eligibility status and period
print("\n--- Sample Sizes by Year and Eligibility ---")
cross = pd.crosstab(analysis_df['YEAR'], analysis_df['daca_eligible'], margins=True)
cross.columns = ['Not Eligible', 'DACA Eligible', 'Total']
print(cross)

print("\n--- Full-time Employment Rates by Year and Eligibility ---")
ft_rates = analysis_df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
ft_rates.columns = ['Not Eligible', 'DACA Eligible']
print(ft_rates.round(4))

# Pre/post comparison
print("\n--- Pre/Post DACA Full-time Employment Rates ---")
pre_post = analysis_df.groupby(['post', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
pre_post.index = ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']
pre_post.columns = ['Not Eligible', 'DACA Eligible']
print(pre_post.round(4))

# Simple DiD calculation
pre_elig = pre_post.loc['Pre-DACA (2006-2011)', 'DACA Eligible']
pre_inelig = pre_post.loc['Pre-DACA (2006-2011)', 'Not Eligible']
post_elig = pre_post.loc['Post-DACA (2013-2016)', 'DACA Eligible']
post_inelig = pre_post.loc['Post-DACA (2013-2016)', 'Not Eligible']

did_simple = (post_elig - pre_elig) - (post_inelig - pre_inelig)
print(f"\nSimple Difference-in-Differences: {did_simple:.4f}")

# Demographic characteristics
print("\n--- Demographics by DACA Eligibility ---")
demo_vars = ['AGE', 'female', 'married', 'employed', 'fulltime']
for var in demo_vars:
    elig_mean = np.average(analysis_df[analysis_df['daca_eligible']==1][var],
                          weights=analysis_df[analysis_df['daca_eligible']==1]['PERWT'])
    inelig_mean = np.average(analysis_df[analysis_df['daca_eligible']==0][var],
                            weights=analysis_df[analysis_df['daca_eligible']==0]['PERWT'])
    print(f"{var:12}: Eligible = {elig_mean:.3f}, Not Eligible = {inelig_mean:.3f}")

# ==============================================================================
# STEP 6: Difference-in-Differences Regression
# ==============================================================================
print("\n[6] Difference-in-Differences Regression Analysis")
print("=" * 80)

# Create interaction term
analysis_df['post_x_eligible'] = analysis_df['post'] * analysis_df['daca_eligible']

# Model 1: Basic DiD
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls('fulltime ~ post + daca_eligible + post_x_eligible',
                 data=analysis_df, weights=analysis_df['PERWT'])
results1 = model1.fit(cov_type='HC1')
print(results1.summary().tables[1])

# Model 2: With demographic controls
print("\n--- Model 2: DiD with Demographics ---")
model2 = smf.wls('fulltime ~ post + daca_eligible + post_x_eligible + AGE + I(AGE**2) + female + married',
                 data=analysis_df, weights=analysis_df['PERWT'])
results2 = model2.fit(cov_type='HC1')
print(results2.summary().tables[1])

# Model 3: With year and state fixed effects
print("\n--- Model 3: DiD with Year FE ---")
analysis_df['year_cat'] = analysis_df['YEAR'].astype('category')
model3 = smf.wls('fulltime ~ C(year_cat) + daca_eligible + post_x_eligible + AGE + I(AGE**2) + female + married',
                 data=analysis_df, weights=analysis_df['PERWT'])
results3 = model3.fit(cov_type='HC1')
# Print only key coefficients
print("Key coefficients:")
for var in ['daca_eligible', 'post_x_eligible', 'AGE', 'female', 'married']:
    if var in results3.params:
        print(f"  {var}: {results3.params[var]:.4f} (SE: {results3.bse[var]:.4f})")

# Model 4: With state fixed effects
print("\n--- Model 4: DiD with Year and State FE ---")
analysis_df['state_cat'] = analysis_df['STATEFIP'].astype('category')
model4 = smf.wls('fulltime ~ C(year_cat) + C(state_cat) + daca_eligible + post_x_eligible + AGE + I(AGE**2) + female + married',
                 data=analysis_df, weights=analysis_df['PERWT'])
results4 = model4.fit(cov_type='HC1')
print("Key coefficients:")
for var in ['daca_eligible', 'post_x_eligible', 'AGE', 'female', 'married']:
    if var in results4.params:
        print(f"  {var}: {results4.params[var]:.4f} (SE: {results4.bse[var]:.4f})")

# ==============================================================================
# STEP 7: Robustness Checks
# ==============================================================================
print("\n[7] Robustness Checks")
print("=" * 80)

# Robustness 1: Alternative control group - naturalized citizens from Mexico
print("\n--- Robustness: Using naturalized citizens as control ---")
robust_df = df[(df['BPL'] == 200) & (df['HISPAN'] == 1) &
               ((df['CITIZEN'] == 3) | (df['CITIZEN'] == 2))].copy()
robust_df = robust_df[(robust_df['AGE'] >= 16) & (robust_df['AGE'] <= 45)]

# Treatment: non-citizen and DACA eligible
# Control: naturalized citizen (similar immigrant background but not affected by DACA)
robust_df['treat'] = ((robust_df['CITIZEN'] == 3) & (robust_df['daca_eligible'] == 1)).astype(int)
robust_df['post_x_treat'] = robust_df['post'] * robust_df['treat']

model_robust = smf.wls('fulltime ~ post + treat + post_x_treat + AGE + I(AGE**2) + female + married',
                       data=robust_df, weights=robust_df['PERWT'])
results_robust = model_robust.fit(cov_type='HC1')
print("Key coefficients:")
for var in ['treat', 'post_x_treat']:
    if var in results_robust.params:
        print(f"  {var}: {results_robust.params[var]:.4f} (SE: {results_robust.bse[var]:.4f})")

# Robustness 2: Triple difference (age-based variation)
print("\n--- Robustness: Narrow age bandwidth (ages 18-35) ---")
narrow_df = analysis_df[(analysis_df['AGE'] >= 18) & (analysis_df['AGE'] <= 35)].copy()
model_narrow = smf.wls('fulltime ~ post + daca_eligible + post_x_eligible + AGE + I(AGE**2) + female + married',
                       data=narrow_df, weights=narrow_df['PERWT'])
results_narrow = model_narrow.fit(cov_type='HC1')
print("Key coefficients:")
for var in ['daca_eligible', 'post_x_eligible']:
    if var in results_narrow.params:
        print(f"  {var}: {results_narrow.params[var]:.4f} (SE: {results_narrow.bse[var]:.4f})")

# Robustness 3: Placebo test (pre-period)
print("\n--- Placebo Test: Fake policy in 2009 (using 2006-2011 data only) ---")
placebo_df = analysis_df[analysis_df['YEAR'] <= 2011].copy()
placebo_df['fake_post'] = (placebo_df['YEAR'] >= 2009).astype(int)
placebo_df['fake_post_x_elig'] = placebo_df['fake_post'] * placebo_df['daca_eligible']

model_placebo = smf.wls('fulltime ~ fake_post + daca_eligible + fake_post_x_elig + AGE + I(AGE**2) + female + married',
                        data=placebo_df, weights=placebo_df['PERWT'])
results_placebo = model_placebo.fit(cov_type='HC1')
print("Key coefficients:")
for var in ['daca_eligible', 'fake_post_x_elig']:
    if var in results_placebo.params:
        print(f"  {var}: {results_placebo.params[var]:.4f} (SE: {results_placebo.bse[var]:.4f})")

# ==============================================================================
# STEP 8: Event Study
# ==============================================================================
print("\n[8] Event Study Analysis")
print("=" * 80)

# Create year dummies interacted with eligibility
event_df = analysis_df.copy()
event_df['year'] = event_df['YEAR']
base_year = 2011  # last pre-treatment year

event_coefs = []
event_ses = []
years_list = sorted(event_df['YEAR'].unique())

for yr in years_list:
    if yr != base_year:
        event_df[f'yr_{yr}'] = (event_df['YEAR'] == yr).astype(int)
        event_df[f'yr_{yr}_x_elig'] = event_df[f'yr_{yr}'] * event_df['daca_eligible']

# Build formula with all year interactions
year_terms = ' + '.join([f'yr_{yr} + yr_{yr}_x_elig' for yr in years_list if yr != base_year])
formula = f'fulltime ~ daca_eligible + {year_terms} + AGE + I(AGE**2) + female + married'

model_event = smf.wls(formula, data=event_df, weights=event_df['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nEvent Study Coefficients (Year x DACA Eligible interactions):")
print("Year    Coefficient    Std.Err.    95% CI")
print("-" * 50)
for yr in years_list:
    if yr != base_year:
        coef_name = f'yr_{yr}_x_elig'
        coef = results_event.params[coef_name]
        se = results_event.bse[coef_name]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        event_coefs.append((yr, coef, se))
        print(f"{yr}    {coef:10.4f}    {se:8.4f}    [{ci_low:.4f}, {ci_high:.4f}]")

# ==============================================================================
# STEP 9: Summary of Results
# ==============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF RESULTS")
print("=" * 80)

print(f"""
Main Finding:
The difference-in-differences estimate of the effect of DACA eligibility on
full-time employment is:

  Preferred Estimate (Model 4 with Year and State FE):
  Coefficient: {results4.params['post_x_eligible']:.4f}
  Standard Error: {results4.bse['post_x_eligible']:.4f}
  95% CI: [{results4.params['post_x_eligible'] - 1.96*results4.bse['post_x_eligible']:.4f}, {results4.params['post_x_eligible'] + 1.96*results4.bse['post_x_eligible']:.4f}]

Sample Size: {len(analysis_df):,}
Treatment Group (DACA Eligible): {analysis_df['daca_eligible'].sum():,}
Control Group (Not Eligible): {(analysis_df['daca_eligible']==0).sum():,}

Interpretation:
DACA eligibility is associated with a {abs(results4.params['post_x_eligible'])*100:.1f} percentage point
{'increase' if results4.params['post_x_eligible'] > 0 else 'decrease'} in the probability of full-time employment
(working 35+ hours per week) among Mexican-born non-citizens.
""")

# Save key results for report
results_summary = {
    'preferred_estimate': results4.params['post_x_eligible'],
    'preferred_se': results4.bse['post_x_eligible'],
    'sample_size': len(analysis_df),
    'n_treated': analysis_df['daca_eligible'].sum(),
    'n_control': (analysis_df['daca_eligible']==0).sum(),
    'model1_coef': results1.params['post_x_eligible'],
    'model1_se': results1.bse['post_x_eligible'],
    'model2_coef': results2.params['post_x_eligible'],
    'model2_se': results2.bse['post_x_eligible'],
    'model3_coef': results3.params['post_x_eligible'],
    'model3_se': results3.bse['post_x_eligible'],
    'model4_coef': results4.params['post_x_eligible'],
    'model4_se': results4.bse['post_x_eligible'],
    'pre_post_table': pre_post,
    'ft_rates': ft_rates,
    'placebo_coef': results_placebo.params['fake_post_x_elig'],
    'placebo_se': results_placebo.bse['fake_post_x_elig'],
    'narrow_coef': results_narrow.params['post_x_eligible'],
    'narrow_se': results_narrow.bse['post_x_eligible'],
    'event_coefs': event_coefs
}

# Save results to file
import pickle
with open('results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)

print("\nResults saved to results_summary.pkl")
print("\nAnalysis complete.")

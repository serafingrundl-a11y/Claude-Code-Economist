"""
DACA Replication Study - Analysis Script
=========================================
Research Question: What is the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States?

Identification Strategy: Difference-in-Differences
- Treatment: DACA-eligible non-citizens
- Control: DACA-ineligible non-citizens (similar demographics)
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For output
import os

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD DATA (using chunked reading for memory efficiency)
# ============================================================================
print("\n[1] Loading data...")

data_path = "data/data.csv"

# Define columns we need
needed_cols = [
    'YEAR', 'PERWT', 'STATEFIP', 'AGE', 'SEX', 'BIRTHQTR', 'BIRTHYR',
    'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'EMPSTAT',
    'UHRSWORK', 'MARST', 'NCHILD'
]

# Read in chunks and filter to target population
chunks = []
chunk_size = 500000

print("Reading data in chunks and filtering to target population...")
for i, chunk in enumerate(pd.read_csv(data_path, usecols=needed_cols, chunksize=chunk_size)):
    # Filter to target population in each chunk:
    # Hispanic-Mexican (HISPAN == 1), Born in Mexico (BPL == 200), Non-citizen (CITIZEN == 3)
    filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200) &
        (chunk['CITIZEN'] == 3)
    ].copy()
    chunks.append(filtered)
    if (i + 1) % 10 == 0:
        print(f"  Processed {(i+1) * chunk_size:,} rows...")

df_target = pd.concat(chunks, ignore_index=True)
del chunks

print(f"Target population loaded: {len(df_target):,} observations")
print(f"Years in data: {sorted(df_target['YEAR'].unique())}")

# ============================================================================
# STEP 2: EXCLUDE 2012 AND DEFINE TIME PERIODS
# ============================================================================
print("\n[2] Defining time periods...")

# Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df_target = df_target[df_target['YEAR'] != 2012].copy()

# Create post-treatment indicator (2013-2016)
df_target['post'] = (df_target['YEAR'] >= 2013).astype(int)

print(f"After excluding 2012: {len(df_target):,}")
print(f"Pre-period (2006-2011): {(df_target['post'] == 0).sum():,}")
print(f"Post-period (2013-2016): {(df_target['post'] == 1).sum():,}")

# ============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n[3] Defining DACA eligibility...")

# DACA eligibility criteria as of June 15, 2012:
# 1. Arrived before age 16
# 2. Born after June 15, 1981 (under 31 as of June 15, 2012)
# 3. In the US since at least June 15, 2007 (YRIMMIG <= 2007)
# 4. Not a citizen (already filtered)

# Calculate age at arrival
# YRIMMIG is year of immigration
# Age at arrival = YRIMMIG - BIRTHYR
df_target['age_at_arrival'] = df_target['YRIMMIG'] - df_target['BIRTHYR']

# For birth date cutoff: Born after June 15, 1981
# Using BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# Born after June 15, 1981 means:
# - BIRTHYR > 1981, OR
# - BIRTHYR == 1981 AND BIRTHQTR >= 3 (Jul-Sep or later)
df_target['born_after_cutoff'] = (
    (df_target['BIRTHYR'] > 1981) |
    ((df_target['BIRTHYR'] == 1981) & (df_target['BIRTHQTR'] >= 3))
)

# Eligibility conditions
df_target['arrived_before_16'] = (df_target['age_at_arrival'] < 16)
df_target['in_us_since_2007'] = (df_target['YRIMMIG'] <= 2007)

# Handle missing YRIMMIG (coded as 0)
df_target.loc[df_target['YRIMMIG'] == 0, 'arrived_before_16'] = False
df_target.loc[df_target['YRIMMIG'] == 0, 'in_us_since_2007'] = False

# DACA eligible if all criteria met
df_target['daca_eligible'] = (
    df_target['arrived_before_16'] &
    df_target['born_after_cutoff'] &
    df_target['in_us_since_2007']
).astype(int)

print(f"Arrived before age 16: {df_target['arrived_before_16'].sum():,}")
print(f"Born after June 15, 1981: {df_target['born_after_cutoff'].sum():,}")
print(f"In US since 2007 or earlier: {df_target['in_us_since_2007'].sum():,}")
print(f"DACA eligible (all criteria): {df_target['daca_eligible'].sum():,}")
print(f"DACA ineligible: {(df_target['daca_eligible'] == 0).sum():,}")

# ============================================================================
# STEP 4: RESTRICT TO WORKING-AGE POPULATION
# ============================================================================
print("\n[4] Restricting to working-age population...")

# Focus on ages 16-64 for labor market analysis
df_target = df_target[(df_target['AGE'] >= 16) & (df_target['AGE'] <= 64)].copy()

print(f"Working age (16-64) sample size: {len(df_target):,}")
print(f"DACA eligible in working-age sample: {df_target['daca_eligible'].sum():,}")
print(f"DACA ineligible in working-age sample: {(df_target['daca_eligible'] == 0).sum():,}")

# ============================================================================
# STEP 5: CREATE OUTCOME VARIABLE
# ============================================================================
print("\n[5] Creating outcome variable...")

# Full-time employment: UHRSWORK >= 35
# Note: UHRSWORK = 0 can mean N/A or truly 0 hours
# EMPSTAT = 1 means employed

df_target['employed'] = (df_target['EMPSTAT'] == 1).astype(int)
df_target['fulltime'] = ((df_target['UHRSWORK'] >= 35) & (df_target['EMPSTAT'] == 1)).astype(int)

print(f"Employed: {df_target['employed'].sum():,} ({100*df_target['employed'].mean():.1f}%)")
print(f"Full-time employed: {df_target['fulltime'].sum():,} ({100*df_target['fulltime'].mean():.1f}%)")

# ============================================================================
# STEP 6: CREATE CONTROL VARIABLES
# ============================================================================
print("\n[6] Creating control variables...")

# Age and age squared
df_target['age_sq'] = df_target['AGE'] ** 2

# Female indicator
df_target['female'] = (df_target['SEX'] == 2).astype(int)

# Married indicator
df_target['married'] = (df_target['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC: 0=N/A, 1=Nursery-4, 2=5-8, 3=9, 4=10, 5=11, 6=12/HS, 7=1yr college,
#       8=2yr college, 9=3yr college, 10=4yr college, 11=5+ college
df_target['educ_lths'] = (df_target['EDUC'] <= 5).astype(int)  # Less than HS
df_target['educ_hs'] = (df_target['EDUC'] == 6).astype(int)     # HS graduate
df_target['educ_somecol'] = (df_target['EDUC'].isin([7, 8, 9])).astype(int)  # Some college
df_target['educ_col'] = (df_target['EDUC'] >= 10).astype(int)   # College+

# Number of children
df_target['has_children'] = (df_target['NCHILD'] > 0).astype(int)

# State fixed effects (will use STATEFIP)
df_target['state'] = df_target['STATEFIP']

# Year fixed effects
df_target['year'] = df_target['YEAR']

print("Control variables created.")
print(f"Female: {100*df_target['female'].mean():.1f}%")
print(f"Married: {100*df_target['married'].mean():.1f}%")
print(f"Less than HS: {100*df_target['educ_lths'].mean():.1f}%")
print(f"HS graduate: {100*df_target['educ_hs'].mean():.1f}%")
print(f"Some college: {100*df_target['educ_somecol'].mean():.1f}%")
print(f"College+: {100*df_target['educ_col'].mean():.1f}%")

# ============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[7] Generating descriptive statistics...")

# Summary by treatment status and time period
def weighted_mean(df, var, weight='PERWT'):
    return np.average(df[var], weights=df[weight])

def weighted_std(df, var, weight='PERWT'):
    avg = weighted_mean(df, var, weight)
    variance = np.average((df[var] - avg)**2, weights=df[weight])
    return np.sqrt(variance)

groups = df_target.groupby(['daca_eligible', 'post'])

desc_stats = []
for (eligible, post), group in groups:
    stats_dict = {
        'DACA Eligible': 'Yes' if eligible else 'No',
        'Period': 'Post (2013-16)' if post else 'Pre (2006-11)',
        'N': len(group),
        'N (weighted)': group['PERWT'].sum(),
        'Age': weighted_mean(group, 'AGE'),
        'Female (%)': 100 * weighted_mean(group, 'female'),
        'Married (%)': 100 * weighted_mean(group, 'married'),
        'Less than HS (%)': 100 * weighted_mean(group, 'educ_lths'),
        'HS Graduate (%)': 100 * weighted_mean(group, 'educ_hs'),
        'Some College (%)': 100 * weighted_mean(group, 'educ_somecol'),
        'College+ (%)': 100 * weighted_mean(group, 'educ_col'),
        'Employed (%)': 100 * weighted_mean(group, 'employed'),
        'Full-time (%)': 100 * weighted_mean(group, 'fulltime'),
    }
    desc_stats.append(stats_dict)

desc_df = pd.DataFrame(desc_stats)
print("\nDescriptive Statistics by Group:")
print(desc_df.to_string(index=False))

# Save descriptive statistics
desc_df.to_csv('descriptive_stats.csv', index=False)

# ============================================================================
# STEP 8: DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n[8] Running difference-in-differences analysis...")

# Create interaction term
df_target['treat_post'] = df_target['daca_eligible'] * df_target['post']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.wls(
    'fulltime ~ daca_eligible + post + treat_post',
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.wls(
    'fulltime ~ daca_eligible + post + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children',
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})
print(model2.summary().tables[1])

# Model 3: DiD with Year Fixed Effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
model3 = smf.wls(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year)',
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})
print(f"Treatment effect (treat_post): {model3.params['treat_post']:.4f}")
print(f"Std. Error: {model3.bse['treat_post']:.4f}")
print(f"t-stat: {model3.tvalues['treat_post']:.3f}")
print(f"p-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with State and Year Fixed Effects (Preferred Specification)
print("\n--- Model 4: DiD with State and Year Fixed Effects (PREFERRED) ---")
model4 = smf.wls(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})

print(f"\n*** PREFERRED ESTIMATE ***")
print(f"Treatment effect (treat_post): {model4.params['treat_post']:.4f}")
print(f"Std. Error: {model4.bse['treat_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")
print(f"t-stat: {model4.tvalues['treat_post']:.3f}")
print(f"p-value: {model4.pvalues['treat_post']:.4f}")
print(f"N: {int(model4.nobs):,}")

# ============================================================================
# STEP 9: ROBUSTNESS CHECKS
# ============================================================================
print("\n[9] Running robustness checks...")

# Robustness 1: Employment (any) as outcome
print("\n--- Robustness 1: Employment (any) as outcome ---")
model_emp = smf.wls(
    'employed ~ daca_eligible + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})
print(f"Effect on employment: {model_emp.params['treat_post']:.4f} (SE: {model_emp.bse['treat_post']:.4f})")

# Robustness 2: Restrict to ages 18-35 (core DACA-age range)
print("\n--- Robustness 2: Ages 18-35 subsample ---")
df_young = df_target[(df_target['AGE'] >= 18) & (df_target['AGE'] <= 35)]
model_young = smf.wls(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_young,
    weights=df_young['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_young['state']})
print(f"Effect (ages 18-35): {model_young.params['treat_post']:.4f} (SE: {model_young.bse['treat_post']:.4f})")
print(f"N: {int(model_young.nobs):,}")

# Robustness 3: Male subsample
print("\n--- Robustness 3: Male subsample ---")
df_male = df_target[df_target['female'] == 0]
model_male = smf.wls(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_male,
    weights=df_male['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_male['state']})
print(f"Effect (males): {model_male.params['treat_post']:.4f} (SE: {model_male.bse['treat_post']:.4f})")

# Robustness 4: Female subsample
print("\n--- Robustness 4: Female subsample ---")
df_female = df_target[df_target['female'] == 1]
model_female = smf.wls(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_female,
    weights=df_female['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_female['state']})
print(f"Effect (females): {model_female.params['treat_post']:.4f} (SE: {model_female.bse['treat_post']:.4f})")

# Robustness 5: Unweighted regression
print("\n--- Robustness 5: Unweighted regression ---")
model_unwt = smf.ols(
    'fulltime ~ daca_eligible + treat_post + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_target
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})
print(f"Effect (unweighted): {model_unwt.params['treat_post']:.4f} (SE: {model_unwt.bse['treat_post']:.4f})")

# ============================================================================
# STEP 10: PLACEBO TEST
# ============================================================================
print("\n[10] Running placebo test...")

# Use only pre-period data, create fake treatment at 2009
df_pre = df_target[df_target['post'] == 0].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_treat'] = df_pre['daca_eligible'] * df_pre['placebo_post']

model_placebo = smf.wls(
    'fulltime ~ daca_eligible + placebo_post + placebo_treat + AGE + age_sq + female + married + '
    'educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)',
    data=df_pre,
    weights=df_pre['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_pre['state']})
print(f"Placebo effect (2009 fake treatment): {model_placebo.params['placebo_treat']:.4f}")
print(f"SE: {model_placebo.bse['placebo_treat']:.4f}")
print(f"p-value: {model_placebo.pvalues['placebo_treat']:.4f}")

# ============================================================================
# STEP 11: EVENT STUDY
# ============================================================================
print("\n[11] Running event study...")

# Create year dummies interacted with treatment
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for yr in years:
    df_target[f'year_{yr}'] = (df_target['YEAR'] == yr).astype(int)
    df_target[f'treat_year_{yr}'] = df_target['daca_eligible'] * df_target[f'year_{yr}']

# Event study regression (omit 2011 as reference)
event_formula = 'fulltime ~ daca_eligible + '
event_formula += ' + '.join([f'treat_year_{yr}' for yr in years if yr != 2011])
event_formula += ' + AGE + age_sq + female + married + educ_hs + educ_somecol + educ_col + has_children + C(year) + C(state)'

model_event = smf.wls(
    event_formula,
    data=df_target,
    weights=df_target['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df_target['state']})

print("\nEvent Study Coefficients (relative to 2011):")
event_results = []
for yr in years:
    if yr != 2011:
        coef = model_event.params[f'treat_year_{yr}']
        se = model_event.bse[f'treat_year_{yr}']
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        print(f"  {yr}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        event_results.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})
    else:
        event_results.append({'year': yr, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# ============================================================================
# STEP 12: SAVE RESULTS
# ============================================================================
print("\n[12] Saving results...")

# Save main regression results
results_dict = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'State+Year FE (Preferred)',
              'Employment Outcome', 'Ages 18-35', 'Males Only', 'Females Only',
              'Unweighted', 'Placebo (2009)'],
    'Coefficient': [
        model1.params['treat_post'],
        model2.params['treat_post'],
        model3.params['treat_post'],
        model4.params['treat_post'],
        model_emp.params['treat_post'],
        model_young.params['treat_post'],
        model_male.params['treat_post'],
        model_female.params['treat_post'],
        model_unwt.params['treat_post'],
        model_placebo.params['placebo_treat']
    ],
    'Std_Error': [
        model1.bse['treat_post'],
        model2.bse['treat_post'],
        model3.bse['treat_post'],
        model4.bse['treat_post'],
        model_emp.bse['treat_post'],
        model_young.bse['treat_post'],
        model_male.bse['treat_post'],
        model_female.bse['treat_post'],
        model_unwt.bse['treat_post'],
        model_placebo.bse['placebo_treat']
    ],
    'p_value': [
        model1.pvalues['treat_post'],
        model2.pvalues['treat_post'],
        model3.pvalues['treat_post'],
        model4.pvalues['treat_post'],
        model_emp.pvalues['treat_post'],
        model_young.pvalues['treat_post'],
        model_male.pvalues['treat_post'],
        model_female.pvalues['treat_post'],
        model_unwt.pvalues['treat_post'],
        model_placebo.pvalues['placebo_treat']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs),
        int(model_emp.nobs),
        int(model_young.nobs),
        int(model_male.nobs),
        int(model_female.nobs),
        int(model_unwt.nobs),
        int(model_placebo.nobs)
    ]
}

results_df = pd.DataFrame(results_dict)
results_df['CI_low'] = results_df['Coefficient'] - 1.96 * results_df['Std_Error']
results_df['CI_high'] = results_df['Coefficient'] + 1.96 * results_df['Std_Error']
results_df.to_csv('regression_results.csv', index=False)

print("\nRegression Results Summary:")
print(results_df.to_string(index=False))

# ============================================================================
# STEP 13: FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"""
PREFERRED ESTIMATE (Model 4: State and Year Fixed Effects):
===========================================================
Sample: Hispanic-Mexican, Mexico-born, non-citizens, ages 16-64
Outcome: Full-time employment (35+ hours/week)
Method: Difference-in-differences with state and year fixed effects

Treatment Effect: {model4.params['treat_post']:.4f}
Standard Error: {model4.bse['treat_post']:.4f} (clustered by state)
95% Confidence Interval: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]
t-statistic: {model4.tvalues['treat_post']:.3f}
p-value: {model4.pvalues['treat_post']:.4f}

Sample Size: {int(model4.nobs):,}
R-squared: {model4.rsquared:.4f}

INTERPRETATION:
DACA eligibility is associated with a {abs(model4.params['treat_post'])*100:.2f} percentage point
{'increase' if model4.params['treat_post'] > 0 else 'decrease'} in the probability of full-time employment.
This effect is {'statistically significant' if model4.pvalues['treat_post'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

print("Analysis complete. Results saved to CSV files.")
print("=" * 70)

# Save key results for the LaTeX report
with open('key_results.txt', 'w') as f:
    f.write(f"PREFERRED_ESTIMATE={model4.params['treat_post']:.6f}\n")
    f.write(f"PREFERRED_SE={model4.bse['treat_post']:.6f}\n")
    f.write(f"PREFERRED_CI_LOW={model4.conf_int().loc['treat_post', 0]:.6f}\n")
    f.write(f"PREFERRED_CI_HIGH={model4.conf_int().loc['treat_post', 1]:.6f}\n")
    f.write(f"PREFERRED_PVALUE={model4.pvalues['treat_post']:.6f}\n")
    f.write(f"PREFERRED_N={int(model4.nobs)}\n")
    f.write(f"PREFERRED_RSQUARED={model4.rsquared:.6f}\n")
    f.write(f"N_TOTAL={len(df_target)}\n")
    f.write(f"N_ELIGIBLE={df_target['daca_eligible'].sum()}\n")
    f.write(f"N_INELIGIBLE={(df_target['daca_eligible']==0).sum()}\n")

    # Save additional stats for report
    f.write(f"MEAN_FULLTIME_ELIGIBLE_PRE={desc_df[(desc_df['DACA Eligible']=='Yes') & (desc_df['Period'].str.contains('Pre'))]['Full-time (%)'].values[0]:.2f}\n")
    f.write(f"MEAN_FULLTIME_ELIGIBLE_POST={desc_df[(desc_df['DACA Eligible']=='Yes') & (desc_df['Period'].str.contains('Post'))]['Full-time (%)'].values[0]:.2f}\n")
    f.write(f"MEAN_FULLTIME_INELIGIBLE_PRE={desc_df[(desc_df['DACA Eligible']=='No') & (desc_df['Period'].str.contains('Pre'))]['Full-time (%)'].values[0]:.2f}\n")
    f.write(f"MEAN_FULLTIME_INELIGIBLE_POST={desc_df[(desc_df['DACA Eligible']=='No') & (desc_df['Period'].str.contains('Post'))]['Full-time (%)'].values[0]:.2f}\n")

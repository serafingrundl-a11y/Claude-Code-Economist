"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA on the probability of full-time employment (35+ hours/week)?

DACA implemented: June 15, 2012
Analysis period: 2006-2016 (pre-period: 2006-2011, post-period: 2013-2016)
2012 is excluded as DACA was implemented mid-year
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: Load and Initial Data Processing (Using Chunked Reading)
# =============================================================================
print("\n[1] Loading data...")

data_path = "data/data.csv"

# Read relevant columns to reduce memory
usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
           'EDUC', 'EDUCD', 'EMPSTAT', 'LABFORCE', 'UHRSWORK']

# Process data in chunks and filter immediately to keep only relevant observations
print("Loading data in chunks and filtering to Hispanic-Mexican Mexico-born sample...")
chunks = []
chunk_count = 0
for chunk in pd.read_csv(data_path, usecols=usecols, chunksize=1000000, low_memory=False):
    # Filter to Hispanic-Mexican ethnicity born in Mexico
    # HISPAN == 1 indicates Mexican Hispanic origin
    # BPL == 200 indicates born in Mexico
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    chunks.append(filtered)
    chunk_count += 1
    print(f"  Processed chunk {chunk_count}, found {len(filtered):,} relevant observations")

df = pd.concat(chunks, ignore_index=True)
del chunks  # Free memory
print(f"\nTotal Hispanic-Mexican Mexico-born observations: {len(df):,}")

# =============================================================================
# STEP 2: Sample Restrictions
# =============================================================================
print("\n[2] Applying sample restrictions...")

# Exclude 2012 (DACA implemented mid-year, cannot distinguish pre/post)
df = df[df['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df):,}")

# Keep working-age population (ages 16-64)
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"After age 16-64 filter: {len(df):,}")

# =============================================================================
# STEP 3: Define DACA Eligibility
# =============================================================================
print("\n[3] Defining DACA eligibility...")

"""
DACA Eligibility Criteria (as of June 15, 2012):
1. Arrived in US before age 16
2. Born after June 15, 1981 (not yet 31 as of June 15, 2012)
3. Lived continuously in US since June 15, 2007 (immigrated by 2007)
4. Present in US on June 15, 2012 without lawful status

Key assumption: Non-citizens (CITIZEN == 3) who haven't naturalized are assumed undocumented.
"""

# Calculate age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility components:
# 1. Arrived before age 16
df['arrived_before_16'] = df['age_at_immig'] < 16

# 2. Born after June 15, 1981 (not yet 31 as of June 15, 2012)
# Using BIRTHYR >= 1982 as conservative measure (born in or after 1982)
df['born_after_1981'] = df['BIRTHYR'] >= 1982

# 3. In US since at least 2007 (continuous residence since June 15, 2007)
df['in_us_since_2007'] = df['YRIMMIG'] <= 2007

# 4. Not a citizen (assume undocumented)
# CITIZEN == 3 means "Not a citizen"
df['not_citizen'] = df['CITIZEN'] == 3

# Define DACA-eligible: meets all criteria
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['born_after_1981'] &
    df['in_us_since_2007'] &
    df['not_citizen']
).astype(int)

print(f"\nDACA eligibility breakdown:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Born after 1981: {df['born_after_1981'].sum():,}")
print(f"  In US since 2007: {df['in_us_since_2007'].sum():,}")
print(f"  Not a citizen: {df['not_citizen'].sum():,}")
print(f"  DACA eligible (all criteria): {df['daca_eligible'].sum():,}")

# =============================================================================
# STEP 4: Define Treatment and Outcome Variables
# =============================================================================
print("\n[4] Defining treatment and outcome variables...")

# Post-DACA period (2013-2016)
df['post'] = (df['YEAR'] >= 2013).astype(int)

# Treatment indicator (DiD interaction)
df['treat_post'] = df['daca_eligible'] * df['post']

# Outcome: Full-time employment (35+ hours per week)
# UHRSWORK = usual hours worked per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Alternative outcome: Employed (in labor force and working)
# EMPSTAT == 1 means employed
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"Post-DACA observations: {df['post'].sum():,}")
print(f"Pre-DACA observations: {(df['post'] == 0).sum():,}")
print(f"Treatment-post observations: {df['treat_post'].sum():,}")

# =============================================================================
# STEP 5: Create Control Variables
# =============================================================================
print("\n[5] Creating control variables...")

# Age and age squared
df['age_sq'] = df['AGE'] ** 2

# Female indicator
df['female'] = (df['SEX'] == 2).astype(int)

# Married indicator
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
# EDUC codes: 0-1 = less than HS, 6 = HS grad, 7-9 = some college, 10-11 = college+
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecoll'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['educ_college'] = (df['EDUC'] >= 10).astype(int)

# Years in US
df['yrs_in_us'] = df['YEAR'] - df['YRIMMIG']
df['yrs_in_us'] = df['yrs_in_us'].clip(lower=0)

# =============================================================================
# STEP 6: Summary Statistics
# =============================================================================
print("\n[6] Summary Statistics")
print("=" * 80)

# Summary by DACA eligibility and period
summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married',
                'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']

print("\n--- Summary Statistics by DACA Eligibility and Period ---")
print("\nDACA-Eligible, Pre-Period (2006-2011):")
pre_elig = df[(df['daca_eligible'] == 1) & (df['post'] == 0)]
print(f"  N = {len(pre_elig):,}")
print(f"  Full-time employment rate: {pre_elig['fulltime'].mean():.4f}")
print(f"  Employment rate: {pre_elig['employed'].mean():.4f}")
print(f"  Mean age: {pre_elig['AGE'].mean():.2f}")
print(f"  Female share: {pre_elig['female'].mean():.4f}")

print("\nDACA-Eligible, Post-Period (2013-2016):")
post_elig = df[(df['daca_eligible'] == 1) & (df['post'] == 1)]
print(f"  N = {len(post_elig):,}")
print(f"  Full-time employment rate: {post_elig['fulltime'].mean():.4f}")
print(f"  Employment rate: {post_elig['employed'].mean():.4f}")
print(f"  Mean age: {post_elig['AGE'].mean():.2f}")
print(f"  Female share: {post_elig['female'].mean():.4f}")

print("\nDACA-Ineligible (Control), Pre-Period (2006-2011):")
pre_inelig = df[(df['daca_eligible'] == 0) & (df['post'] == 0)]
print(f"  N = {len(pre_inelig):,}")
print(f"  Full-time employment rate: {pre_inelig['fulltime'].mean():.4f}")
print(f"  Employment rate: {pre_inelig['employed'].mean():.4f}")
print(f"  Mean age: {pre_inelig['AGE'].mean():.2f}")
print(f"  Female share: {pre_inelig['female'].mean():.4f}")

print("\nDACA-Ineligible (Control), Post-Period (2013-2016):")
post_inelig = df[(df['daca_eligible'] == 0) & (df['post'] == 1)]
print(f"  N = {len(post_inelig):,}")
print(f"  Full-time employment rate: {post_inelig['fulltime'].mean():.4f}")
print(f"  Employment rate: {post_inelig['employed'].mean():.4f}")
print(f"  Mean age: {post_inelig['AGE'].mean():.2f}")
print(f"  Female share: {post_inelig['female'].mean():.4f}")

# =============================================================================
# STEP 7: Simple Difference-in-Differences Calculation
# =============================================================================
print("\n[7] Simple Difference-in-Differences")
print("=" * 80)

# Calculate means for DiD
mean_pre_treat = pre_elig['fulltime'].mean()
mean_post_treat = post_elig['fulltime'].mean()
mean_pre_control = pre_inelig['fulltime'].mean()
mean_post_control = post_inelig['fulltime'].mean()

# DiD estimate
did_estimate = (mean_post_treat - mean_pre_treat) - (mean_post_control - mean_pre_control)

print(f"\nFull-time Employment Rates:")
print(f"                      Pre-DACA    Post-DACA    Difference")
print(f"  DACA-Eligible:      {mean_pre_treat:.4f}       {mean_post_treat:.4f}        {mean_post_treat - mean_pre_treat:+.4f}")
print(f"  DACA-Ineligible:    {mean_pre_control:.4f}       {mean_post_control:.4f}        {mean_post_control - mean_pre_control:+.4f}")
print(f"\n  Difference-in-Differences: {did_estimate:+.4f}")

# =============================================================================
# STEP 8: Regression Analysis
# =============================================================================
print("\n[8] Regression Analysis")
print("=" * 80)

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
X1 = df[['daca_eligible', 'post', 'treat_post']].astype(float)
X1 = sm.add_constant(X1)
y = df['fulltime'].astype(float)
weights = df['PERWT'].astype(float)

model1 = sm.WLS(y, X1, weights=weights).fit(cov_type='HC1')
print(model1.summary())

# Extract coefficient and SE for treat_post
coef_m1 = model1.params['treat_post']
se_m1 = model1.bse['treat_post']
ci_low_m1 = coef_m1 - 1.96 * se_m1
ci_high_m1 = coef_m1 + 1.96 * se_m1

print(f"\nModel 1 - Treatment Effect (treat_post):")
print(f"  Coefficient: {coef_m1:.4f}")
print(f"  Std. Error: {se_m1:.4f}")
print(f"  95% CI: [{ci_low_m1:.4f}, {ci_high_m1:.4f}]")
print(f"  N: {len(df):,}")

# Model 2: DiD with individual controls
print("\n--- Model 2: DiD with Individual Controls ---")
X2 = df[['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq',
         'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X2 = sm.add_constant(X2)

model2 = sm.WLS(y, X2, weights=weights).fit(cov_type='HC1')
print(model2.summary())

coef_m2 = model2.params['treat_post']
se_m2 = model2.bse['treat_post']
ci_low_m2 = coef_m2 - 1.96 * se_m2
ci_high_m2 = coef_m2 + 1.96 * se_m2

print(f"\nModel 2 - Treatment Effect (treat_post):")
print(f"  Coefficient: {coef_m2:.4f}")
print(f"  Std. Error: {se_m2:.4f}")
print(f"  95% CI: [{ci_low_m2:.4f}, {ci_high_m2:.4f}]")
print(f"  N: {len(df):,}")

# Model 3: DiD with year and state fixed effects
print("\n--- Model 3: DiD with Year and State Fixed Effects ---")

# Create year dummies (reference year = 2006)
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True).astype(float)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True).astype(float)

X3_base = df[['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq',
              'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X3 = pd.concat([X3_base, year_dummies, state_dummies], axis=1)
X3 = sm.add_constant(X3)

model3 = sm.WLS(y, X3, weights=weights).fit(cov_type='HC1')

coef_m3 = model3.params['treat_post']
se_m3 = model3.bse['treat_post']
ci_low_m3 = coef_m3 - 1.96 * se_m3
ci_high_m3 = coef_m3 + 1.96 * se_m3

print(f"\nModel 3 - Treatment Effect (treat_post):")
print(f"  Coefficient: {coef_m3:.4f}")
print(f"  Std. Error: {se_m3:.4f}")
print(f"  95% CI: [{ci_low_m3:.4f}, {ci_high_m3:.4f}]")
print(f"  N: {len(df):,}")
print(f"  R-squared: {model3.rsquared:.4f}")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n[9] Robustness Checks")
print("=" * 80)

# Robustness 1: Alternative outcome - Employment (any)
print("\n--- Robustness Check 1: Employment (Any Hours) as Outcome ---")
y_emp = df['employed'].astype(float)
model_rob1 = sm.WLS(y_emp, X3, weights=weights).fit(cov_type='HC1')
coef_rob1 = model_rob1.params['treat_post']
se_rob1 = model_rob1.bse['treat_post']
print(f"  Treatment effect on employment: {coef_rob1:.4f} (SE: {se_rob1:.4f})")

# Robustness 2: Restrict to ages 18-30 (prime DACA-eligible ages)
print("\n--- Robustness Check 2: Ages 18-30 Only ---")
df_young = df[(df['AGE'] >= 18) & (df['AGE'] <= 30)].copy()
X_young = df_young[['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq',
                    'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X_young = sm.add_constant(X_young)
y_young = df_young['fulltime'].astype(float)
w_young = df_young['PERWT'].astype(float)
model_rob2 = sm.WLS(y_young, X_young, weights=w_young).fit(cov_type='HC1')
coef_rob2 = model_rob2.params['treat_post']
se_rob2 = model_rob2.bse['treat_post']
print(f"  Treatment effect (ages 18-30): {coef_rob2:.4f} (SE: {se_rob2:.4f})")
print(f"  N: {len(df_young):,}")

# Robustness 3: By gender
print("\n--- Robustness Check 3: By Gender ---")
df_male = df[df['female'] == 0].copy()
df_female = df[df['female'] == 1].copy()

X_male = df_male[['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq',
                  'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X_male = sm.add_constant(X_male)
model_male = sm.WLS(df_male['fulltime'].astype(float), X_male, weights=df_male['PERWT'].astype(float)).fit(cov_type='HC1')

X_female = df_female[['daca_eligible', 'post', 'treat_post', 'AGE', 'age_sq',
                      'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X_female = sm.add_constant(X_female)
model_female = sm.WLS(df_female['fulltime'].astype(float), X_female, weights=df_female['PERWT'].astype(float)).fit(cov_type='HC1')

coef_male = model_male.params['treat_post']
se_male = model_male.bse['treat_post']
coef_female = model_female.params['treat_post']
se_female = model_female.bse['treat_post']

print(f"  Male treatment effect: {coef_male:.4f} (SE: {se_male:.4f})")
print(f"  Female treatment effect: {coef_female:.4f} (SE: {se_female:.4f})")

# =============================================================================
# STEP 10: Event Study / Pre-Trends Analysis
# =============================================================================
print("\n[10] Event Study / Pre-Trends Analysis")
print("=" * 80)

# Create year-specific treatment effects (relative to 2011)
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_study_results = []

for year in years:
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'treat_year_{year}'] = df['daca_eligible'] * df[f'year_{year}']

# Reference year is 2011
treat_year_cols = [f'treat_year_{y}' for y in years if y != 2011]
year_cols = [f'year_{y}' for y in years if y != 2011]

X_event = df[['daca_eligible'] + year_cols + treat_year_cols +
             ['AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_somecoll', 'educ_college', 'yrs_in_us']].astype(float)
X_event = sm.add_constant(X_event)

model_event = sm.WLS(y, X_event, weights=weights).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Treatment x Year, relative to 2011):")
event_coefs = []
for year in years:
    if year != 2011:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        event_coefs.append({'year': year, 'coefficient': coef, 'std_error': se})
        print(f"  {year}: {coef:+.4f} (SE: {se:.4f})")
    else:
        event_coefs.append({'year': year, 'coefficient': 0, 'std_error': 0})
        print(f"  {year}: 0.0000 (reference)")

# =============================================================================
# STEP 11: Save Results
# =============================================================================
print("\n[11] Saving Results")
print("=" * 80)

# Create results summary
results_summary = {
    'Model': ['Basic DiD', 'DiD + Controls', 'DiD + Controls + FE'],
    'Coefficient': [coef_m1, coef_m2, coef_m3],
    'Std_Error': [se_m1, se_m2, se_m3],
    'CI_Lower': [ci_low_m1, ci_low_m2, ci_low_m3],
    'CI_Upper': [ci_high_m1, ci_high_m2, ci_high_m3],
    'N': [len(df), len(df), len(df)]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save event study results
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# =============================================================================
# STEP 12: Final Summary
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
PREFERRED ESTIMATE (Model 3: DiD with Controls and Year/State Fixed Effects):

  Effect of DACA eligibility on full-time employment: {coef_m3:.4f}
  Standard Error: {se_m3:.4f}
  95% Confidence Interval: [{ci_low_m3:.4f}, {ci_high_m3:.4f}]
  Sample Size: {len(df):,}

INTERPRETATION:
  DACA eligibility is associated with a {abs(coef_m3)*100:.2f} percentage point
  {'increase' if coef_m3 > 0 else 'decrease'} in the probability of full-time employment
  among Hispanic-Mexican individuals born in Mexico.

  This effect is {'statistically significant' if abs(coef_m3/se_m3) > 1.96 else 'not statistically significant'}
  at the 5% level (t-statistic: {coef_m3/se_m3:.2f}).
""")

# Save detailed summary statistics
summary_stats = {
    'Group': ['DACA-Eligible Pre', 'DACA-Eligible Post', 'DACA-Ineligible Pre', 'DACA-Ineligible Post'],
    'N': [len(pre_elig), len(post_elig), len(pre_inelig), len(post_inelig)],
    'Fulltime_Rate': [mean_pre_treat, mean_post_treat, mean_pre_control, mean_post_control],
    'Employed_Rate': [pre_elig['employed'].mean(), post_elig['employed'].mean(),
                      pre_inelig['employed'].mean(), post_inelig['employed'].mean()],
    'Mean_Age': [pre_elig['AGE'].mean(), post_elig['AGE'].mean(),
                 pre_inelig['AGE'].mean(), post_inelig['AGE'].mean()],
    'Female_Share': [pre_elig['female'].mean(), post_elig['female'].mean(),
                     pre_inelig['female'].mean(), post_inelig['female'].mean()],
    'Married_Share': [pre_elig['married'].mean(), post_elig['married'].mean(),
                      pre_inelig['married'].mean(), post_inelig['married'].mean()]
}

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

# Save robustness check results
robustness_results = {
    'Check': ['Employment Outcome', 'Ages 18-30', 'Males Only', 'Females Only'],
    'Coefficient': [coef_rob1, coef_rob2, coef_male, coef_female],
    'Std_Error': [se_rob1, se_rob2, se_male, se_female],
    'N': [len(df), len(df_young), len(df_male), len(df_female)]
}
robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv('robustness_results.csv', index=False)
print("Robustness results saved to robustness_results.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

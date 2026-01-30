"""
DACA Effect on Full-Time Employment: Replication Analysis
=========================================================

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for the Deferred Action for
Childhood Arrivals (DACA) program on the probability of full-time employment (35+ hrs/week)?

Identification Strategy: Difference-in-Differences
- Treatment group: DACA-eligible individuals (non-citizens, arrived before age 16,
  under 31 on June 15, 2012, arrived before June 2007)
- Control group: Similar Hispanic-Mexican Mexican-born individuals who don't meet
  all eligibility criteria
- Pre-period: 2006-2011 (before DACA announced)
- Post-period: 2013-2016 (after DACA implemented)
- 2012 excluded due to implementation occurring mid-year

Author: Independent Replication
Date: 2025
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import gc
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA REPLICATION STUDY: EFFECT ON FULL-TIME EMPLOYMENT")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD DATA IN CHUNKS AND FILTER TO RELEVANT SAMPLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA (FILTERING DURING LOAD)")
print("=" * 80)

# Read in chunks and filter immediately to reduce memory
chunks = []
chunk_size = 1000000

# Define columns we need
usecols = ['YEAR', 'STATEFIP', 'PUMA', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR',
           'BIRTHYR', 'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
           'YRSUSA1', 'EDUC', 'EMPSTAT', 'UHRSWORK', 'METRO']

for i, chunk in enumerate(pd.read_csv('data/data.csv', usecols=usecols, chunksize=chunk_size)):
    # Filter to Hispanic-Mexican (HISPAN == 1) born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)
    print(f"  Processed chunk {i+1}: {len(filtered):,} observations kept")
    gc.collect()

df_sample = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

print(f"\nTotal observations after initial filter: {len(df_sample):,}")
print(f"Years in data: {sorted(df_sample['YEAR'].unique())}")

# ============================================================================
# STEP 2: ADDITIONAL SAMPLE RESTRICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: ADDITIONAL SAMPLE RESTRICTIONS")
print("=" * 80)

print(f"Starting sample (Hispanic-Mexican, Mexican-born): {len(df_sample):,}")

# Exclude 2012 (DACA implemented mid-year, can't distinguish pre/post)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_sample):,}")

# Working age restriction (18-64) - standard for employment analysis
df_sample = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 64)].copy()
print(f"After working age restriction (18-64): {len(df_sample):,}")

# ============================================================================
# STEP 3: DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DEFINE DACA ELIGIBILITY")
print("=" * 80)

"""
DACA Eligibility Requirements (as of June 15, 2012):
1. Arrived in US before 16th birthday
2. Under 31 years old on June 15, 2012 (born after June 15, 1981)
3. Continuously in US since June 15, 2007 (arrived by 2007)
4. Not a citizen and no lawful status

Key insight: We need to define eligibility as of June 15, 2012, not as of survey year.
This means we look at characteristics fixed at the 2012 cutoff.
"""

# CRITERION 1: Arrived before 16th birthday
# Age at arrival = Year of immigration - Birth year
# We need YRIMMIG > 0 (has immigration year)
df_sample['yrimmig_valid'] = df_sample['YRIMMIG'] > 0

# Calculate age at arrival for those with valid immigration year
df_sample['age_at_arrival'] = np.where(
    df_sample['yrimmig_valid'],
    df_sample['YRIMMIG'] - df_sample['BIRTHYR'],
    np.nan
)

# Arrived before 16th birthday
df_sample['arrived_before_16'] = df_sample['age_at_arrival'] < 16

# CRITERION 2: Under 31 on June 15, 2012 (born after June 15, 1981)
# Conservative: born in 1982 or later (definitely under 31)
# Or born in 1981 with birth quarter 3 or 4 (July-Dec)
df_sample['under_31_in_2012'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
)

# CRITERION 3: In US since June 15, 2007 (arrived by 2007)
df_sample['arrived_by_2007'] = df_sample['YRIMMIG'] <= 2007

# CRITERION 4: Not a citizen
# CITIZEN: 0=N/A, 1=Born abroad of American parents, 2=Naturalized, 3=Not a citizen
# We want CITIZEN == 3 (Not a citizen)
df_sample['non_citizen'] = df_sample['CITIZEN'] == 3

# Define DACA eligibility (all criteria must be met)
df_sample['daca_eligible'] = (
    df_sample['arrived_before_16'] &
    df_sample['under_31_in_2012'] &
    df_sample['arrived_by_2007'] &
    df_sample['non_citizen'] &
    df_sample['yrimmig_valid']  # Must have valid immigration year
)

print(f"\nDACA Eligibility Criteria Breakdown:")
print(f"  Valid immigration year: {df_sample['yrimmig_valid'].sum():,} ({df_sample['yrimmig_valid'].mean()*100:.1f}%)")
print(f"  Arrived before age 16: {df_sample['arrived_before_16'].sum():,} ({df_sample['arrived_before_16'].mean()*100:.1f}%)")
print(f"  Under 31 in 2012: {df_sample['under_31_in_2012'].sum():,} ({df_sample['under_31_in_2012'].mean()*100:.1f}%)")
print(f"  Arrived by 2007: {df_sample['arrived_by_2007'].sum():,} ({df_sample['arrived_by_2007'].mean()*100:.1f}%)")
print(f"  Non-citizen: {df_sample['non_citizen'].sum():,} ({df_sample['non_citizen'].mean()*100:.1f}%)")
print(f"  DACA eligible (all criteria): {df_sample['daca_eligible'].sum():,} ({df_sample['daca_eligible'].mean()*100:.1f}%)")

# ============================================================================
# STEP 4: DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: DEFINE OUTCOME VARIABLE - FULL-TIME EMPLOYMENT")
print("=" * 80)

"""
Outcome: Full-time employment (usually working 35+ hours per week)
Using UHRSWORK (usual hours worked per week)
- UHRSWORK >= 35 indicates full-time
- UHRSWORK == 0 indicates not employed or N/A
"""

# Full-time employment (35+ hours)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Also create employed variable for reference
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

print(f"\nOutcome Variable Summary:")
print(f"  Full-time employed (35+ hrs): {df_sample['fulltime'].sum():,} ({df_sample['fulltime'].mean()*100:.1f}%)")
print(f"  Employed (any hours): {df_sample['employed'].sum():,} ({df_sample['employed'].mean()*100:.1f}%)")

# ============================================================================
# STEP 5: DEFINE TREATMENT AND TIME VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: DEFINE TREATMENT AND TIME VARIABLES")
print("=" * 80)

# Post-DACA period (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Treatment indicator
df_sample['treat'] = df_sample['daca_eligible'].astype(int)

# Interaction term (DID estimator)
df_sample['treat_post'] = df_sample['treat'] * df_sample['post']

print(f"\nTreatment and Time Variables:")
print(f"  Pre-period observations (2006-2011): {(df_sample['post'] == 0).sum():,}")
print(f"  Post-period observations (2013-2016): {(df_sample['post'] == 1).sum():,}")
print(f"  Treatment group (DACA eligible): {df_sample['treat'].sum():,}")
print(f"  Control group (not eligible): {(df_sample['treat'] == 0).sum():,}")

# ============================================================================
# STEP 6: CREATE CONTROL VARIABLES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATE CONTROL VARIABLES")
print("=" * 80)

# Age and age squared
df_sample['age_sq'] = df_sample['AGE'] ** 2

# Female indicator
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)

# Marital status
df_sample['married'] = (df_sample['MARST'] <= 2).astype(int)  # 1=married spouse present, 2=married spouse absent

# Education categories
# EDUC: 0=N/A, 1=None, 2=Grades 1-4, 3=Grades 5-6, 4=Grades 7-8, 5=Grade 9,
#       6=Grade 10, 7=Grade 11, 8=Grade 12, 9=1yr college, 10=2yr college, 11=4yr college
df_sample['educ_less_hs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['educ_hs'] = (df_sample['EDUC'].isin([6, 7])).astype(int)
df_sample['educ_hs_grad'] = (df_sample['EDUC'] == 8).astype(int)
df_sample['educ_some_college'] = (df_sample['EDUC'].isin([9, 10])).astype(int)
df_sample['educ_college'] = (df_sample['EDUC'] >= 11).astype(int)

# Years in USA
df_sample['yrs_in_usa'] = df_sample['YRSUSA1']
df_sample.loc[df_sample['yrs_in_usa'] == 0, 'yrs_in_usa'] = np.nan

# Metro status
df_sample['metro'] = (df_sample['METRO'] >= 2).astype(int)

print(f"\nControl Variables Summary:")
print(f"  Mean age: {df_sample['AGE'].mean():.1f}")
print(f"  Female: {df_sample['female'].mean()*100:.1f}%")
print(f"  Married: {df_sample['married'].mean()*100:.1f}%")
print(f"  Less than HS: {df_sample['educ_less_hs'].mean()*100:.1f}%")
print(f"  High school grad: {df_sample['educ_hs_grad'].mean()*100:.1f}%")
print(f"  Some college: {df_sample['educ_some_college'].mean()*100:.1f}%")
print(f"  College grad+: {df_sample['educ_college'].mean()*100:.1f}%")
print(f"  Metro area: {df_sample['metro'].mean()*100:.1f}%")

# ============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Summary by treatment status and time period
print("\n--- Full-Time Employment Rates by Group and Period ---")
summary_table = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean'
}).round(3)
print(summary_table)

# Simple DID calculation
ft_treat_pre = df_sample[(df_sample['treat']==1) & (df_sample['post']==0)]['fulltime'].mean()
ft_treat_post = df_sample[(df_sample['treat']==1) & (df_sample['post']==1)]['fulltime'].mean()
ft_control_pre = df_sample[(df_sample['treat']==0) & (df_sample['post']==0)]['fulltime'].mean()
ft_control_post = df_sample[(df_sample['treat']==0) & (df_sample['post']==1)]['fulltime'].mean()

print(f"\n--- Simple DID Calculation ---")
print(f"Treatment group pre-DACA:  {ft_treat_pre:.4f}")
print(f"Treatment group post-DACA: {ft_treat_post:.4f}")
print(f"Treatment change:          {ft_treat_post - ft_treat_pre:.4f}")
print(f"Control group pre-DACA:    {ft_control_pre:.4f}")
print(f"Control group post-DACA:   {ft_control_post:.4f}")
print(f"Control change:            {ft_control_post - ft_control_pre:.4f}")
print(f"DID estimate:              {(ft_treat_post - ft_treat_pre) - (ft_control_post - ft_control_pre):.4f}")

# ============================================================================
# STEP 8: MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("=" * 80)

# Model 1: Basic DID (no controls)
print("\n--- Model 1: Basic DID (no controls) ---")
X1 = df_sample[['treat', 'post', 'treat_post']].copy()
X1 = sm.add_constant(X1)
y = df_sample['fulltime']

# Use robust standard errors
model1 = sm.OLS(y, X1).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DID with demographic controls
print("\n--- Model 2: DID with Demographic Controls ---")
controls = ['AGE', 'age_sq', 'female', 'married', 'educ_hs', 'educ_hs_grad',
            'educ_some_college', 'educ_college', 'metro']

X2 = df_sample[['treat', 'post', 'treat_post'] + controls].copy()
X2 = sm.add_constant(X2)
X2 = X2.dropna()
y2 = df_sample.loc[X2.index, 'fulltime']

model2 = sm.OLS(y2, X2).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DID with year fixed effects
print("\n--- Model 3: DID with Year Fixed Effects ---")
year_dummies = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True).astype(float)
X3 = pd.concat([df_sample[['treat', 'treat_post'] + controls].astype(float), year_dummies], axis=1)
X3 = sm.add_constant(X3)
X3 = X3.dropna()
y3 = df_sample.loc[X3.index, 'fulltime'].astype(float)

model3 = sm.OLS(y3, X3).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DID with state fixed effects
print("\n--- Model 4: DID with Year and State Fixed Effects ---")
state_dummies = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True).astype(float)
X4 = pd.concat([df_sample[['treat', 'treat_post'] + controls].astype(float), year_dummies, state_dummies], axis=1)
X4 = sm.add_constant(X4)
X4 = X4.dropna()
y4 = df_sample.loc[X4.index, 'fulltime'].astype(float)

model4 = sm.OLS(y4, X4).fit(cov_type='HC1')
print("\nKey coefficients from Model 4:")
print(f"  treat_post (DID estimate): {model4.params['treat_post']:.4f}")
print(f"  Standard Error: {model4.bse['treat_post']:.4f}")
print(f"  t-statistic: {model4.tvalues['treat_post']:.4f}")
print(f"  p-value: {model4.pvalues['treat_post']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# ============================================================================
# STEP 9: ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: ROBUSTNESS CHECKS")
print("=" * 80)

# Robustness 1: Different age restrictions (16-40 for younger sample)
print("\n--- Robustness 1: Younger Sample (Age 18-40) ---")
df_young = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 40)].copy()
year_dummies_y = pd.get_dummies(df_young['YEAR'], prefix='year', drop_first=True).astype(float)
state_dummies_y = pd.get_dummies(df_young['STATEFIP'], prefix='state', drop_first=True).astype(float)
Xr1 = pd.concat([df_young[['treat', 'treat_post'] + controls].astype(float), year_dummies_y, state_dummies_y], axis=1)
Xr1 = sm.add_constant(Xr1)
Xr1 = Xr1.dropna()
yr1 = df_young.loc[Xr1.index, 'fulltime'].astype(float)
model_r1 = sm.OLS(yr1, Xr1).fit(cov_type='HC1')
print(f"  treat_post: {model_r1.params['treat_post']:.4f} (SE: {model_r1.bse['treat_post']:.4f}, p={model_r1.pvalues['treat_post']:.4f})")
print(f"  N = {len(yr1):,}")

# Robustness 2: Male subsample
print("\n--- Robustness 2: Male Subsample ---")
df_male = df_sample[df_sample['female'] == 0].copy()
year_dummies_m = pd.get_dummies(df_male['YEAR'], prefix='year', drop_first=True).astype(float)
state_dummies_m = pd.get_dummies(df_male['STATEFIP'], prefix='state', drop_first=True).astype(float)
controls_m = [c for c in controls if c != 'female']
Xr2 = pd.concat([df_male[['treat', 'treat_post'] + controls_m].astype(float), year_dummies_m, state_dummies_m], axis=1)
Xr2 = sm.add_constant(Xr2)
Xr2 = Xr2.dropna()
yr2 = df_male.loc[Xr2.index, 'fulltime'].astype(float)
model_r2 = sm.OLS(yr2, Xr2).fit(cov_type='HC1')
print(f"  treat_post: {model_r2.params['treat_post']:.4f} (SE: {model_r2.bse['treat_post']:.4f}, p={model_r2.pvalues['treat_post']:.4f})")
print(f"  N = {len(yr2):,}")

# Robustness 3: Female subsample
print("\n--- Robustness 3: Female Subsample ---")
df_female = df_sample[df_sample['female'] == 1].copy()
year_dummies_f = pd.get_dummies(df_female['YEAR'], prefix='year', drop_first=True).astype(float)
state_dummies_f = pd.get_dummies(df_female['STATEFIP'], prefix='state', drop_first=True).astype(float)
Xr3 = pd.concat([df_female[['treat', 'treat_post'] + controls_m].astype(float), year_dummies_f, state_dummies_f], axis=1)
Xr3 = sm.add_constant(Xr3)
Xr3 = Xr3.dropna()
yr3 = df_female.loc[Xr3.index, 'fulltime'].astype(float)
model_r3 = sm.OLS(yr3, Xr3).fit(cov_type='HC1')
print(f"  treat_post: {model_r3.params['treat_post']:.4f} (SE: {model_r3.bse['treat_post']:.4f}, p={model_r3.pvalues['treat_post']:.4f})")
print(f"  N = {len(yr3):,}")

# Robustness 4: Alternative outcome - any employment
print("\n--- Robustness 4: Alternative Outcome - Any Employment ---")
X4_emp = X4.copy()
y4_emp = df_sample.loc[X4.index, 'employed'].astype(float)
model_r4 = sm.OLS(y4_emp, X4_emp).fit(cov_type='HC1')
print(f"  treat_post: {model_r4.params['treat_post']:.4f} (SE: {model_r4.bse['treat_post']:.4f}, p={model_r4.pvalues['treat_post']:.4f})")
print(f"  N = {len(y4_emp):,}")

# ============================================================================
# STEP 10: EVENT STUDY / PRE-TRENDS TEST
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: EVENT STUDY - TEST FOR PARALLEL TRENDS")
print("=" * 80)

# Create year-specific treatment effects (relative to 2011)
df_sample['treat_2006'] = df_sample['treat'] * (df_sample['YEAR'] == 2006).astype(int)
df_sample['treat_2007'] = df_sample['treat'] * (df_sample['YEAR'] == 2007).astype(int)
df_sample['treat_2008'] = df_sample['treat'] * (df_sample['YEAR'] == 2008).astype(int)
df_sample['treat_2009'] = df_sample['treat'] * (df_sample['YEAR'] == 2009).astype(int)
df_sample['treat_2010'] = df_sample['treat'] * (df_sample['YEAR'] == 2010).astype(int)
# 2011 is reference year (omitted)
df_sample['treat_2013'] = df_sample['treat'] * (df_sample['YEAR'] == 2013).astype(int)
df_sample['treat_2014'] = df_sample['treat'] * (df_sample['YEAR'] == 2014).astype(int)
df_sample['treat_2015'] = df_sample['treat'] * (df_sample['YEAR'] == 2015).astype(int)
df_sample['treat_2016'] = df_sample['treat'] * (df_sample['YEAR'] == 2016).astype(int)

event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']

year_dummies_es = pd.get_dummies(df_sample['YEAR'], prefix='year', drop_first=True).astype(float)
state_dummies_es = pd.get_dummies(df_sample['STATEFIP'], prefix='state', drop_first=True).astype(float)
X_es = pd.concat([df_sample[['treat'] + event_vars + controls].astype(float), year_dummies_es, state_dummies_es], axis=1)
X_es = sm.add_constant(X_es)
X_es = X_es.dropna()
y_es = df_sample.loc[X_es.index, 'fulltime'].astype(float)

model_es = sm.OLS(y_es, X_es).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 60)
for var in event_vars:
    coef = model_es.params[var]
    se = model_es.bse[var]
    pval = model_es.pvalues[var]
    print(f"  {var}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# ============================================================================
# STEP 11: STORE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: SUMMARY OF MAIN RESULTS")
print("=" * 80)

# Preferred specification is Model 4 (with year and state FE)
preferred_coef = model4.params['treat_post']
preferred_se = model4.bse['treat_post']
preferred_ci_low = model4.conf_int().loc['treat_post', 0]
preferred_ci_high = model4.conf_int().loc['treat_post', 1]
preferred_pval = model4.pvalues['treat_post']
preferred_n = len(y4)

print(f"\nPREFERRED ESTIMATE (Model 4 with Year and State FE):")
print(f"=" * 60)
print(f"  Effect of DACA eligibility on full-time employment:")
print(f"  Coefficient (treat_post): {preferred_coef:.4f}")
print(f"  Standard Error (robust): {preferred_se:.4f}")
print(f"  95% Confidence Interval: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  p-value: {preferred_pval:.4f}")
print(f"  Sample Size: {preferred_n:,}")
print(f"")
print(f"  Interpretation: DACA eligibility is associated with a")
print(f"  {preferred_coef*100:.2f} percentage point {'increase' if preferred_coef > 0 else 'decrease'}")
print(f"  in the probability of full-time employment.")

# Save results to a dictionary for later use
results = {
    'preferred_estimate': preferred_coef,
    'preferred_se': preferred_se,
    'preferred_ci_low': preferred_ci_low,
    'preferred_ci_high': preferred_ci_high,
    'preferred_pval': preferred_pval,
    'preferred_n': preferred_n,
    'model1_coef': model1.params['treat_post'],
    'model2_coef': model2.params['treat_post'],
    'model3_coef': model3.params['treat_post'],
    'ft_treat_pre': ft_treat_pre,
    'ft_treat_post': ft_treat_post,
    'ft_control_pre': ft_control_pre,
    'ft_control_post': ft_control_post,
    'simple_did': (ft_treat_post - ft_treat_pre) - (ft_control_post - ft_control_pre),
    'robustness_young': model_r1.params['treat_post'],
    'robustness_male': model_r2.params['treat_post'],
    'robustness_female': model_r3.params['treat_post'],
    'robustness_employed': model_r4.params['treat_post']
}

# Save detailed results to CSV for the report
results_df = pd.DataFrame({
    'Specification': ['Model 1: Basic DID', 'Model 2: + Demographics',
                      'Model 3: + Year FE', 'Model 4: + State FE (Preferred)',
                      'Robustness: Age 18-40', 'Robustness: Males Only',
                      'Robustness: Females Only', 'Robustness: Any Employment'],
    'Coefficient': [model1.params['treat_post'], model2.params['treat_post'],
                   model3.params['treat_post'], model4.params['treat_post'],
                   model_r1.params['treat_post'], model_r2.params['treat_post'],
                   model_r3.params['treat_post'], model_r4.params['treat_post']],
    'Std_Error': [model1.bse['treat_post'], model2.bse['treat_post'],
                 model3.bse['treat_post'], model4.bse['treat_post'],
                 model_r1.bse['treat_post'], model_r2.bse['treat_post'],
                 model_r3.bse['treat_post'], model_r4.bse['treat_post']],
    'p_value': [model1.pvalues['treat_post'], model2.pvalues['treat_post'],
               model3.pvalues['treat_post'], model4.pvalues['treat_post'],
               model_r1.pvalues['treat_post'], model_r2.pvalues['treat_post'],
               model_r3.pvalues['treat_post'], model_r4.pvalues['treat_post']],
    'N': [len(model1.resid), len(model2.resid), len(model3.resid), len(model4.resid),
          len(model_r1.resid), len(model_r2.resid), len(model_r3.resid), len(model_r4.resid)]
})

results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# Event study results
event_df = pd.DataFrame({
    'Year': [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016],
    'Coefficient': [model_es.params[f'treat_{y}'] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]],
    'Std_Error': [model_es.bse[f'treat_{y}'] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]],
    'CI_Low': [model_es.conf_int().loc[f'treat_{y}', 0] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]],
    'CI_High': [model_es.conf_int().loc[f'treat_{y}', 1] for y in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]]
})
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Descriptive statistics table
desc_stats = df_sample.groupby('treat').agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'educ_less_hs': 'mean',
    'educ_hs_grad': 'mean',
    'educ_some_college': 'mean',
    'educ_college': 'mean',
    'PERWT': 'sum'
}).round(3)
desc_stats.columns = ['Full-Time Emp Rate', 'Employment Rate', 'Mean Age',
                      'Female Share', 'Married Share', 'Less than HS',
                      'HS Graduate', 'Some College', 'College+', 'Weighted N']
desc_stats.to_csv('descriptive_statistics.csv')
print("Descriptive statistics saved to descriptive_statistics.csv")

# Save full summary by group and period
summary_by_group = df_sample.groupby(['treat', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean'
}).round(4)
summary_by_group.to_csv('summary_by_group.csv')
print("Summary by group saved to summary_by_group.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

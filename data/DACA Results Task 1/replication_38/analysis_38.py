"""
DACA Impact on Full-Time Employment - Replication Analysis
============================================================
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of DACA eligibility on the
probability of full-time employment (35+ hours/week)?

Author: Replication 38
Data: American Community Survey 2006-2016 via IPUMS
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_38")

print("="*70)
print("DACA Impact on Full-Time Employment - Replication Analysis")
print("="*70)

#############################################################################
# STEP 1: LOAD DATA
#############################################################################
print("\n[1] Loading ACS data...")

# Load the data - use chunks for large file
chunks = []
chunksize = 500000

for chunk in pd.read_csv('data/data.csv', chunksize=chunksize, low_memory=False):
    # Pre-filter to reduce memory: Hispanic Mexican only, born in Mexico
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &  # Mexican Hispanic
        (chunk['BPL'] == 200)     # Born in Mexico
    ]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"   Loaded {len(df):,} observations (Hispanic-Mexican, Mexico-born)")

#############################################################################
# STEP 2: DEFINE KEY VARIABLES
#############################################################################
print("\n[2] Creating analysis variables...")

# Basic cleaning
df = df.copy()

# Year of survey
df['year'] = df['YEAR'].astype(int)

# Create post-DACA indicator (2013-2016)
# Exclude 2012 due to mid-year implementation
df['post'] = (df['year'] >= 2013).astype(int)

# Birth year
df['birthyr'] = df['BIRTHYR'].astype(int)

# Age at survey
df['age'] = df['AGE'].astype(int)

# Sex (1=male, 2=female)
df['female'] = (df['SEX'] == 2).astype(int)

# Immigration year
df['yrimmig'] = df['YRIMMIG'].replace(0, np.nan)

# Years in US at time of survey
df['yrs_in_us'] = df['year'] - df['yrimmig']

# Age at arrival
df['age_at_arrival'] = df['yrimmig'] - df['birthyr']

# Citizenship status
# CITIZEN: 0=N/A, 1=Born abroad of US parents, 2=Naturalized, 3=Not a citizen
df['noncitizen'] = (df['CITIZEN'] == 3).astype(int)

# Education (general version)
# 0=N/A, 1=No schooling, 2=1-4 grades, 3=5-8 grades, 4=9th grade,
# 5=10th grade, 6=11th grade, 7=12th grade/no diploma, 8=HS diploma/GED,
# 9=Some college, 10=Associate's, 11=Bachelor's, 12+=Graduate
df['educ'] = df['EDUC'].astype(int)
df['hs_or_more'] = (df['educ'] >= 6).astype(int)
df['college'] = (df['educ'] >= 9).astype(int)
df['bachelors'] = (df['educ'] >= 10).astype(int)

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# State FIPS code
df['statefip'] = df['STATEFIP'].astype(int)

# Person weight
df['perwt'] = df['PERWT'].astype(float)

#############################################################################
# STEP 3: DEFINE OUTCOME VARIABLE
#############################################################################
print("\n[3] Defining outcome: Full-time employment...")

# Usual hours worked per week
df['uhrswork'] = df['UHRSWORK'].fillna(0).astype(int)

# Employment status (1=Employed, 2=Unemployed, 3=NILF)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

# Full-time employment: usually works 35+ hours per week
df['fulltime'] = (df['uhrswork'] >= 35).astype(int)

# Alternative: Full-time conditional on being employed
df['fulltime_if_emp'] = np.where(df['employed'] == 1,
                                  (df['uhrswork'] >= 35).astype(int),
                                  np.nan)

#############################################################################
# STEP 4: DEFINE DACA ELIGIBILITY
#############################################################################
print("\n[4] Defining DACA eligibility criteria...")

# DACA Eligibility Requirements:
# 1. Arrived before 16th birthday
# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 3. Continuously present since June 15, 2007 (arrived by 2007)
# 4. Non-citizen (not naturalized)
# 5. Hispanic-Mexican born in Mexico (already filtered)

# Calculate age as of June 15, 2012
# Using birth quarter for precision when available
df['birthqtr'] = df['BIRTHQTR'].fillna(0).astype(int)

# Age on June 15, 2012
# If born in Q1-Q2 (Jan-June), already had birthday by June 15
# If born in Q3-Q4 (July-Dec), had not yet had birthday by June 15
df['age_june_2012'] = 2012 - df['birthyr']
# Adjust for those born after June 15
df.loc[df['birthqtr'].isin([3, 4]), 'age_june_2012'] -= 1

# DACA eligibility components
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)
df['under_31_2012'] = (df['age_june_2012'] < 31).astype(int)  # Born after June 15, 1981
df['in_us_by_2007'] = (df['yrimmig'] <= 2007).astype(int)

# Combined DACA eligibility
df['daca_eligible'] = (
    (df['noncitizen'] == 1) &
    (df['arrived_before_16'] == 1) &
    (df['under_31_2012'] == 1) &
    (df['in_us_by_2007'] == 1)
).astype(int)

# Treatment indicator
df['treat'] = df['daca_eligible']

print(f"   DACA eligible: {df['daca_eligible'].sum():,} observations")

#############################################################################
# STEP 5: DEFINE SAMPLE RESTRICTIONS
#############################################################################
print("\n[5] Applying sample restrictions...")

# Restrict to working-age adults (18-55 at time of survey)
# Using 55 as upper bound to avoid retirement effects
df_analysis = df[
    (df['age'] >= 18) &
    (df['age'] <= 55) &
    (df['noncitizen'] == 1) &  # Non-citizens only (potentially undocumented)
    (df['yrimmig'].notna()) &  # Valid immigration year
    (df['year'] != 2012)       # Exclude 2012 (treatment timing ambiguous)
].copy()

print(f"   After restrictions: {len(df_analysis):,} observations")

# Further restrict to those with valid arrival information
df_analysis = df_analysis[df_analysis['age_at_arrival'].notna()].copy()
print(f"   With valid arrival info: {len(df_analysis):,} observations")

#############################################################################
# STEP 6: CREATE COMPARISON GROUP
#############################################################################
print("\n[6] Defining treatment and control groups...")

# Control group: Similar individuals NOT eligible for DACA
# Primary approach: Age-ineligible (31+ in 2012) but otherwise similar
# They arrived before 16 and by 2007, but are too old

df_analysis['control_age_inelig'] = (
    (df_analysis['noncitizen'] == 1) &
    (df_analysis['arrived_before_16'] == 1) &
    (df_analysis['in_us_by_2007'] == 1) &
    (df_analysis['under_31_2012'] == 0)  # Too old
).astype(int)

# Alternative control: Arrived after 2007 (too recently)
df_analysis['control_recent_arrival'] = (
    (df_analysis['noncitizen'] == 1) &
    (df_analysis['in_us_by_2007'] == 0) &  # Arrived after 2007
    (df_analysis['under_31_2012'] == 1)
).astype(int)

# For main analysis, use age-based control
# Restrict to treatment or control group
df_main = df_analysis[
    (df_analysis['treat'] == 1) | (df_analysis['control_age_inelig'] == 1)
].copy()

print(f"   Treatment (DACA eligible): {df_main['treat'].sum():,}")
print(f"   Control (age ineligible): {df_main['control_age_inelig'].sum():,}")
print(f"   Total for DiD analysis: {len(df_main):,}")

#############################################################################
# STEP 7: DESCRIPTIVE STATISTICS
#############################################################################
print("\n[7] Generating descriptive statistics...")

def weighted_stats(data, var, weight='perwt'):
    """Calculate weighted mean and std"""
    w = data[weight]
    v = data[var]
    valid = ~(v.isna() | w.isna())
    w = w[valid]
    v = v[valid]
    wmean = np.average(v, weights=w)
    wvar = np.average((v - wmean)**2, weights=w)
    return wmean, np.sqrt(wvar)

# Pre/Post split
pre = df_main[df_main['post'] == 0]
post = df_main[df_main['post'] == 1]

# Treatment/Control split
treat = df_main[df_main['treat'] == 1]
control = df_main[df_main['treat'] == 0]

desc_stats = {}

# By treatment status
for group_name, group_data in [('Treatment', treat), ('Control', control)]:
    stats_dict = {}
    for var in ['age', 'female', 'married', 'hs_or_more', 'yrs_in_us',
                'fulltime', 'employed', 'uhrswork']:
        mean, std = weighted_stats(group_data, var)
        stats_dict[var] = {'mean': mean, 'std': std}
    stats_dict['n'] = len(group_data)
    desc_stats[group_name] = stats_dict

print("\nDescriptive Statistics (Weighted):")
print("-" * 60)
print(f"{'Variable':<20} {'Treatment':>15} {'Control':>15}")
print("-" * 60)
for var in ['age', 'female', 'married', 'hs_or_more', 'yrs_in_us', 'fulltime', 'employed']:
    t_mean = desc_stats['Treatment'][var]['mean']
    c_mean = desc_stats['Control'][var]['mean']
    print(f"{var:<20} {t_mean:>15.3f} {c_mean:>15.3f}")
print(f"{'N':<20} {desc_stats['Treatment']['n']:>15,} {desc_stats['Control']['n']:>15,}")

#############################################################################
# STEP 8: DIFFERENCE-IN-DIFFERENCES ESTIMATION
#############################################################################
print("\n[8] Running Difference-in-Differences analysis...")

# Create interaction term
df_main['treat_post'] = df_main['treat'] * df_main['post']

# Create year dummies
df_main['year_cat'] = pd.Categorical(df_main['year'])
year_dummies = pd.get_dummies(df_main['year_cat'], prefix='year', drop_first=True)
df_main = pd.concat([df_main, year_dummies], axis=1)

# Create state dummies for fixed effects
df_main['state_cat'] = pd.Categorical(df_main['statefip'])

# Model 1: Basic DiD (no controls)
print("\nModel 1: Basic DiD")
model1 = smf.wls('fulltime ~ treat + post + treat_post',
                  data=df_main, weights=df_main['perwt']).fit(
                  cov_type='cluster', cov_kwds={'groups': df_main['statefip']})
print(f"   DiD estimate: {model1.params['treat_post']:.4f}")
print(f"   Std Error: {model1.bse['treat_post']:.4f}")
print(f"   P-value: {model1.pvalues['treat_post']:.4f}")

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ treat + post + treat_post + age + I(age**2) + female + married + hs_or_more + college',
                  data=df_main, weights=df_main['perwt']).fit(
                  cov_type='cluster', cov_kwds={'groups': df_main['statefip']})
print(f"   DiD estimate: {model2.params['treat_post']:.4f}")
print(f"   Std Error: {model2.bse['treat_post']:.4f}")
print(f"   P-value: {model2.pvalues['treat_post']:.4f}")

# Model 3: DiD with controls and state fixed effects
print("\nModel 3: DiD with controls and state FE")
model3 = smf.wls('fulltime ~ treat + post + treat_post + age + I(age**2) + female + married + hs_or_more + college + C(statefip)',
                  data=df_main, weights=df_main['perwt']).fit(
                  cov_type='cluster', cov_kwds={'groups': df_main['statefip']})
print(f"   DiD estimate: {model3.params['treat_post']:.4f}")
print(f"   Std Error: {model3.bse['treat_post']:.4f}")
print(f"   P-value: {model3.pvalues['treat_post']:.4f}")

# Model 4: DiD with controls, state FE, and year FE (PREFERRED)
print("\nModel 4: DiD with controls, state FE, and year FE (PREFERRED)")
model4 = smf.wls('fulltime ~ treat + treat_post + age + I(age**2) + female + married + hs_or_more + college + C(statefip) + C(year)',
                  data=df_main, weights=df_main['perwt']).fit(
                  cov_type='cluster', cov_kwds={'groups': df_main['statefip']})
print(f"   DiD estimate: {model4.params['treat_post']:.4f}")
print(f"   Std Error: {model4.bse['treat_post']:.4f}")
print(f"   P-value: {model4.pvalues['treat_post']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['treat_post', 0]:.4f}, {model4.conf_int().loc['treat_post', 1]:.4f}]")

# Store preferred estimate
preferred_estimate = {
    'coefficient': model4.params['treat_post'],
    'std_error': model4.bse['treat_post'],
    'pvalue': model4.pvalues['treat_post'],
    'ci_lower': model4.conf_int().loc['treat_post', 0],
    'ci_upper': model4.conf_int().loc['treat_post', 1],
    'n': len(df_main)
}

#############################################################################
# STEP 9: ROBUSTNESS CHECKS
#############################################################################
print("\n[9] Running robustness checks...")

# Robustness 1: Employment (extensive margin)
print("\nRobustness 1: Employment (extensive margin)")
model_emp = smf.wls('employed ~ treat + treat_post + age + I(age**2) + female + married + hs_or_more + college + C(statefip) + C(year)',
                     data=df_main, weights=df_main['perwt']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_main['statefip']})
print(f"   DiD estimate: {model_emp.params['treat_post']:.4f}")
print(f"   Std Error: {model_emp.bse['treat_post']:.4f}")
print(f"   P-value: {model_emp.pvalues['treat_post']:.4f}")

# Robustness 2: Hours worked (intensive margin)
print("\nRobustness 2: Hours worked (intensive margin)")
df_working = df_main[df_main['employed'] == 1].copy()
model_hrs = smf.wls('uhrswork ~ treat + treat_post + age + I(age**2) + female + married + hs_or_more + college + C(statefip) + C(year)',
                     data=df_working, weights=df_working['perwt']).fit(
                     cov_type='cluster', cov_kwds={'groups': df_working['statefip']})
print(f"   DiD estimate: {model_hrs.params['treat_post']:.4f}")
print(f"   Std Error: {model_hrs.bse['treat_post']:.4f}")
print(f"   P-value: {model_hrs.pvalues['treat_post']:.4f}")

# Robustness 3: Alternative control group (recent arrivals)
print("\nRobustness 3: Alternative control (recent arrivals)")
df_alt = df_analysis[
    (df_analysis['treat'] == 1) | (df_analysis['control_recent_arrival'] == 1)
].copy()
df_alt['treat_post'] = df_alt['treat'] * df_alt['post']

if len(df_alt) > 100:
    model_alt = smf.wls('fulltime ~ treat + treat_post + age + I(age**2) + female + married + hs_or_more + college + C(statefip) + C(year)',
                         data=df_alt, weights=df_alt['perwt']).fit(
                         cov_type='cluster', cov_kwds={'groups': df_alt['statefip']})
    print(f"   DiD estimate: {model_alt.params['treat_post']:.4f}")
    print(f"   Std Error: {model_alt.bse['treat_post']:.4f}")
    print(f"   P-value: {model_alt.pvalues['treat_post']:.4f}")
else:
    print("   Insufficient observations for alternative control")

# Robustness 4: Subsample by gender
print("\nRobustness 4: By gender")
for sex, sex_name in [(0, 'Male'), (1, 'Female')]:
    df_sex = df_main[df_main['female'] == sex].copy()
    model_sex = smf.wls('fulltime ~ treat + treat_post + age + I(age**2) + married + hs_or_more + college + C(statefip) + C(year)',
                         data=df_sex, weights=df_sex['perwt']).fit(
                         cov_type='cluster', cov_kwds={'groups': df_sex['statefip']})
    print(f"   {sex_name}: DiD = {model_sex.params['treat_post']:.4f} (SE: {model_sex.bse['treat_post']:.4f})")

#############################################################################
# STEP 10: EVENT STUDY
#############################################################################
print("\n[10] Event study analysis...")

# Create year-specific treatment effects
years = sorted(df_main['year'].unique())
base_year = 2011  # Reference year (last pre-treatment year)

event_study_results = []
for yr in years:
    if yr == base_year:
        event_study_results.append({
            'year': yr,
            'coef': 0,
            'se': 0,
            'ci_lower': 0,
            'ci_upper': 0
        })
    else:
        df_main[f'treat_year_{yr}'] = (df_main['treat'] * (df_main['year'] == yr)).astype(int)

# Run event study regression
year_interact_vars = ' + '.join([f'treat_year_{yr}' for yr in years if yr != base_year])
formula = f'fulltime ~ treat + {year_interact_vars} + age + I(age**2) + female + married + hs_or_more + college + C(statefip) + C(year)'

model_event = smf.wls(formula, data=df_main, weights=df_main['perwt']).fit(
    cov_type='cluster', cov_kwds={'groups': df_main['statefip']})

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for yr in years:
    if yr == base_year:
        print(f"   {yr}: 0.0000 (reference)")
    else:
        param = f'treat_year_{yr}'
        coef = model_event.params[param]
        se = model_event.bse[param]
        print(f"   {yr}: {coef:.4f} (SE: {se:.4f})")
        event_study_results.append({
            'year': yr,
            'coef': coef,
            'se': se,
            'ci_lower': model_event.conf_int().loc[param, 0],
            'ci_upper': model_event.conf_int().loc[param, 1]
        })

#############################################################################
# STEP 11: SAVE RESULTS
#############################################################################
print("\n[11] Saving results...")

# Create results summary
results_summary = {
    'preferred_estimate': preferred_estimate,
    'descriptive_stats': desc_stats,
    'event_study': event_study_results,
    'robustness': {
        'employment': {
            'coef': model_emp.params['treat_post'],
            'se': model_emp.bse['treat_post']
        },
        'hours': {
            'coef': model_hrs.params['treat_post'],
            'se': model_hrs.bse['treat_post']
        }
    }
}

# Save to JSON
with open('results_38.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print("\nResults saved to results_38.json")

#############################################################################
# STEP 12: SUMMARY OUTPUT
#############################################################################
print("\n" + "="*70)
print("SUMMARY OF RESULTS")
print("="*70)

print(f"\nPreferred Estimate (Model 4: Full controls, State FE, Year FE)")
print("-"*60)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"   Coefficient: {preferred_estimate['coefficient']:.4f}")
print(f"   Standard Error: {preferred_estimate['std_error']:.4f}")
print(f"   95% CI: [{preferred_estimate['ci_lower']:.4f}, {preferred_estimate['ci_upper']:.4f}]")
print(f"   P-value: {preferred_estimate['pvalue']:.4f}")
print(f"   Sample Size: {preferred_estimate['n']:,}")

print("\nInterpretation:")
if preferred_estimate['pvalue'] < 0.05:
    direction = "increased" if preferred_estimate['coefficient'] > 0 else "decreased"
    print(f"   DACA eligibility is associated with a statistically significant")
    print(f"   {abs(preferred_estimate['coefficient']*100):.1f} percentage point {direction} in the")
    print(f"   probability of full-time employment.")
else:
    print(f"   The effect of DACA eligibility on full-time employment is not")
    print(f"   statistically significant at the 5% level.")

print("\n" + "="*70)
print("Analysis complete.")
print("="*70)

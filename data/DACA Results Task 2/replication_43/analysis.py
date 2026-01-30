"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican individuals born in Mexico.

Treatment: Ages 26-30 at time of DACA (June 15, 2012)
Control: Ages 31-35 at time of DACA (otherwise eligible but for age)
Outcome: Full-time employment (usually working 35+ hours/week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load Data
# =============================================================================
print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 70)
print("\nStep 1: Loading data...")

# Load the ACS data
data_path = "data/data.csv"
df = pd.read_csv(data_path)

print(f"Total observations loaded: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# STEP 2: Define Sample Selection Criteria
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Defining sample selection criteria...")
print("=" * 70)

# Save initial count
initial_n = len(df)

# DACA eligibility criteria (adapted from instructions):
# 1. Hispanic-Mexican ethnicity (HISPAN == 1)
# 2. Born in Mexico (BPL == 200)
# 3. Not a citizen (CITIZEN == 3, indicating not a citizen)
# 4. Immigrated before age 16 (requires BIRTHYR and YRIMMIG)
# 5. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
# 6. Present in US on June 15, 2012

# Note: We cannot distinguish documented vs undocumented. Per instructions,
# assume anyone who is not a citizen and who has not received immigration papers
# is undocumented for DACA purposes. CITIZEN==3 means "Not a citizen".

# Filter 1: Hispanic-Mexican
df_sample = df[df['HISPAN'] == 1].copy()
print(f"\nAfter Hispanic-Mexican filter: {len(df_sample):,} ({100*len(df_sample)/initial_n:.1f}%)")

# Filter 2: Born in Mexico (BPL == 200)
df_sample = df_sample[df_sample['BPL'] == 200].copy()
print(f"After born in Mexico filter: {len(df_sample):,}")

# Filter 3: Not a citizen (CITIZEN == 3)
# Per instructions: assume anyone who is not a citizen and who has not received
# immigration papers is undocumented
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"After non-citizen filter: {len(df_sample):,}")

# Filter 4: Arrived before age 16 (DACA requirement)
# Age at immigration = YRIMMIG - BIRTHYR
# We need YRIMMIG > 0 (valid immigration year)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
df_sample['age_at_immigration'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_immigration'] < 16].copy()
print(f"After arrived before age 16 filter: {len(df_sample):,}")

# Filter 5: Continuous presence since June 15, 2007 (arrived by 2007)
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()
print(f"After continuous presence (YRIMMIG <= 2007) filter: {len(df_sample):,}")

# =============================================================================
# STEP 3: Define Treatment and Control Groups Based on Age at DACA
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Defining treatment and control groups...")
print("=" * 70)

# DACA announced June 15, 2012
# Treatment group: Ages 26-30 on June 15, 2012
# Control group: Ages 31-35 on June 15, 2012 (would have been eligible but for age)

# Calculate age on June 15, 2012
# Person's age in 2012 = 2012 - BIRTHYR
# But we need to account for birth quarter
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# For simplicity, we use age as of mid-year 2012

df_sample['age_june_2012'] = 2012 - df_sample['BIRTHYR']

# Adjustment for birth quarter (conservative approach):
# If born in Q3 or Q4 (July onwards), they hadn't had birthday by June 15, so subtract 1
# Actually, BIRTHQTR 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
# If BIRTHQTR >= 3 (July or later), subtract 1 from age
df_sample.loc[df_sample['BIRTHQTR'] >= 3, 'age_june_2012'] -= 1

print(f"\nAge distribution on June 15, 2012:")
age_dist = df_sample['age_june_2012'].value_counts().sort_index()
print(age_dist[(age_dist.index >= 20) & (age_dist.index <= 40)])

# Define treatment group (ages 26-30) and control group (ages 31-35)
df_sample['treat_group'] = np.where(
    (df_sample['age_june_2012'] >= 26) & (df_sample['age_june_2012'] <= 30), 1,
    np.where((df_sample['age_june_2012'] >= 31) & (df_sample['age_june_2012'] <= 35), 0, np.nan)
)

# Keep only treatment and control groups
df_analysis = df_sample[df_sample['treat_group'].notna()].copy()
print(f"\nObservations in treatment group (ages 26-30): {(df_analysis['treat_group']==1).sum():,}")
print(f"Observations in control group (ages 31-35): {(df_analysis['treat_group']==0).sum():,}")
print(f"Total analysis sample: {len(df_analysis):,}")

# =============================================================================
# STEP 4: Define Pre and Post Periods
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Defining pre and post treatment periods...")
print("=" * 70)

# DACA implemented in 2012
# Pre-period: 2006-2011
# Post-period: 2013-2016 (as specified in instructions)
# Exclude 2012 because we can't distinguish before/after June 15, 2012

df_analysis['post'] = np.where(df_analysis['YEAR'] >= 2013, 1,
                               np.where(df_analysis['YEAR'] <= 2011, 0, np.nan))

# Remove 2012 observations (ambiguous period)
df_analysis = df_analysis[df_analysis['post'].notna()].copy()

print(f"\nPre-period (2006-2011): {(df_analysis['post']==0).sum():,}")
print(f"Post-period (2013-2016): {(df_analysis['post']==1).sum():,}")
print(f"Final analysis sample: {len(df_analysis):,}")

# Year distribution
print(f"\nYear distribution:")
print(df_analysis['YEAR'].value_counts().sort_index())

# =============================================================================
# STEP 5: Define Outcome Variable
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Defining outcome variable (full-time employment)...")
print("=" * 70)

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: usual hours worked per week

# First, check EMPSTAT and UHRSWORK distributions
print(f"\nEmployment status distribution (EMPSTAT):")
print(df_analysis['EMPSTAT'].value_counts().sort_index())

print(f"\nUSAL hours worked distribution (UHRSWORK):")
print(df_analysis['UHRSWORK'].describe())

# Define full-time employment: UHRSWORK >= 35
# Note: People not in labor force may have UHRSWORK = 0
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"\nFull-time employment rate: {df_analysis['fulltime'].mean()*100:.2f}%")

# =============================================================================
# STEP 6: Create Covariates
# =============================================================================
print("\n" + "=" * 70)
print("Step 6: Creating covariates...")
print("=" * 70)

# Age (at time of survey)
df_analysis['age'] = df_analysis['AGE']

# Sex (1=Male, 2=Female)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

# Marital status (1=Married with spouse present)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)

# Education categories
# EDUC: 0=N/A, 1=Nursery-4, 2=5-8, 3=9, 4=10, 5=11, 6=12, 7-11=college
df_analysis['hs_or_more'] = (df_analysis['EDUC'] >= 6).astype(int)
df_analysis['some_college'] = (df_analysis['EDUC'] >= 7).astype(int)

# Number of children
df_analysis['nchild'] = df_analysis['NCHILD']

# State fixed effects
df_analysis['state'] = df_analysis['STATEFIP']

# Years in US
df_analysis['yrs_in_us'] = df_analysis['YRSUSA1']

print("Covariates created: age, female, married, hs_or_more, some_college, nchild, state, yrs_in_us")

# =============================================================================
# STEP 7: Summary Statistics
# =============================================================================
print("\n" + "=" * 70)
print("Step 7: Summary statistics...")
print("=" * 70)

# Summary stats by group and period
summary_vars = ['fulltime', 'age', 'female', 'married', 'hs_or_more', 'some_college', 'nchild']

print("\n" + "-" * 50)
print("Panel A: Pre-period (2006-2011)")
print("-" * 50)
pre_data = df_analysis[df_analysis['post'] == 0]
for var in summary_vars:
    treat_mean = pre_data[pre_data['treat_group']==1][var].mean()
    ctrl_mean = pre_data[pre_data['treat_group']==0][var].mean()
    print(f"{var:15s}: Treat={treat_mean:.3f}, Control={ctrl_mean:.3f}, Diff={treat_mean-ctrl_mean:.3f}")

print("\n" + "-" * 50)
print("Panel B: Post-period (2013-2016)")
print("-" * 50)
post_data = df_analysis[df_analysis['post'] == 1]
for var in summary_vars:
    treat_mean = post_data[post_data['treat_group']==1][var].mean()
    ctrl_mean = post_data[post_data['treat_group']==0][var].mean()
    print(f"{var:15s}: Treat={treat_mean:.3f}, Control={ctrl_mean:.3f}, Diff={treat_mean-ctrl_mean:.3f}")

# =============================================================================
# STEP 8: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 70)
print("Step 8: Difference-in-Differences Analysis")
print("=" * 70)

# Create interaction term
df_analysis['treat_post'] = df_analysis['treat_group'] * df_analysis['post']

# Use person weights
df_analysis['weight'] = df_analysis['PERWT']

# Model 1: Basic DiD (no covariates)
print("\n" + "-" * 50)
print("Model 1: Basic DiD (no covariates)")
print("-" * 50)

model1 = smf.wls('fulltime ~ treat_group + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['weight']).fit(cov_type='HC1')
print(model1.summary())

# Extract DiD coefficient
did_coef = model1.params['treat_post']
did_se = model1.bse['treat_post']
did_pval = model1.pvalues['treat_post']
did_ci_low = model1.conf_int().loc['treat_post', 0]
did_ci_high = model1.conf_int().loc['treat_post', 1]

print(f"\nDiD Estimate: {did_coef:.4f}")
print(f"Standard Error: {did_se:.4f}")
print(f"95% CI: [{did_ci_low:.4f}, {did_ci_high:.4f}]")
print(f"P-value: {did_pval:.4f}")

# Model 2: DiD with demographic covariates
print("\n" + "-" * 50)
print("Model 2: DiD with demographic covariates")
print("-" * 50)

model2 = smf.wls('fulltime ~ treat_group + post + treat_post + female + married + hs_or_more + nchild',
                  data=df_analysis,
                  weights=df_analysis['weight']).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n" + "-" * 50)
print("Model 3: DiD with year fixed effects")
print("-" * 50)

df_analysis['year_factor'] = pd.Categorical(df_analysis['YEAR'])
model3 = smf.wls('fulltime ~ treat_group + treat_post + female + married + hs_or_more + nchild + C(YEAR)',
                  data=df_analysis,
                  weights=df_analysis['weight']).fit(cov_type='HC1')

# Print key coefficients
print(f"\nDiD Estimate (treat_post): {model3.params['treat_post']:.4f}")
print(f"Standard Error: {model3.bse['treat_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['treat_post', 0]:.4f}, {model3.conf_int().loc['treat_post', 1]:.4f}]")

# Model 4: DiD with state and year fixed effects (PREFERRED SPECIFICATION)
print("\n" + "-" * 50)
print("Model 4: DiD with state and year fixed effects (PREFERRED)")
print("-" * 50)

model4 = smf.wls('fulltime ~ treat_group + treat_post + female + married + hs_or_more + nchild + C(YEAR) + C(state)',
                  data=df_analysis,
                  weights=df_analysis['weight']).fit(cov_type='HC1')

# Print key coefficients
preferred_coef = model4.params['treat_post']
preferred_se = model4.bse['treat_post']
preferred_ci = model4.conf_int().loc['treat_post']
preferred_pval = model4.pvalues['treat_post']

print(f"\nPREFERRED ESTIMATE")
print(f"DiD Estimate (treat_post): {preferred_coef:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"P-value: {preferred_pval:.4f}")

# =============================================================================
# STEP 9: Robustness Checks
# =============================================================================
print("\n" + "=" * 70)
print("Step 9: Robustness Checks")
print("=" * 70)

# Robustness 1: Alternative age bandwidth (24-32 vs 33-37)
print("\n" + "-" * 50)
print("Robustness 1: Narrower age bandwidth (27-29 vs 32-34)")
print("-" * 50)

df_robust1 = df_sample.copy()
df_robust1['treat_group_narrow'] = np.where(
    (df_robust1['age_june_2012'] >= 27) & (df_robust1['age_june_2012'] <= 29), 1,
    np.where((df_robust1['age_june_2012'] >= 32) & (df_robust1['age_june_2012'] <= 34), 0, np.nan)
)
df_robust1 = df_robust1[df_robust1['treat_group_narrow'].notna()].copy()
df_robust1['post'] = np.where(df_robust1['YEAR'] >= 2013, 1,
                              np.where(df_robust1['YEAR'] <= 2011, 0, np.nan))
df_robust1 = df_robust1[df_robust1['post'].notna()].copy()
df_robust1['fulltime'] = (df_robust1['UHRSWORK'] >= 35).astype(int)
df_robust1['treat_post'] = df_robust1['treat_group_narrow'] * df_robust1['post']
df_robust1['female'] = (df_robust1['SEX'] == 2).astype(int)
df_robust1['married'] = (df_robust1['MARST'] == 1).astype(int)
df_robust1['hs_or_more'] = (df_robust1['EDUC'] >= 6).astype(int)
df_robust1['nchild'] = df_robust1['NCHILD']
df_robust1['state'] = df_robust1['STATEFIP']

model_r1 = smf.wls('fulltime ~ treat_group_narrow + treat_post + female + married + hs_or_more + nchild + C(YEAR) + C(state)',
                   data=df_robust1,
                   weights=df_robust1['PERWT']).fit(cov_type='HC1')
print(f"N = {len(df_robust1):,}")
print(f"DiD Estimate: {model_r1.params['treat_post']:.4f}")
print(f"SE: {model_r1.bse['treat_post']:.4f}")
print(f"95% CI: [{model_r1.conf_int().loc['treat_post', 0]:.4f}, {model_r1.conf_int().loc['treat_post', 1]:.4f}]")

# Robustness 2: Placebo test (pre-treatment period only)
print("\n" + "-" * 50)
print("Robustness 2: Placebo test (2009-2011 as 'post')")
print("-" * 50)

df_placebo = df_analysis[df_analysis['YEAR'] <= 2011].copy()
df_placebo['post_placebo'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['treat_post_placebo'] = df_placebo['treat_group'] * df_placebo['post_placebo']

model_placebo = smf.wls('fulltime ~ treat_group + post_placebo + treat_post_placebo + female + married + hs_or_more + nchild + C(YEAR) + C(state)',
                        data=df_placebo,
                        weights=df_placebo['PERWT']).fit(cov_type='HC1')
print(f"N = {len(df_placebo):,}")
print(f"Placebo DiD Estimate: {model_placebo.params['treat_post_placebo']:.4f}")
print(f"SE: {model_placebo.bse['treat_post_placebo']:.4f}")
print(f"95% CI: [{model_placebo.conf_int().loc['treat_post_placebo', 0]:.4f}, {model_placebo.conf_int().loc['treat_post_placebo', 1]:.4f}]")
print(f"P-value: {model_placebo.pvalues['treat_post_placebo']:.4f}")

# Robustness 3: By sex
print("\n" + "-" * 50)
print("Robustness 3: Heterogeneity by sex")
print("-" * 50)

# Males
df_male = df_analysis[df_analysis['female'] == 0].copy()
model_male = smf.wls('fulltime ~ treat_group + treat_post + married + hs_or_more + nchild + C(YEAR) + C(state)',
                     data=df_male,
                     weights=df_male['weight']).fit(cov_type='HC1')
print(f"Males (N={len(df_male):,}): DiD={model_male.params['treat_post']:.4f}, SE={model_male.bse['treat_post']:.4f}")

# Females
df_female = df_analysis[df_analysis['female'] == 1].copy()
model_female = smf.wls('fulltime ~ treat_group + treat_post + married + hs_or_more + nchild + C(YEAR) + C(state)',
                       data=df_female,
                       weights=df_female['weight']).fit(cov_type='HC1')
print(f"Females (N={len(df_female):,}): DiD={model_female.params['treat_post']:.4f}, SE={model_female.bse['treat_post']:.4f}")

# Robustness 4: Event study (year-by-year effects)
print("\n" + "-" * 50)
print("Robustness 4: Event study (year-by-year effects)")
print("-" * 50)

# Create year interactions (reference year: 2011)
df_event = df_analysis.copy()
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in years:
    df_event[f'treat_year_{year}'] = df_event['treat_group'] * (df_event['YEAR'] == year).astype(int)

# Create formula
year_terms = ' + '.join([f'treat_year_{y}' for y in years])
event_formula = f'fulltime ~ treat_group + {year_terms} + female + married + hs_or_more + nchild + C(YEAR) + C(state)'

model_event = smf.wls(event_formula,
                      data=df_event,
                      weights=df_event['weight']).fit(cov_type='HC1')

print("Year-specific treatment effects (reference: 2011):")
for year in years:
    coef = model_event.params[f'treat_year_{year}']
    se = model_event.bse[f'treat_year_{year}']
    print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

# =============================================================================
# STEP 10: Calculate 2x2 DiD manually for verification
# =============================================================================
print("\n" + "=" * 70)
print("Step 10: Manual 2x2 DiD calculation (verification)")
print("=" * 70)

# Calculate weighted means
def weighted_mean(data, var, weight):
    return np.average(data[var], weights=data[weight])

# Pre-period means
treat_pre = weighted_mean(df_analysis[(df_analysis['treat_group']==1) & (df_analysis['post']==0)],
                          'fulltime', 'weight')
ctrl_pre = weighted_mean(df_analysis[(df_analysis['treat_group']==0) & (df_analysis['post']==0)],
                         'fulltime', 'weight')

# Post-period means
treat_post = weighted_mean(df_analysis[(df_analysis['treat_group']==1) & (df_analysis['post']==1)],
                           'fulltime', 'weight')
ctrl_post = weighted_mean(df_analysis[(df_analysis['treat_group']==0) & (df_analysis['post']==1)],
                          'fulltime', 'weight')

# Calculate DiD
did_manual = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)

print(f"\n2x2 Table of Full-Time Employment Rates:")
print("-" * 50)
print(f"{'':20s} {'Pre-Period':>12s} {'Post-Period':>12s} {'Diff':>10s}")
print("-" * 50)
print(f"{'Treatment (26-30)':20s} {treat_pre:>12.4f} {treat_post:>12.4f} {treat_post-treat_pre:>10.4f}")
print(f"{'Control (31-35)':20s} {ctrl_pre:>12.4f} {ctrl_post:>12.4f} {ctrl_post-ctrl_pre:>10.4f}")
print("-" * 50)
print(f"{'Diff-in-Diff':20s} {'':>12s} {'':>12s} {did_manual:>10.4f}")

# =============================================================================
# STEP 11: Save results for report
# =============================================================================
print("\n" + "=" * 70)
print("Step 11: Saving results...")
print("=" * 70)

# Save key results to a file
results = {
    'sample_size': int(len(df_analysis)),
    'n_treatment': int((df_analysis['treat_group']==1).sum()),
    'n_control': int((df_analysis['treat_group']==0).sum()),
    'n_pre': int((df_analysis['post']==0).sum()),
    'n_post': int((df_analysis['post']==1).sum()),

    # Main results
    'model1_coef': float(model1.params['treat_post']),
    'model1_se': float(model1.bse['treat_post']),
    'model1_pval': float(model1.pvalues['treat_post']),

    'model2_coef': float(model2.params['treat_post']),
    'model2_se': float(model2.bse['treat_post']),

    'model3_coef': float(model3.params['treat_post']),
    'model3_se': float(model3.bse['treat_post']),

    'preferred_coef': float(preferred_coef),
    'preferred_se': float(preferred_se),
    'preferred_ci_low': float(preferred_ci[0]),
    'preferred_ci_high': float(preferred_ci[1]),
    'preferred_pval': float(preferred_pval),

    # 2x2 table values
    'treat_pre': float(treat_pre),
    'treat_post_mean': float(treat_post),
    'ctrl_pre': float(ctrl_pre),
    'ctrl_post_mean': float(ctrl_post),
    'did_manual': float(did_manual),

    # Robustness results
    'narrow_bw_coef': float(model_r1.params['treat_post']),
    'narrow_bw_se': float(model_r1.bse['treat_post']),

    'placebo_coef': float(model_placebo.params['treat_post_placebo']),
    'placebo_se': float(model_placebo.bse['treat_post_placebo']),
    'placebo_pval': float(model_placebo.pvalues['treat_post_placebo']),

    'male_coef': float(model_male.params['treat_post']),
    'male_se': float(model_male.bse['treat_post']),
    'female_coef': float(model_female.params['treat_post']),
    'female_se': float(model_female.bse['treat_post']),
}

# Event study results
for year in years:
    results[f'event_{year}_coef'] = float(model_event.params[f'treat_year_{year}'])
    results[f'event_{year}_se'] = float(model_event.bse[f'treat_year_{year}'])

# Summary statistics
summary_stats = {}
for var in ['fulltime', 'age', 'female', 'married', 'hs_or_more', 'nchild']:
    summary_stats[f'{var}_mean'] = float(df_analysis[var].mean())
    summary_stats[f'{var}_std'] = float(df_analysis[var].std())
    summary_stats[f'{var}_treat_mean'] = float(df_analysis[df_analysis['treat_group']==1][var].mean())
    summary_stats[f'{var}_ctrl_mean'] = float(df_analysis[df_analysis['treat_group']==0][var].mean())

results.update(summary_stats)

# Save to file
import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to results.json")

# =============================================================================
# STEP 12: Final Summary
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
PREFERRED ESTIMATE (Model 4: DiD with state and year FE)
=========================================================
Sample Size: {len(df_analysis):,}
Treatment Group (ages 26-30): {(df_analysis['treat_group']==1).sum():,}
Control Group (ages 31-35): {(df_analysis['treat_group']==0).sum():,}

Effect on Full-Time Employment (35+ hours/week):
  Point Estimate: {preferred_coef:.4f} ({preferred_coef*100:.2f} percentage points)
  Standard Error: {preferred_se:.4f}
  95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]
  P-value: {preferred_pval:.4f}

Interpretation:
DACA eligibility is associated with a {abs(preferred_coef)*100:.2f} percentage point
{"increase" if preferred_coef > 0 else "decrease"} in the probability of full-time employment
among Hispanic-Mexican individuals born in Mexico who arrived in the US before age 16.
This effect is {"statistically significant" if preferred_pval < 0.05 else "not statistically significant"} at the 5% level.
""")

print("Analysis complete!")

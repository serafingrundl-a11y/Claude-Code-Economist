"""
DACA Replication Analysis
Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the
United States, what was the causal impact of eligibility for DACA on the probability
of full-time employment (35+ hours/week)?

DACA eligibility criteria:
1. Arrived unlawfully in the US before their 16th birthday
2. Had not yet had their 31st birthday as of June 15, 2012
3. Lived continuously in the US since June 15, 2007
4. Were present in the US on June 15, 2012 and did not have lawful status

Analysis approach: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Output directory
import os
output_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("=" * 70)

# ============================================================================
# STEP 1: LOAD AND FILTER DATA
# ============================================================================
print("\n[1] Loading and filtering data...")

# Load data in chunks to handle large file
chunks = []
chunksize = 1000000

for i, chunk in enumerate(pd.read_csv('data/data.csv', chunksize=chunksize)):
    # Filter to Hispanic-Mexican (HISPAN == 1) and Mexican-born (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    if (i + 1) % 5 == 0:
        print(f"  Processed {(i+1) * chunksize:,} rows...")

df = pd.concat(chunks, ignore_index=True)
print(f"  Total sample after filtering to Hispanic-Mexican, Mexican-born: {len(df):,}")

# ============================================================================
# STEP 2: DEFINE DACA ELIGIBILITY
# ============================================================================
print("\n[2] Defining DACA eligibility criteria...")

# DACA was implemented June 15, 2012
# Eligibility requirements:
# 1. Arrived before age 16 (ageimmig < 16)
# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (YRIMMIG <= 2007)
# 4. Not a citizen (CITIZEN == 3, "Not a citizen")

# Calculate age at immigration
# YRIMMIG is year of immigration
# BIRTHYR is birth year
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# For age calculation as of June 15, 2012:
# Born after June 15, 1981 = under 31
# We use BIRTHYR <= 1981 to be conservative (some born in 1981 would be 30)
# More precisely, need birthyear > 1981 OR (birthyear = 1981 AND birthqtr >= 3)
# For simplicity and conservative estimation, use birthyear >= 1982 for clearly under 31

# Create eligibility indicator
# Eligible if:
# 1. age_at_immig < 16 (arrived before 16th birthday)
# 2. BIRTHYR >= 1982 (under 31 as of June 2012, conservative)
# 3. YRIMMIG <= 2007 (in US since at least 2007)
# 4. CITIZEN == 3 (non-citizen)

# Note: We can only identify likely undocumented status through non-citizenship
# Some naturalized citizens may have been undocumented earlier, but we focus on
# current non-citizens as the treatment group

df['daca_eligible'] = (
    (df['age_at_immig'] >= 0) &  # Valid immigration age
    (df['age_at_immig'] < 16) &   # Arrived before 16
    (df['BIRTHYR'] >= 1982) &     # Under 31 as of June 2012
    (df['YRIMMIG'] <= 2007) &     # In US since at least 2007
    (df['YRIMMIG'] > 0) &         # Valid immigration year
    (df['CITIZEN'] == 3)          # Non-citizen
).astype(int)

print(f"  DACA-eligible individuals: {df['daca_eligible'].sum():,}")
print(f"  Non-eligible individuals: {(1 - df['daca_eligible']).sum():,}")

# ============================================================================
# STEP 3: DEFINE OUTCOME VARIABLE
# ============================================================================
print("\n[3] Defining outcome variable (full-time employment)...")

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: usual hours worked per week
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Also create employed variable (EMPSTAT == 1)
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"  Full-time workers: {df['fulltime'].sum():,} ({100*df['fulltime'].mean():.1f}%)")
print(f"  Employed: {df['employed'].sum():,} ({100*df['employed'].mean():.1f}%)")

# ============================================================================
# STEP 4: DEFINE TREATMENT PERIOD
# ============================================================================
print("\n[4] Defining treatment period...")

# DACA implemented June 2012
# Pre-period: 2006-2011
# Transition: 2012 (cannot distinguish before/after June)
# Post-period: 2013-2016

df['post'] = (df['YEAR'] >= 2013).astype(int)

# For robustness, also create year-specific indicators
for year in range(2006, 2017):
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

print(f"  Pre-DACA observations (2006-2012): {(df['YEAR'] <= 2012).sum():,}")
print(f"  Post-DACA observations (2013-2016): {(df['YEAR'] >= 2013).sum():,}")

# ============================================================================
# STEP 5: CREATE CONTROL VARIABLES
# ============================================================================
print("\n[5] Creating control variables...")

# Age (at time of survey)
df['age'] = df['AGE']
df['age_sq'] = df['AGE'] ** 2

# Sex (1 = male, 2 = female)
df['female'] = (df['SEX'] == 2).astype(int)

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education
# EDUC categories: 0=N/A, 1=No school, 2=N/A, 3=Grade 1-4, 4=Grade 5-8, 5=Grade 9,
# 6=Grade 10, 7=Grade 11, 8=Grade 12, 9=1 yr college, 10=2-3 yrs college, 11=4+ yrs college
df['less_than_hs'] = (df['EDUC'] < 6).astype(int)
df['hs_grad'] = (df['EDUC'].isin([6, 7, 8])).astype(int)
df['some_college'] = (df['EDUC'].isin([9, 10])).astype(int)
df['college_plus'] = (df['EDUC'] >= 11).astype(int)

# Years in US
df['years_in_us'] = df['YRSUSA1']

# Region
df['region'] = df['REGION']

# Metro status
df['metro'] = df['METRO']

# State fixed effects (will use state FIPS)
df['state'] = df['STATEFIP']

print("  Control variables created: age, age_sq, female, married, education dummies, years_in_us")

# ============================================================================
# STEP 6: SAMPLE RESTRICTIONS
# ============================================================================
print("\n[6] Applying sample restrictions...")

# Restrict to working-age adults (16-64)
# This is standard in labor economics
df_analysis = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
print(f"  After restricting to ages 16-64: {len(df_analysis):,}")

# Exclude 2012 (transition year where we can't distinguish pre/post DACA)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"  After excluding 2012: {len(df_analysis):,}")

# Drop observations with missing key variables
df_analysis = df_analysis.dropna(subset=['fulltime', 'daca_eligible', 'YEAR', 'AGE', 'SEX', 'EDUC'])
print(f"  After dropping missing values: {len(df_analysis):,}")

# ============================================================================
# STEP 7: DESCRIPTIVE STATISTICS
# ============================================================================
print("\n[7] Computing descriptive statistics...")

# Summary statistics by treatment/control and pre/post
summary_stats = df_analysis.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'female': 'mean',
    'married': 'mean',
    'less_than_hs': 'mean',
    'hs_grad': 'mean',
    'some_college': 'mean',
    'college_plus': 'mean',
    'years_in_us': 'mean'
})

print("\nSummary Statistics by Group and Period:")
print(summary_stats)

# Save summary statistics
summary_stats.to_csv(os.path.join(output_dir, 'summary_stats.csv'))

# ============================================================================
# STEP 8: DIFFERENCE-IN-DIFFERENCES ANALYSIS
# ============================================================================
print("\n[8] Running Difference-in-Differences analysis...")

# Create interaction term
df_analysis['daca_x_post'] = df_analysis['daca_eligible'] * df_analysis['post']

# Model 1: Simple DiD
print("\n--- Model 1: Simple DiD ---")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with demographic controls ---")
model2 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + age + age_sq + female + married',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})
print(model2.summary())

# Model 3: DiD with demographic and education controls
print("\n--- Model 3: DiD with demographic and education controls ---")
model3 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with state fixed effects ---")
model4 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state)',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})
print(f"DiD coefficient (daca_x_post): {model4.params['daca_x_post']:.4f}")
print(f"Standard error: {model4.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model4.tvalues['daca_x_post']:.4f}")
print(f"p-value: {model4.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:.0f}")

# Model 5: DiD with year fixed effects (preferred specification)
print("\n--- Model 5: DiD with year and state fixed effects (PREFERRED) ---")
model5 = smf.ols('fulltime ~ daca_eligible + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                 data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})
print(f"DiD coefficient (daca_x_post): {model5.params['daca_x_post']:.4f}")
print(f"Standard error: {model5.bse['daca_x_post']:.4f}")
print(f"t-statistic: {model5.tvalues['daca_x_post']:.4f}")
print(f"p-value: {model5.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {model5.nobs:.0f}")

# ============================================================================
# STEP 9: ROBUSTNESS CHECKS
# ============================================================================
print("\n[9] Running robustness checks...")

# Robustness 1: Alternative age restrictions (18-55)
print("\n--- Robustness 1: Ages 18-55 ---")
df_robust1 = df[(df['AGE'] >= 18) & (df['AGE'] <= 55) & (df['YEAR'] != 2012)].copy()
df_robust1['daca_x_post'] = df_robust1['daca_eligible'] * df_robust1['post']
df_robust1 = df_robust1.dropna(subset=['fulltime', 'daca_eligible', 'YEAR', 'AGE', 'SEX', 'EDUC'])

robust1 = smf.ols('fulltime ~ daca_eligible + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_robust1).fit(cov_type='cluster', cov_kwds={'groups': df_robust1['state']})
print(f"DiD coefficient: {robust1.params['daca_x_post']:.4f} (SE: {robust1.bse['daca_x_post']:.4f})")

# Robustness 2: Include 2012
print("\n--- Robustness 2: Include 2012 ---")
df_robust2 = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()
df_robust2['daca_x_post'] = df_robust2['daca_eligible'] * df_robust2['post']
df_robust2 = df_robust2.dropna(subset=['fulltime', 'daca_eligible', 'YEAR', 'AGE', 'SEX', 'EDUC'])

robust2 = smf.ols('fulltime ~ daca_eligible + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_robust2).fit(cov_type='cluster', cov_kwds={'groups': df_robust2['state']})
print(f"DiD coefficient: {robust2.params['daca_x_post']:.4f} (SE: {robust2.bse['daca_x_post']:.4f})")

# Robustness 3: Males only
print("\n--- Robustness 3: Males only ---")
df_robust3 = df_analysis[df_analysis['female'] == 0].copy()
robust3 = smf.ols('fulltime ~ daca_eligible + daca_x_post + age + age_sq + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_robust3).fit(cov_type='cluster', cov_kwds={'groups': df_robust3['state']})
print(f"DiD coefficient: {robust3.params['daca_x_post']:.4f} (SE: {robust3.bse['daca_x_post']:.4f})")

# Robustness 4: Females only
print("\n--- Robustness 4: Females only ---")
df_robust4 = df_analysis[df_analysis['female'] == 1].copy()
robust4 = smf.ols('fulltime ~ daca_eligible + daca_x_post + age + age_sq + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_robust4).fit(cov_type='cluster', cov_kwds={'groups': df_robust4['state']})
print(f"DiD coefficient: {robust4.params['daca_x_post']:.4f} (SE: {robust4.bse['daca_x_post']:.4f})")

# Robustness 5: Broader age at immigration cutoff (<18)
print("\n--- Robustness 5: Age at immigration < 18 ---")
df_robust5 = df[(df['AGE'] >= 16) & (df['AGE'] <= 64) & (df['YEAR'] != 2012)].copy()
df_robust5['daca_eligible_broad'] = (
    (df_robust5['age_at_immig'] >= 0) &
    (df_robust5['age_at_immig'] < 18) &  # Broader cutoff
    (df_robust5['BIRTHYR'] >= 1982) &
    (df_robust5['YRIMMIG'] <= 2007) &
    (df_robust5['YRIMMIG'] > 0) &
    (df_robust5['CITIZEN'] == 3)
).astype(int)
df_robust5['daca_x_post_broad'] = df_robust5['daca_eligible_broad'] * df_robust5['post']
df_robust5 = df_robust5.dropna(subset=['fulltime', 'daca_eligible_broad', 'YEAR', 'AGE', 'SEX', 'EDUC'])

robust5 = smf.ols('fulltime ~ daca_eligible_broad + daca_x_post_broad + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_robust5).fit(cov_type='cluster', cov_kwds={'groups': df_robust5['state']})
print(f"DiD coefficient: {robust5.params['daca_x_post_broad']:.4f} (SE: {robust5.bse['daca_x_post_broad']:.4f})")

# ============================================================================
# STEP 10: EVENT STUDY
# ============================================================================
print("\n[10] Running event study analysis...")

# Create year-specific treatment effects
df_analysis['daca_x_2006'] = df_analysis['daca_eligible'] * df_analysis['year_2006']
df_analysis['daca_x_2007'] = df_analysis['daca_eligible'] * df_analysis['year_2007']
df_analysis['daca_x_2008'] = df_analysis['daca_eligible'] * df_analysis['year_2008']
df_analysis['daca_x_2009'] = df_analysis['daca_eligible'] * df_analysis['year_2009']
df_analysis['daca_x_2010'] = df_analysis['daca_eligible'] * df_analysis['year_2010']
# 2011 is reference year (omitted)
df_analysis['daca_x_2013'] = df_analysis['daca_eligible'] * df_analysis['year_2013']
df_analysis['daca_x_2014'] = df_analysis['daca_eligible'] * df_analysis['year_2014']
df_analysis['daca_x_2015'] = df_analysis['daca_eligible'] * df_analysis['year_2015']
df_analysis['daca_x_2016'] = df_analysis['daca_eligible'] * df_analysis['year_2016']

event_study = smf.ols('''fulltime ~ daca_eligible +
                         daca_x_2006 + daca_x_2007 + daca_x_2008 + daca_x_2009 + daca_x_2010 +
                         daca_x_2013 + daca_x_2014 + daca_x_2015 + daca_x_2016 +
                         age + age_sq + female + married + hs_grad + some_college + college_plus +
                         C(state) + C(YEAR)''',
                      data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['state']})

print("\nEvent Study Coefficients:")
event_vars = ['daca_x_2006', 'daca_x_2007', 'daca_x_2008', 'daca_x_2009', 'daca_x_2010',
              'daca_x_2013', 'daca_x_2014', 'daca_x_2015', 'daca_x_2016']
for var in event_vars:
    print(f"  {var}: {event_study.params[var]:.4f} (SE: {event_study.bse[var]:.4f}, p={event_study.pvalues[var]:.4f})")

# Save event study results
event_results = pd.DataFrame({
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coef': [event_study.params.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
            [event_study.params.get(f'daca_x_{y}', 0) for y in [2013, 2014, 2015, 2016]],
    'se': [event_study.bse.get(f'daca_x_{y}', 0) for y in [2006, 2007, 2008, 2009, 2010]] + [0] +
          [event_study.bse.get(f'daca_x_{y}', 0) for y in [2013, 2014, 2015, 2016]]
})
event_results.to_csv(os.path.join(output_dir, 'event_study_results.csv'), index=False)

# ============================================================================
# STEP 11: PLACEBO TEST
# ============================================================================
print("\n[11] Running placebo test (pre-trends)...")

# Use only pre-DACA period (2006-2011) with fake treatment in 2009
df_placebo = df[(df['AGE'] >= 16) & (df['AGE'] <= 64) & (df['YEAR'].isin([2006, 2007, 2008, 2009, 2010, 2011]))].copy()
df_placebo['fake_post'] = (df_placebo['YEAR'] >= 2009).astype(int)
df_placebo['daca_x_fake_post'] = df_placebo['daca_eligible'] * df_placebo['fake_post']
df_placebo = df_placebo.dropna(subset=['fulltime', 'daca_eligible', 'YEAR', 'AGE', 'SEX', 'EDUC'])

placebo = smf.ols('fulltime ~ daca_eligible + daca_x_fake_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                  data=df_placebo).fit(cov_type='cluster', cov_kwds={'groups': df_placebo['state']})
print(f"Placebo DiD coefficient: {placebo.params['daca_x_fake_post']:.4f} (SE: {placebo.bse['daca_x_fake_post']:.4f}, p={placebo.pvalues['daca_x_fake_post']:.4f})")

# ============================================================================
# STEP 12: TRIPLE DIFFERENCES
# ============================================================================
print("\n[12] Running triple-differences analysis...")

# Use citizens as an additional control group
# This helps control for Mexican-specific shocks
df_triple = df[(df['AGE'] >= 16) & (df['AGE'] <= 64) & (df['YEAR'] != 2012)].copy()

# DACA-eligible non-citizens vs all others (citizens + non-eligible non-citizens)
df_triple['non_citizen'] = (df_triple['CITIZEN'] == 3).astype(int)
df_triple['daca_elig_noncit'] = df_triple['daca_eligible']

# Triple interaction
df_triple['post'] = (df_triple['YEAR'] >= 2013).astype(int)
df_triple['daca_x_post'] = df_triple['daca_elig_noncit'] * df_triple['post']
df_triple = df_triple.dropna(subset=['fulltime', 'daca_elig_noncit', 'YEAR', 'AGE', 'SEX', 'EDUC'])

triple = smf.ols('fulltime ~ daca_elig_noncit + daca_x_post + age + age_sq + female + married + hs_grad + some_college + college_plus + C(state) + C(YEAR)',
                 data=df_triple).fit(cov_type='cluster', cov_kwds={'groups': df_triple['state']})
print(f"Triple-diff coefficient: {triple.params['daca_x_post']:.4f} (SE: {triple.bse['daca_x_post']:.4f})")

# ============================================================================
# STEP 13: SAVE RESULTS
# ============================================================================
print("\n[13] Saving results...")

# Create results summary
results_summary = {
    'Model': ['Simple DiD', 'DiD + Demographics', 'DiD + Demographics + Education',
              'DiD + State FE', 'DiD + State + Year FE (Preferred)'],
    'Coefficient': [model1.params['daca_x_post'], model2.params['daca_x_post'],
                    model3.params['daca_x_post'], model4.params['daca_x_post'],
                    model5.params['daca_x_post']],
    'SE': [model1.bse['daca_x_post'], model2.bse['daca_x_post'],
           model3.bse['daca_x_post'], model4.bse['daca_x_post'],
           model5.bse['daca_x_post']],
    'p-value': [model1.pvalues['daca_x_post'], model2.pvalues['daca_x_post'],
                model3.pvalues['daca_x_post'], model4.pvalues['daca_x_post'],
                model5.pvalues['daca_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs)],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared]
}

results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(output_dir, 'main_results.csv'), index=False)

# Robustness results
robustness_summary = {
    'Specification': ['Ages 18-55', 'Include 2012', 'Males only', 'Females only', 'Age at immig < 18'],
    'Coefficient': [robust1.params['daca_x_post'], robust2.params['daca_x_post'],
                    robust3.params['daca_x_post'], robust4.params['daca_x_post'],
                    robust5.params['daca_x_post_broad']],
    'SE': [robust1.bse['daca_x_post'], robust2.bse['daca_x_post'],
           robust3.bse['daca_x_post'], robust4.bse['daca_x_post'],
           robust5.bse['daca_x_post_broad']],
    'N': [int(robust1.nobs), int(robust2.nobs), int(robust3.nobs),
          int(robust4.nobs), int(robust5.nobs)]
}

robustness_df = pd.DataFrame(robustness_summary)
robustness_df.to_csv(os.path.join(output_dir, 'robustness_results.csv'), index=False)

# ============================================================================
# STEP 14: PRINT FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print(f"\nPreferred Estimate (Model 5: DiD with Year and State FE):")
print(f"  Effect of DACA eligibility on full-time employment: {model5.params['daca_x_post']:.4f}")
print(f"  Standard Error (clustered at state level): {model5.bse['daca_x_post']:.4f}")
print(f"  95% Confidence Interval: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['daca_x_post']:.4f}")
print(f"  Sample Size: {int(model5.nobs):,}")

print(f"\nInterpretation:")
coef = model5.params['daca_x_post']
if coef > 0:
    print(f"  DACA eligibility is associated with a {coef*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(coef)*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model5.pvalues['daca_x_post'] < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
elif model5.pvalues['daca_x_post'] < 0.10:
    print(f"  This effect is statistically significant at the 10% level.")
else:
    print(f"  This effect is not statistically significant at conventional levels.")

print("\n" + "=" * 70)
print("Analysis complete. Results saved to CSV files.")
print("=" * 70)

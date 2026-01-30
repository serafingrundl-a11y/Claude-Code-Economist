"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals
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
print("DACA REPLICATION STUDY - DATA EXPLORATION")
print("="*80)

# Load data - only necessary columns to manage memory
print("\nLoading data (this may take a moment)...")
cols_needed = ['YEAR', 'PERWT', 'AGE', 'BIRTHYR', 'BIRTHQTR', 'SEX',
               'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG',
               'EDUC', 'EDUCD', 'MARST', 'EMPSTAT', 'LABFORCE', 'UHRSWORK',
               'STATEFIP']

# Read data in chunks to manage memory
chunk_size = 1000000
chunks = []
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=chunk_size):
    # Initial filter: Hispanic-Mexican and born in Mexico
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"Loaded {len(df):,} Hispanic-Mexican, Mexican-born observations")

# Explore the data
print("\n" + "="*80)
print("DATA EXPLORATION")
print("="*80)

print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

print("\n--- Citizenship Status ---")
print("CITIZEN codes: 0=N/A, 1=Born abroad of American parents, 2=Naturalized, 3=Not a citizen")
print(df['CITIZEN'].value_counts().sort_index())

print("\n--- Age Distribution (summary) ---")
print(df['AGE'].describe())

print("\n--- Year of Immigration (non-zero, sample) ---")
yrimmig_dist = df[df['YRIMMIG'] > 0]['YRIMMIG'].value_counts().sort_index()
print(f"Range: {yrimmig_dist.index.min()} to {yrimmig_dist.index.max()}")
print(f"Non-zero observations: {len(df[df['YRIMMIG'] > 0]):,}")

print("\n--- Employment Variables ---")
print("EMPSTAT distribution:")
print(df['EMPSTAT'].value_counts().sort_index())
print("\nUHRSWORK range:", df['UHRSWORK'].min(), "-", df['UHRSWORK'].max())

print("\n" + "="*80)
print("SAMPLE CONSTRUCTION")
print("="*80)

# Start with all Hispanic-Mexican, Mexican-born
print(f"\n1. Starting sample (Hispanic-Mexican, Mexican-born): {len(df):,}")

# Keep non-citizens only (proxy for undocumented)
df_noncit = df[df['CITIZEN'] == 3].copy()
print(f"2. After keeping non-citizens only: {len(df_noncit):,}")

# Remove observations with missing immigration year
df_noncit = df_noncit[df_noncit['YRIMMIG'] > 0]
print(f"3. After removing missing YRIMMIG: {len(df_noncit):,}")

# Calculate age at immigration
df_noncit['age_at_immig'] = df_noncit['YRIMMIG'] - df_noncit['BIRTHYR']
print(f"\nAge at immigration range: {df_noncit['age_at_immig'].min()} to {df_noncit['age_at_immig'].max()}")

# Filter to working-age population (16-64)
df_working = df_noncit[(df_noncit['AGE'] >= 16) & (df_noncit['AGE'] <= 64)].copy()
print(f"4. After restricting to ages 16-64: {len(df_working):,}")

# Create DACA eligibility indicator
# Eligible if:
# 1. Arrived before age 16
# 2. Born in 1982 or later (under 31 as of June 15, 2012)
# 3. Immigrated by 2007 (5 years continuous presence by 2012)

df_working['arrived_before_16'] = df_working['age_at_immig'] < 16
df_working['under_31_in_2012'] = df_working['BIRTHYR'] >= 1982
df_working['present_by_2007'] = df_working['YRIMMIG'] <= 2007

# DACA eligible if all three conditions met
df_working['daca_eligible'] = (df_working['arrived_before_16'] &
                               df_working['under_31_in_2012'] &
                               df_working['present_by_2007']).astype(int)

print("\n--- DACA Eligibility Breakdown ---")
print(f"Arrived before age 16: {df_working['arrived_before_16'].sum():,}")
print(f"Under 31 in 2012: {df_working['under_31_in_2012'].sum():,}")
print(f"Present by 2007: {df_working['present_by_2007'].sum():,}")
print(f"All three (DACA eligible): {df_working['daca_eligible'].sum():,}")

# Create post-DACA indicator (2013-2016 as treatment period, excluding 2012)
df_working['post_daca'] = (df_working['YEAR'] >= 2013).astype(int)

# Exclude 2012 as it's a partial treatment year
df_analysis = df_working[df_working['YEAR'] != 2012].copy()
print(f"\n5. After excluding 2012: {len(df_analysis):,}")

# Create full-time employment outcome (35+ hours per week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Also create employed indicator
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# Create interaction term
df_analysis['eligible_x_post'] = df_analysis['daca_eligible'] * df_analysis['post_daca']

print("\n--- Final Analytic Sample Summary ---")
print(f"Total observations: {len(df_analysis):,}")
print(f"Pre-DACA (2006-2011): {len(df_analysis[df_analysis['post_daca']==0]):,}")
print(f"Post-DACA (2013-2016): {len(df_analysis[df_analysis['post_daca']==1]):,}")
print(f"DACA Eligible: {df_analysis['daca_eligible'].sum():,}")
print(f"DACA Ineligible: {len(df_analysis) - df_analysis['daca_eligible'].sum():,}")

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)

# Education categories
df_analysis['educ_lesshs'] = (df_analysis['EDUC'] < 6).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] == 6).astype(int)
df_analysis['educ_somecoll'] = ((df_analysis['EDUC'] > 6) & (df_analysis['EDUC'] < 10)).astype(int)
df_analysis['educ_college'] = (df_analysis['EDUC'] >= 10).astype(int)

# Age squared
df_analysis['age_sq'] = df_analysis['AGE'] ** 2

# Save analysis dataset
df_analysis.to_csv('analysis_data.csv', index=False)
print("\nAnalysis dataset saved to 'analysis_data.csv'")

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Summary statistics by eligibility and period
print("\n--- Mean Full-Time Employment by Group and Period ---")
summary = df_analysis.groupby(['daca_eligible', 'post_daca']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean'],
    'PERWT': ['sum']
}).round(4)
print(summary)

# Weighted means
print("\n--- Weighted Mean Full-Time Employment ---")
for eligible in [0, 1]:
    for post in [0, 1]:
        mask = (df_analysis['daca_eligible'] == eligible) & (df_analysis['post_daca'] == post)
        subset = df_analysis[mask]
        weighted_mean = np.average(subset['fulltime'], weights=subset['PERWT'])
        n = len(subset)
        period = "Post" if post else "Pre"
        group = "Eligible" if eligible else "Ineligible"
        print(f"{group}, {period}: {weighted_mean:.4f} (n={n:,})")

# Simple DiD calculation
print("\n--- Simple Difference-in-Differences ---")
means = df_analysis.groupby(['daca_eligible', 'post_daca'])['fulltime'].mean()
did = (means[1, 1] - means[1, 0]) - (means[0, 1] - means[0, 0])
print(f"Change in Eligible: {means[1, 1] - means[1, 0]:.4f}")
print(f"Change in Ineligible: {means[0, 1] - means[0, 0]:.4f}")
print(f"DiD Estimate: {did:.4f}")

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD
print("\n--- Model 1: Basic Difference-in-Differences ---")
model1 = smf.ols('fulltime ~ daca_eligible + post_daca + eligible_x_post',
                 data=df_analysis).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: With demographic controls
print("\n--- Model 2: With Demographic Controls ---")
model2 = smf.ols('fulltime ~ daca_eligible + post_daca + eligible_x_post + '
                 'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college',
                 data=df_analysis).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: With year fixed effects
print("\n--- Model 3: With Year Fixed Effects ---")
df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model3 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + '
                 'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                 'C(YEAR)',
                 data=df_analysis).fit(cov_type='HC1')
# Print only key coefficients
print("Key coefficients:")
for var in ['daca_eligible', 'eligible_x_post', 'AGE', 'female', 'married']:
    if var in model3.params.index:
        print(f"  {var}: {model3.params[var]:.4f} (se: {model3.bse[var]:.4f})")

# Model 4: With state fixed effects (preferred specification)
print("\n--- Model 4: With State and Year Fixed Effects (Preferred) ---")
model4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + '
                 'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                 'C(YEAR) + C(STATEFIP)',
                 data=df_analysis).fit(cov_type='HC1')
print("\nKey coefficients:")
for var in ['daca_eligible', 'eligible_x_post', 'AGE', 'female', 'married']:
    if var in model4.params.index:
        print(f"  {var}: {model4.params[var]:.4f} (se: {model4.bse[var]:.4f}, p: {model4.pvalues[var]:.4f})")

print(f"\nN = {int(model4.nobs):,}")
print(f"R-squared = {model4.rsquared:.4f}")

# Store preferred estimate
preferred_estimate = model4.params['eligible_x_post']
preferred_se = model4.bse['eligible_x_post']
preferred_ci_low = preferred_estimate - 1.96 * preferred_se
preferred_ci_high = preferred_estimate + 1.96 * preferred_se

print("\n" + "="*80)
print("PREFERRED ESTIMATE SUMMARY")
print("="*80)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  Point estimate: {preferred_estimate:.4f}")
print(f"  Standard error: {preferred_se:.4f}")
print(f"  95% CI: [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  t-statistic: {preferred_estimate/preferred_se:.3f}")
print(f"  p-value: {model4.pvalues['eligible_x_post']:.4f}")
print(f"  Sample size: {int(model4.nobs):,}")

# Save key results
results = {
    'model': ['Basic DiD', 'With Controls', 'Year FE', 'Year + State FE'],
    'coefficient': [model1.params['eligible_x_post'],
                   model2.params['eligible_x_post'],
                   model3.params['eligible_x_post'],
                   model4.params['eligible_x_post']],
    'std_error': [model1.bse['eligible_x_post'],
                 model2.bse['eligible_x_post'],
                 model3.bse['eligible_x_post'],
                 model4.bse['eligible_x_post']],
    'n_obs': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs)]
}
results_df = pd.DataFrame(results)
results_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to 'regression_results.csv'")

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Alternative age range (18-35)
print("\n--- Robustness 1: Ages 18-35 Only ---")
df_young = df_analysis[(df_analysis['AGE'] >= 18) & (df_analysis['AGE'] <= 35)]
model_r1 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + '
                   'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                   'C(YEAR) + C(STATEFIP)',
                   data=df_young).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r1.params['eligible_x_post']:.4f} (se: {model_r1.bse['eligible_x_post']:.4f})")
print(f"N = {int(model_r1.nobs):,}")

# Robustness 2: Employment (any) as outcome
print("\n--- Robustness 2: Any Employment as Outcome ---")
model_r2 = smf.ols('employed ~ daca_eligible + eligible_x_post + '
                   'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                   'C(YEAR) + C(STATEFIP)',
                   data=df_analysis).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r2.params['eligible_x_post']:.4f} (se: {model_r2.bse['eligible_x_post']:.4f})")
print(f"N = {int(model_r2.nobs):,}")

# Robustness 3: Male only sample
print("\n--- Robustness 3: Males Only ---")
df_male = df_analysis[df_analysis['female'] == 0]
model_r3 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + '
                   'AGE + age_sq + married + educ_hs + educ_somecoll + educ_college + '
                   'C(YEAR) + C(STATEFIP)',
                   data=df_male).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r3.params['eligible_x_post']:.4f} (se: {model_r3.bse['eligible_x_post']:.4f})")
print(f"N = {int(model_r3.nobs):,}")

# Robustness 4: Female only sample
print("\n--- Robustness 4: Females Only ---")
df_female = df_analysis[df_analysis['female'] == 1]
model_r4 = smf.ols('fulltime ~ daca_eligible + eligible_x_post + '
                   'AGE + age_sq + married + educ_hs + educ_somecoll + educ_college + '
                   'C(YEAR) + C(STATEFIP)',
                   data=df_female).fit(cov_type='HC1')
print(f"DiD coefficient: {model_r4.params['eligible_x_post']:.4f} (se: {model_r4.bse['eligible_x_post']:.4f})")
print(f"N = {int(model_r4.nobs):,}")

# Placebo test: Pre-trends
print("\n--- Placebo Test: Pre-Period Trends (2006-2011) ---")
df_pre = df_analysis[df_analysis['post_daca'] == 0].copy()
# Create fake "post" for years 2009-2011
df_pre['fake_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['fake_interaction'] = df_pre['daca_eligible'] * df_pre['fake_post']
model_placebo = smf.ols('fulltime ~ daca_eligible + fake_post + fake_interaction + '
                        'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                        'C(STATEFIP)',
                        data=df_pre).fit(cov_type='HC1')
print(f"Placebo DiD coefficient: {model_placebo.params['fake_interaction']:.4f} (se: {model_placebo.bse['fake_interaction']:.4f})")
print(f"(Should be close to zero if parallel trends hold)")

# Event study coefficients
print("\n--- Event Study: Year-by-Year Effects ---")
df_analysis['eligible_x_2006'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2006).astype(int)
df_analysis['eligible_x_2007'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2007).astype(int)
df_analysis['eligible_x_2008'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2008).astype(int)
df_analysis['eligible_x_2009'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2009).astype(int)
df_analysis['eligible_x_2010'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2010).astype(int)
# 2011 is reference year
df_analysis['eligible_x_2013'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2013).astype(int)
df_analysis['eligible_x_2014'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2014).astype(int)
df_analysis['eligible_x_2015'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2015).astype(int)
df_analysis['eligible_x_2016'] = df_analysis['daca_eligible'] * (df_analysis['YEAR'] == 2016).astype(int)

event_model = smf.ols('fulltime ~ daca_eligible + '
                      'eligible_x_2006 + eligible_x_2007 + eligible_x_2008 + '
                      'eligible_x_2009 + eligible_x_2010 + '
                      'eligible_x_2013 + eligible_x_2014 + eligible_x_2015 + eligible_x_2016 + '
                      'AGE + age_sq + female + married + educ_hs + educ_somecoll + educ_college + '
                      'C(YEAR) + C(STATEFIP)',
                      data=df_analysis).fit(cov_type='HC1')

print("Year-by-Year Interaction Coefficients (relative to 2011):")
event_vars = ['eligible_x_2006', 'eligible_x_2007', 'eligible_x_2008',
              'eligible_x_2009', 'eligible_x_2010',
              'eligible_x_2013', 'eligible_x_2014', 'eligible_x_2015', 'eligible_x_2016']
for var in event_vars:
    year = var.split('_')[-1]
    coef = event_model.params[var]
    se = event_model.bse[var]
    print(f"  {year}: {coef:.4f} (se: {se:.4f})")

# Save event study results
event_results = []
for var in event_vars:
    year = int(var.split('_')[-1])
    event_results.append({
        'year': year,
        'coefficient': event_model.params[var],
        'std_error': event_model.bse[var],
        'ci_low': event_model.params[var] - 1.96 * event_model.bse[var],
        'ci_high': event_model.params[var] + 1.96 * event_model.bse[var]
    })
# Add reference year
event_results.append({'year': 2011, 'coefficient': 0, 'std_error': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_results).sort_values('year')
event_df.to_csv('event_study_results.csv', index=False)
print("\nEvent study results saved to 'event_study_results.csv'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Final summary for report
print("\n### FINAL RESULTS FOR REPORT ###")
print(f"\nPreferred Specification: OLS with State and Year Fixed Effects")
print(f"Outcome: Full-time employment (35+ hours/week)")
print(f"Treatment: DACA eligibility × Post-2012")
print(f"\nMain Result:")
print(f"  β = {preferred_estimate:.4f}")
print(f"  SE = {preferred_se:.4f}")
print(f"  95% CI = [{preferred_ci_low:.4f}, {preferred_ci_high:.4f}]")
print(f"  N = {int(model4.nobs):,}")
print(f"\nInterpretation: DACA eligibility is associated with a {preferred_estimate*100:.2f} percentage point")
print(f"{'increase' if preferred_estimate > 0 else 'decrease'} in full-time employment.")

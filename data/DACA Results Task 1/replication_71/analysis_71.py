"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.

Author: Replication 71
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

# =============================================================================
# 1. LOAD DATA (Using chunked loading to manage memory)
# =============================================================================
print("\n1. Loading data (chunked approach for memory efficiency)...")

# Only load columns we need
cols_needed = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'MARST', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC',
               'EMPSTAT', 'UHRSWORK']

# Filter during loading: HISPAN == 1 (Mexican) and BPL == 200 (Mexico)
chunks = []
for chunk in pd.read_csv('data/data.csv', usecols=cols_needed, chunksize=1000000):
    # Apply initial filters to reduce memory
    chunk_filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)
print(f"   Loaded {len(df):,} observations (Hispanic-Mexican, Mexican-born)")

# =============================================================================
# 2. SAMPLE CONSTRUCTION
# =============================================================================
print("\n2. Constructing analytic sample...")

# Step 2a: Already restricted to Hispanic-Mexican and Mexican-born
df_sample = df.copy()
print(f"   Hispanic-Mexican, Mexican-born: {len(df_sample):,}")

# Step 2b: Restrict to non-citizens (CITIZEN == 3)
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()
print(f"   After non-citizen restriction: {len(df_sample):,}")

# Step 2c: Restrict to working-age population (18-55 for cleaner comparison)
df_sample = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 55)].copy()
print(f"   After age 18-55 restriction: {len(df_sample):,}")

# Step 2d: Exclude 2012 (transitional year)
df_sample = df_sample[df_sample['YEAR'] != 2012].copy()
print(f"   After excluding 2012: {len(df_sample):,}")

# Drop rows with missing YRIMMIG (needed for eligibility)
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()
print(f"   After dropping missing YRIMMIG: {len(df_sample):,}")

# =============================================================================
# 3. CREATE VARIABLES
# =============================================================================
print("\n3. Creating analysis variables...")

# Post-DACA indicator (2013-2016)
df_sample['post'] = (df_sample['YEAR'] >= 2013).astype(int)

# Calculate age at arrival in US
# YRIMMIG = year of immigration, BIRTHYR = birth year
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']

# Calculate if born after June 15, 1981 (under 31 on June 15, 2012)
# Being conservative: born in 1982 or later definitely qualifies
# Born in 1981: depends on birth quarter. Q3/Q4 (July-Dec) qualifies.
# BIRTHQTR: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec
df_sample['under_31_on_daca'] = (
    (df_sample['BIRTHYR'] >= 1982) |
    ((df_sample['BIRTHYR'] == 1981) & (df_sample['BIRTHQTR'] >= 3))
).astype(int)

# Arrived before 16th birthday
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16).astype(int)

# Present in US since June 15, 2007 (arrived in 2007 or earlier)
df_sample['present_since_2007'] = (df_sample['YRIMMIG'] <= 2007).astype(int)

# DACA eligible: meets all criteria
df_sample['daca_eligible'] = (
    (df_sample['arrived_before_16'] == 1) &
    (df_sample['under_31_on_daca'] == 1) &
    (df_sample['present_since_2007'] == 1)
).astype(int)

# Full-time employment (usually works 35+ hours per week)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Employed indicator
df_sample['employed'] = (df_sample['EMPSTAT'] == 1).astype(int)

# Create interaction term for DiD
df_sample['daca_x_post'] = df_sample['daca_eligible'] * df_sample['post']

# Demographic controls
df_sample['female'] = (df_sample['SEX'] == 2).astype(int)
df_sample['married'] = (df_sample['MARST'].isin([1, 2])).astype(int)

# Education categories
df_sample['less_than_hs'] = (df_sample['EDUC'] < 6).astype(int)
df_sample['hs_graduate'] = (df_sample['EDUC'] == 6).astype(int)
df_sample['some_college'] = (df_sample['EDUC'].isin([7, 8, 9])).astype(int)
df_sample['college_plus'] = (df_sample['EDUC'] >= 10).astype(int)

# Factor variables for fixed effects
df_sample['year_factor'] = df_sample['YEAR'].astype(str)
df_sample['state_factor'] = df_sample['STATEFIP'].astype(str)

print(f"   DACA-eligible observations: {df_sample['daca_eligible'].sum():,}")
print(f"   Non-eligible observations: {(1 - df_sample['daca_eligible']).sum():,}")

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n4. Descriptive Statistics")
print("-" * 60)

# Sample sizes by year and eligibility
print("\nSample sizes by year and DACA eligibility:")
sample_by_year = df_sample.groupby(['YEAR', 'daca_eligible']).size().unstack(fill_value=0)
sample_by_year.columns = ['Ineligible', 'Eligible']
print(sample_by_year)

# Pre/post periods
pre_df = df_sample[df_sample['post'] == 0]
post_df = df_sample[df_sample['post'] == 1]

print("\n\nMean full-time employment rates:")
print("-" * 40)

# DiD table
eligible_pre = pre_df[pre_df['daca_eligible'] == 1]['fulltime'].mean()
eligible_post = post_df[post_df['daca_eligible'] == 1]['fulltime'].mean()
ineligible_pre = pre_df[pre_df['daca_eligible'] == 0]['fulltime'].mean()
ineligible_post = post_df[post_df['daca_eligible'] == 0]['fulltime'].mean()

print(f"                    Pre-DACA    Post-DACA   Difference")
print(f"Eligible:           {eligible_pre:.4f}      {eligible_post:.4f}      {eligible_post - eligible_pre:+.4f}")
print(f"Ineligible:         {ineligible_pre:.4f}      {ineligible_post:.4f}      {ineligible_post - ineligible_pre:+.4f}")
print(f"\nDiD Estimate:       {(eligible_post - eligible_pre) - (ineligible_post - ineligible_pre):+.4f}")

# Summary statistics table
print("\n\nSummary Statistics by DACA Eligibility:")
print("-" * 60)
summary_vars = ['fulltime', 'employed', 'AGE', 'female', 'married',
                'less_than_hs', 'hs_graduate', 'some_college', 'college_plus',
                'UHRSWORK']

for var in summary_vars:
    elig_mean = df_sample[df_sample['daca_eligible'] == 1][var].mean()
    inelig_mean = df_sample[df_sample['daca_eligible'] == 0][var].mean()
    print(f"{var:20s}: Eligible = {elig_mean:.4f}, Ineligible = {inelig_mean:.4f}")

# =============================================================================
# 5. MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n\n5. Regression Analysis")
print("=" * 80)

# Model 1: Basic DiD
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post',
                  data=df_sample, weights=df_sample['PERWT'])
results1 = model1.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(results1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
model2 = smf.wls('fulltime ~ daca_eligible + post + daca_x_post + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus',
                  data=df_sample, weights=df_sample['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(results2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\nModel 3: DiD with year fixed effects")
model3 = smf.wls('fulltime ~ daca_eligible + daca_x_post + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor)',
                  data=df_sample, weights=df_sample['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"\nDiD coefficient (daca_x_post): {results3.params['daca_x_post']:.5f}")
print(f"Standard Error: {results3.bse['daca_x_post']:.5f}")
print(f"t-statistic: {results3.tvalues['daca_x_post']:.3f}")
print(f"p-value: {results3.pvalues['daca_x_post']:.4f}")

# Model 4: Full model with state and year fixed effects
print("\nModel 4: DiD with state and year fixed effects")
model4 = smf.wls('fulltime ~ daca_eligible + daca_x_post + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)',
                  data=df_sample, weights=df_sample['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"\nDiD coefficient (daca_x_post): {results4.params['daca_x_post']:.5f}")
print(f"Standard Error: {results4.bse['daca_x_post']:.5f}")
print(f"t-statistic: {results4.tvalues['daca_x_post']:.3f}")
print(f"p-value: {results4.pvalues['daca_x_post']:.4f}")
print(f"95% CI: [{results4.conf_int().loc['daca_x_post', 0]:.5f}, {results4.conf_int().loc['daca_x_post', 1]:.5f}]")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n\n6. Robustness Checks")
print("=" * 80)

# Robustness 1: Employment (extensive margin)
print("\nRobustness 1: Employment (Extensive Margin)")
model_emp = smf.wls('employed ~ daca_eligible + daca_x_post + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)',
                     data=df_sample, weights=df_sample['PERWT'])
results_emp = model_emp.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})
print(f"DiD coefficient (employment): {results_emp.params['daca_x_post']:.5f} (SE: {results_emp.bse['daca_x_post']:.5f})")

# Robustness 2: Restricted age range (20-45)
print("\nRobustness 2: Age 20-45 subsample")
df_age_restricted = df_sample[(df_sample['AGE'] >= 20) & (df_sample['AGE'] <= 45)]
model_age = smf.wls('fulltime ~ daca_eligible + daca_x_post + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)',
                     data=df_age_restricted, weights=df_age_restricted['PERWT'])
results_age = model_age.fit(cov_type='cluster', cov_kwds={'groups': df_age_restricted['STATEFIP']})
print(f"DiD coefficient (age 20-45): {results_age.params['daca_x_post']:.5f} (SE: {results_age.bse['daca_x_post']:.5f})")

# Robustness 3: Males only
print("\nRobustness 3: Males only")
df_male = df_sample[df_sample['female'] == 0]
model_male = smf.wls('fulltime ~ daca_eligible + daca_x_post + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)',
                      data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']})
print(f"DiD coefficient (males): {results_male.params['daca_x_post']:.5f} (SE: {results_male.bse['daca_x_post']:.5f})")

# Robustness 4: Females only
print("\nRobustness 4: Females only")
df_female = df_sample[df_sample['female'] == 1]
model_female = smf.wls('fulltime ~ daca_eligible + daca_x_post + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)',
                        data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']})
print(f"DiD coefficient (females): {results_female.params['daca_x_post']:.5f} (SE: {results_female.bse['daca_x_post']:.5f})")

# =============================================================================
# 7. EVENT STUDY
# =============================================================================
print("\n\n7. Event Study Analysis")
print("=" * 80)

# Create year dummies interacted with eligibility
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_sample[f'eligible_x_{year}'] = df_sample['daca_eligible'] * (df_sample['YEAR'] == year).astype(int)

# Event study regression (2011 is reference year)
event_formula = 'fulltime ~ daca_eligible + '
event_formula += ' + '.join([f'eligible_x_{year}' for year in years if year != 2011])
event_formula += ' + female + married + AGE + I(AGE**2) + less_than_hs + some_college + college_plus + C(year_factor) + C(state_factor)'

model_event = smf.wls(event_formula, data=df_sample, weights=df_sample['PERWT'])
results_event = model_event.fit(cov_type='cluster', cov_kwds={'groups': df_sample['STATEFIP']})

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
event_coefs = []
for year in years:
    if year != 2011:
        coef = results_event.params[f'eligible_x_{year}']
        se = results_event.bse[f'eligible_x_{year}']
        print(f"  Year {year}: {coef:+.5f} (SE: {se:.5f})")
        event_coefs.append({'year': year, 'coef': coef, 'se': se})
    else:
        print(f"  Year {year}: 0.00000 (reference)")
        event_coefs.append({'year': year, 'coef': 0.0, 'se': 0.0})

# Save event study results
event_df = pd.DataFrame(event_coefs)
event_df.to_csv('event_study_results.csv', index=False)

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n\n8. Saving Results")
print("=" * 80)

# Create results dictionary
results_dict = {
    'main_estimate': results4.params['daca_x_post'],
    'main_se': results4.bse['daca_x_post'],
    'main_ci_low': results4.conf_int().loc['daca_x_post', 0],
    'main_ci_high': results4.conf_int().loc['daca_x_post', 1],
    'main_pvalue': results4.pvalues['daca_x_post'],
    'n_total': len(df_sample),
    'n_eligible': int(df_sample['daca_eligible'].sum()),
    'n_ineligible': int((1 - df_sample['daca_eligible']).sum()),
    'eligible_pre': eligible_pre,
    'eligible_post': eligible_post,
    'ineligible_pre': ineligible_pre,
    'ineligible_post': ineligible_post,
    'raw_did': (eligible_post - eligible_pre) - (ineligible_post - ineligible_pre),
    # Model 1 results
    'model1_coef': results1.params['daca_x_post'],
    'model1_se': results1.bse['daca_x_post'],
    # Model 2 results
    'model2_coef': results2.params['daca_x_post'],
    'model2_se': results2.bse['daca_x_post'],
    # Model 3 results
    'model3_coef': results3.params['daca_x_post'],
    'model3_se': results3.bse['daca_x_post'],
    # Robustness
    'emp_coef': results_emp.params['daca_x_post'],
    'emp_se': results_emp.bse['daca_x_post'],
    'age_restricted_coef': results_age.params['daca_x_post'],
    'age_restricted_se': results_age.bse['daca_x_post'],
    'male_coef': results_male.params['daca_x_post'],
    'male_se': results_male.bse['daca_x_post'],
    'female_coef': results_female.params['daca_x_post'],
    'female_se': results_female.bse['daca_x_post'],
}

# Save to CSV
results_df = pd.DataFrame([results_dict])
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save summary statistics
summary_stats = df_sample.groupby('daca_eligible').agg({
    'fulltime': 'mean',
    'employed': 'mean',
    'AGE': 'mean',
    'female': 'mean',
    'married': 'mean',
    'less_than_hs': 'mean',
    'hs_graduate': 'mean',
    'some_college': 'mean',
    'college_plus': 'mean',
    'PERWT': 'sum'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to summary_statistics.csv")

# Print final summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(f"\nPreferred Estimate (Model 4 - State and Year FE):")
print(f"  DiD Coefficient: {results_dict['main_estimate']:.5f}")
print(f"  Standard Error:  {results_dict['main_se']:.5f}")
print(f"  95% CI:          [{results_dict['main_ci_low']:.5f}, {results_dict['main_ci_high']:.5f}]")
print(f"  p-value:         {results_dict['main_pvalue']:.4f}")
print(f"\nSample Size:       {results_dict['n_total']:,}")
print(f"  Eligible:        {results_dict['n_eligible']:,}")
print(f"  Ineligible:      {results_dict['n_ineligible']:,}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)

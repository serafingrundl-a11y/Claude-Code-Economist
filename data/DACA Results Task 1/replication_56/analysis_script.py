"""
DACA Replication Study - Analysis Script
Study 56: Impact of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born individuals in the United States

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in
the United States, what was the causal impact of eligibility for DACA on the
probability of full-time employment (35+ hours/week)?
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')
import os

# Set working directory
os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_56")

print("=" * 70)
print("DACA REPLICATION STUDY - ANALYSIS SCRIPT")
print("Study 56: Effect of DACA Eligibility on Full-Time Employment")
print("=" * 70)

# =============================================================================
# STEP 1: Load and Filter Data
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading and Filtering Data")
print("=" * 70)

# Define columns to read (to reduce memory usage)
cols_to_use = [
    'YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR',
    'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
    'EDUC', 'EDUCD', 'EMPSTAT', 'EMPSTATD', 'LABFORCE', 'UHRSWORK'
]

# Read data in chunks and filter
print("Reading data (this may take a few minutes due to file size)...")
chunk_size = 500000
filtered_chunks = []

# Filter criteria applied during reading:
# - Hispanic-Mexican (HISPAN == 1)
# - Born in Mexico (BPL == 200)
# - Working age (16-64)

for chunk in pd.read_csv('data/data.csv', usecols=cols_to_use, chunksize=chunk_size):
    # Filter to Hispanic-Mexican, Mexican-born, working age
    mask = (
        (chunk['HISPAN'] == 1) &  # Hispanic-Mexican
        (chunk['BPL'] == 200) &    # Born in Mexico
        (chunk['AGE'] >= 16) &     # Working age lower bound
        (chunk['AGE'] <= 64)       # Working age upper bound
    )
    filtered_chunks.append(chunk[mask])

df = pd.concat(filtered_chunks, ignore_index=True)
print(f"Total observations (Hispanic-Mexican, Mexico-born, age 16-64): {len(df):,}")

# =============================================================================
# STEP 2: Construct DACA Eligibility Indicators
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: Constructing DACA Eligibility Indicators")
print("=" * 70)

# DACA Eligibility Criteria:
# 1. Arrived before age 16
# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
# 3. Lived continuously in US since June 15, 2007 (arrived by 2007)
# 4. Not a citizen (CITIZEN == 3)

# Calculate age at arrival
# Note: YRIMMIG is year of immigration, BIRTHYR is birth year
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']

# Handle cases where YRIMMIG is 0 (N/A or missing)
df.loc[df['YRIMMIG'] == 0, 'age_at_arrival'] = np.nan

# Criterion 1: Arrived before age 16
df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['age_at_arrival'] >= 0)

# Criterion 2: Under 31 as of June 15, 2012 (born after 1981)
# Being conservative: born in 1982 or later definitely qualifies
# Born in 1981 might qualify depending on birth month
# Using birth quarter to refine: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
# June 15 is in Q2
df['born_after_cutoff'] = (
    (df['BIRTHYR'] > 1981) |
    ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))  # Born July 1981 or later
)

# Criterion 3: Arrived by 2007 (to satisfy continuous presence since June 2007)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)

# Criterion 4: Non-citizen (proxy for undocumented)
df['non_citizen'] = df['CITIZEN'] == 3

# Combined DACA eligibility indicator
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['born_after_cutoff'] &
    df['arrived_by_2007'] &
    df['non_citizen']
).astype(int)

print(f"\nDACA Eligibility Summary:")
print(f"  Arrived before age 16: {df['arrived_before_16'].sum():,}")
print(f"  Born after cutoff (under 31 in 2012): {df['born_after_cutoff'].sum():,}")
print(f"  Arrived by 2007: {df['arrived_by_2007'].sum():,}")
print(f"  Non-citizen: {df['non_citizen'].sum():,}")
print(f"  DACA Eligible (all criteria): {df['daca_eligible'].sum():,}")

# =============================================================================
# STEP 3: Create Analysis Variables
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: Creating Analysis Variables")
print("=" * 70)

# Outcome variable: Full-time employment (35+ hours/week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Post-DACA indicator (2013-2016)
# Excluding 2012 as transition year
df['post_daca'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term
df['eligible_x_post'] = df['daca_eligible'] * df['post_daca']

# Create control variables
# Sex (female indicator)
df['female'] = (df['SEX'] == 2).astype(int)

# Age squared
df['age_sq'] = df['AGE'] ** 2

# Marital status
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

# Education categories
df['educ_hs'] = (df['EDUCD'] >= 62).astype(int)  # High school or more
df['educ_college'] = (df['EDUCD'] >= 101).astype(int)  # College or more

# Years in USA (using YRSUSA1 if available)
df['years_in_usa'] = df['YRSUSA1']
df.loc[df['years_in_usa'] == 0, 'years_in_usa'] = np.nan

# In labor force indicator
df['in_labor_force'] = (df['LABFORCE'] == 2).astype(int)

# Employed indicator
df['employed'] = (df['EMPSTAT'] == 1).astype(int)

print(f"Outcome variable - Full-time employment rate: {df['fulltime'].mean():.3f}")
print(f"Post-DACA observations: {df['post_daca'].sum():,}")
print(f"Pre-DACA observations: {(df['post_daca'] == 0).sum():,}")

# =============================================================================
# STEP 4: Create Analysis Sample
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: Creating Analysis Sample")
print("=" * 70)

# Main analysis sample: non-citizens only (both eligible and non-eligible)
# This ensures we're comparing similar populations
df_analysis = df[df['non_citizen'] == True].copy()

# Exclude 2012 (transition year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012]

print(f"Analysis sample (non-citizens, excl. 2012): {len(df_analysis):,}")
print(f"  DACA Eligible: {df_analysis['daca_eligible'].sum():,}")
print(f"  Not DACA Eligible: {(df_analysis['daca_eligible'] == 0).sum():,}")

# Summary statistics by eligibility status
print("\n" + "-" * 50)
print("Summary by DACA Eligibility Status:")
print("-" * 50)
for eligible in [0, 1]:
    subset = df_analysis[df_analysis['daca_eligible'] == eligible]
    label = "DACA Eligible" if eligible == 1 else "Not DACA Eligible"
    print(f"\n{label}:")
    print(f"  N: {len(subset):,}")
    print(f"  Mean age: {subset['AGE'].mean():.1f}")
    print(f"  Female %: {subset['female'].mean()*100:.1f}%")
    print(f"  Married %: {subset['married'].mean()*100:.1f}%")
    print(f"  Full-time employment rate: {subset['fulltime'].mean()*100:.1f}%")
    print(f"  In labor force %: {subset['in_labor_force'].mean()*100:.1f}%")

# =============================================================================
# STEP 5: Summary Statistics Tables
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: Summary Statistics")
print("=" * 70)

# Table 1: Sample characteristics by eligibility and period
def weighted_mean(data, var, weight='PERWT'):
    """Calculate weighted mean."""
    return np.average(data[var], weights=data[weight])

def weighted_std(data, var, weight='PERWT'):
    """Calculate weighted standard deviation."""
    weights = data[weight]
    mean = weighted_mean(data, var, weight)
    variance = np.average((data[var] - mean)**2, weights=weights)
    return np.sqrt(variance)

summary_vars = ['AGE', 'female', 'married', 'educ_hs', 'fulltime', 'employed', 'in_labor_force', 'UHRSWORK']
summary_labels = ['Age', 'Female', 'Married', 'HS Education+', 'Full-Time Employed',
                  'Employed', 'In Labor Force', 'Usual Hours Worked']

summary_table = []
for period in ['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)']:
    for eligible in ['Eligible', 'Not Eligible']:
        if period.startswith('Pre'):
            period_mask = df_analysis['YEAR'] <= 2011
        else:
            period_mask = df_analysis['YEAR'] >= 2013

        if eligible == 'Eligible':
            elig_mask = df_analysis['daca_eligible'] == 1
        else:
            elig_mask = df_analysis['daca_eligible'] == 0

        subset = df_analysis[period_mask & elig_mask]

        row = {'Period': period, 'Group': eligible, 'N': len(subset)}
        for var, label in zip(summary_vars, summary_labels):
            row[label] = weighted_mean(subset, var) if len(subset) > 0 else np.nan
        summary_table.append(row)

summary_df = pd.DataFrame(summary_table)
print("\nTable 1: Summary Statistics by DACA Eligibility and Period")
print("-" * 80)
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv('summary_statistics.csv', index=False)
print("\nSaved: summary_statistics.csv")

# =============================================================================
# STEP 6: Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: Difference-in-Differences Analysis")
print("=" * 70)

# Ensure we have complete cases for regression
reg_vars = ['fulltime', 'daca_eligible', 'post_daca', 'eligible_x_post',
            'AGE', 'age_sq', 'female', 'married', 'educ_hs', 'PERWT', 'STATEFIP', 'YEAR']
df_reg = df_analysis.dropna(subset=reg_vars)
print(f"Regression sample (complete cases): {len(df_reg):,}")

# Model 1: Basic DiD (no controls)
print("\n" + "-" * 50)
print("Model 1: Basic DiD (no controls)")
print("-" * 50)

model1 = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post',
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n" + "-" * 50)
print("Model 2: DiD with demographic controls")
print("-" * 50)

model2 = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married + educ_hs',
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with state fixed effects
print("\n" + "-" * 50)
print("Model 3: DiD with state fixed effects")
print("-" * 50)

# Create state dummies
df_reg['state_fe'] = pd.Categorical(df_reg['STATEFIP'])

model3 = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married + educ_hs + C(STATEFIP)',
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')

# Print relevant coefficients only (not all state FE)
print("Key coefficients from Model 3:")
for var in ['Intercept', 'daca_eligible', 'post_daca', 'eligible_x_post',
            'AGE', 'age_sq', 'female', 'married', 'educ_hs']:
    if var in model3.params.index:
        coef = model3.params[var]
        se = model3.bse[var]
        pval = model3.pvalues[var]
        print(f"  {var:20s}: {coef:10.5f} (SE: {se:.5f}, p={pval:.4f})")

# Model 4: DiD with state and year fixed effects
print("\n" + "-" * 50)
print("Model 4: DiD with state and year fixed effects")
print("-" * 50)

model4 = smf.wls(
    'fulltime ~ daca_eligible + eligible_x_post + AGE + age_sq + female + married + educ_hs + C(STATEFIP) + C(YEAR)',
    data=df_reg,
    weights=df_reg['PERWT']
).fit(cov_type='HC1')

print("Key coefficients from Model 4:")
for var in ['Intercept', 'daca_eligible', 'eligible_x_post',
            'AGE', 'age_sq', 'female', 'married', 'educ_hs']:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        pval = model4.pvalues[var]
        print(f"  {var:20s}: {coef:10.5f} (SE: {se:.5f}, p={pval:.4f})")

# =============================================================================
# STEP 7: Robustness Checks
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: Robustness Checks")
print("=" * 70)

# Robustness 1: Alternative age range (18-35 for eligible group focus)
print("\n" + "-" * 50)
print("Robustness 1: Restricted age range (18-35)")
print("-" * 50)

df_rob1 = df_reg[(df_reg['AGE'] >= 18) & (df_reg['AGE'] <= 35)]
model_rob1 = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + female + married + educ_hs',
    data=df_rob1,
    weights=df_rob1['PERWT']
).fit(cov_type='HC1')
print(f"Sample size: {len(df_rob1):,}")
print(f"DiD coefficient (eligible_x_post): {model_rob1.params['eligible_x_post']:.5f}")
print(f"Standard error: {model_rob1.bse['eligible_x_post']:.5f}")
print(f"P-value: {model_rob1.pvalues['eligible_x_post']:.4f}")

# Robustness 2: Placebo test - use only pre-DACA period (2006-2011), fake treatment at 2009
print("\n" + "-" * 50)
print("Robustness 2: Placebo test (fake treatment at 2009)")
print("-" * 50)

df_pre = df_reg[df_reg['YEAR'] <= 2011].copy()
df_pre['placebo_post'] = (df_pre['YEAR'] >= 2009).astype(int)
df_pre['placebo_interaction'] = df_pre['daca_eligible'] * df_pre['placebo_post']

model_placebo = smf.wls(
    'fulltime ~ daca_eligible + placebo_post + placebo_interaction + AGE + age_sq + female + married + educ_hs',
    data=df_pre,
    weights=df_pre['PERWT']
).fit(cov_type='HC1')
print(f"Sample size: {len(df_pre):,}")
print(f"Placebo DiD coefficient: {model_placebo.params['placebo_interaction']:.5f}")
print(f"Standard error: {model_placebo.bse['placebo_interaction']:.5f}")
print(f"P-value: {model_placebo.pvalues['placebo_interaction']:.4f}")

# Robustness 3: Men only
print("\n" + "-" * 50)
print("Robustness 3: Men only")
print("-" * 50)

df_men = df_reg[df_reg['female'] == 0]
model_men = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + married + educ_hs',
    data=df_men,
    weights=df_men['PERWT']
).fit(cov_type='HC1')
print(f"Sample size: {len(df_men):,}")
print(f"DiD coefficient (eligible_x_post): {model_men.params['eligible_x_post']:.5f}")
print(f"Standard error: {model_men.bse['eligible_x_post']:.5f}")
print(f"P-value: {model_men.pvalues['eligible_x_post']:.4f}")

# Robustness 4: Women only
print("\n" + "-" * 50)
print("Robustness 4: Women only")
print("-" * 50)

df_women = df_reg[df_reg['female'] == 1]
model_women = smf.wls(
    'fulltime ~ daca_eligible + post_daca + eligible_x_post + AGE + age_sq + married + educ_hs',
    data=df_women,
    weights=df_women['PERWT']
).fit(cov_type='HC1')
print(f"Sample size: {len(df_women):,}")
print(f"DiD coefficient (eligible_x_post): {model_women.params['eligible_x_post']:.5f}")
print(f"Standard error: {model_women.bse['eligible_x_post']:.5f}")
print(f"P-value: {model_women.pvalues['eligible_x_post']:.4f}")

# =============================================================================
# STEP 8: Event Study Analysis
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: Event Study Analysis")
print("=" * 70)

# Create year dummies interacted with eligibility
# Reference year: 2011 (last pre-treatment year)
years = sorted(df_reg['YEAR'].unique())
ref_year = 2011

event_coeffs = {}
for year in years:
    if year != ref_year:
        df_reg[f'year_{year}'] = (df_reg['YEAR'] == year).astype(int)
        df_reg[f'eligible_x_{year}'] = df_reg['daca_eligible'] * df_reg[f'year_{year}']

# Build formula for event study
year_vars = [f'year_{y}' for y in years if y != ref_year]
interaction_vars = [f'eligible_x_{y}' for y in years if y != ref_year]
formula = 'fulltime ~ daca_eligible + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(interaction_vars) + ' + AGE + age_sq + female + married + educ_hs'

model_event = smf.wls(formula, data=df_reg, weights=df_reg['PERWT']).fit(cov_type='HC1')

# Extract event study coefficients
event_study_results = []
for year in years:
    if year == ref_year:
        event_study_results.append({
            'Year': year,
            'Coefficient': 0,
            'SE': 0,
            'CI_lower': 0,
            'CI_upper': 0
        })
    else:
        var = f'eligible_x_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        event_study_results.append({
            'Year': year,
            'Coefficient': coef,
            'SE': se,
            'CI_lower': coef - 1.96*se,
            'CI_upper': coef + 1.96*se
        })

event_df = pd.DataFrame(event_study_results)
print("\nEvent Study Coefficients (relative to 2011):")
print(event_df.to_string(index=False))

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("\nSaved: event_study_results.csv")

# Plot event study
plt.figure(figsize=(10, 6))
plt.errorbar(event_df['Year'], event_df['Coefficient'],
             yerr=1.96*event_df['SE'], fmt='o-', capsize=5, color='blue')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=2012.5, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Coefficient (relative to 2011)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('event_study_plot.png', dpi=300)
plt.close()
print("Saved: event_study_plot.png")

# =============================================================================
# STEP 9: Trends in Full-Time Employment
# =============================================================================
print("\n" + "=" * 70)
print("STEP 9: Trends Analysis")
print("=" * 70)

# Calculate weighted means by year and eligibility
trends = df_reg.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: pd.Series({
        'fulltime_rate': np.average(x['fulltime'], weights=x['PERWT']),
        'n_obs': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nFull-Time Employment Rates by Year and DACA Eligibility:")
print(trends.pivot(index='YEAR', columns='daca_eligible', values='fulltime_rate'))

# Save trends
trends.to_csv('employment_trends.csv', index=False)
print("\nSaved: employment_trends.csv")

# Plot trends
plt.figure(figsize=(10, 6))
for eligible in [0, 1]:
    subset = trends[trends['daca_eligible'] == eligible]
    label = 'DACA Eligible' if eligible == 1 else 'Not DACA Eligible'
    plt.plot(subset['YEAR'], subset['fulltime_rate'], 'o-', label=label, linewidth=2, markersize=8)

plt.axvline(x=2012.5, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Trends by DACA Eligibility', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('employment_trends_plot.png', dpi=300)
plt.close()
print("Saved: employment_trends_plot.png")

# =============================================================================
# STEP 10: Final Results Summary
# =============================================================================
print("\n" + "=" * 70)
print("STEP 10: Final Results Summary")
print("=" * 70)

# Compile results table
results_summary = {
    'Model': ['Basic DiD', 'With Controls', 'State FE', 'State + Year FE'],
    'DiD_Coefficient': [
        model1.params['eligible_x_post'],
        model2.params['eligible_x_post'],
        model3.params['eligible_x_post'],
        model4.params['eligible_x_post']
    ],
    'Std_Error': [
        model1.bse['eligible_x_post'],
        model2.bse['eligible_x_post'],
        model3.bse['eligible_x_post'],
        model4.bse['eligible_x_post']
    ],
    'P_Value': [
        model1.pvalues['eligible_x_post'],
        model2.pvalues['eligible_x_post'],
        model3.pvalues['eligible_x_post'],
        model4.pvalues['eligible_x_post']
    ],
    'N': [
        int(model1.nobs),
        int(model2.nobs),
        int(model3.nobs),
        int(model4.nobs)
    ],
    'R_Squared': [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared
    ]
}

results_df = pd.DataFrame(results_summary)
results_df['CI_lower'] = results_df['DiD_Coefficient'] - 1.96 * results_df['Std_Error']
results_df['CI_upper'] = results_df['DiD_Coefficient'] + 1.96 * results_df['Std_Error']

print("\nMain DiD Results:")
print(results_df.to_string(index=False))

# Save results
results_df.to_csv('main_results.csv', index=False)
print("\nSaved: main_results.csv")

# Robustness results
robustness_summary = {
    'Model': ['Age 18-35', 'Placebo (2009)', 'Men Only', 'Women Only'],
    'DiD_Coefficient': [
        model_rob1.params['eligible_x_post'],
        model_placebo.params['placebo_interaction'],
        model_men.params['eligible_x_post'],
        model_women.params['eligible_x_post']
    ],
    'Std_Error': [
        model_rob1.bse['eligible_x_post'],
        model_placebo.bse['placebo_interaction'],
        model_men.bse['eligible_x_post'],
        model_women.bse['eligible_x_post']
    ],
    'P_Value': [
        model_rob1.pvalues['eligible_x_post'],
        model_placebo.pvalues['placebo_interaction'],
        model_men.pvalues['eligible_x_post'],
        model_women.pvalues['eligible_x_post']
    ],
    'N': [
        int(model_rob1.nobs),
        int(model_placebo.nobs),
        int(model_men.nobs),
        int(model_women.nobs)
    ]
}

robustness_df = pd.DataFrame(robustness_summary)
print("\nRobustness Check Results:")
print(robustness_df.to_string(index=False))

robustness_df.to_csv('robustness_results.csv', index=False)
print("\nSaved: robustness_results.csv")

# =============================================================================
# STEP 11: Export Full Regression Tables for Report
# =============================================================================
print("\n" + "=" * 70)
print("STEP 11: Exporting Regression Tables")
print("=" * 70)

# Create detailed regression output
with open('regression_output.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA REPLICATION STUDY - FULL REGRESSION OUTPUT\n")
    f.write("=" * 80 + "\n\n")

    f.write("MODEL 1: Basic DiD (No Controls)\n")
    f.write("-" * 80 + "\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("MODEL 2: DiD with Demographic Controls\n")
    f.write("-" * 80 + "\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("MODEL 3: DiD with State Fixed Effects\n")
    f.write("-" * 80 + "\n")
    f.write("Note: State fixed effects coefficients omitted for brevity\n")
    f.write(f"DiD Coefficient (eligible_x_post): {model3.params['eligible_x_post']:.5f}\n")
    f.write(f"Standard Error: {model3.bse['eligible_x_post']:.5f}\n")
    f.write(f"P-value: {model3.pvalues['eligible_x_post']:.5f}\n")
    f.write(f"R-squared: {model3.rsquared:.5f}\n")
    f.write(f"N: {int(model3.nobs):,}\n\n")

    f.write("MODEL 4: DiD with State and Year Fixed Effects\n")
    f.write("-" * 80 + "\n")
    f.write("Note: Fixed effects coefficients omitted for brevity\n")
    f.write(f"DiD Coefficient (eligible_x_post): {model4.params['eligible_x_post']:.5f}\n")
    f.write(f"Standard Error: {model4.bse['eligible_x_post']:.5f}\n")
    f.write(f"P-value: {model4.pvalues['eligible_x_post']:.5f}\n")
    f.write(f"R-squared: {model4.rsquared:.5f}\n")
    f.write(f"N: {int(model4.nobs):,}\n\n")

    f.write("EVENT STUDY RESULTS\n")
    f.write("-" * 80 + "\n")
    f.write(event_df.to_string(index=False) + "\n\n")

    f.write("ROBUSTNESS CHECKS\n")
    f.write("-" * 80 + "\n")
    f.write(robustness_df.to_string(index=False) + "\n")

print("Saved: regression_output.txt")

# =============================================================================
# PREFERRED ESTIMATE
# =============================================================================
print("\n" + "=" * 70)
print("PREFERRED ESTIMATE (Model 2: DiD with Controls)")
print("=" * 70)

preferred_coef = model2.params['eligible_x_post']
preferred_se = model2.bse['eligible_x_post']
preferred_ci_lower = preferred_coef - 1.96 * preferred_se
preferred_ci_upper = preferred_coef + 1.96 * preferred_se
preferred_n = int(model2.nobs)

print(f"\nEffect Size: {preferred_coef:.5f}")
print(f"Standard Error: {preferred_se:.5f}")
print(f"95% CI: [{preferred_ci_lower:.5f}, {preferred_ci_upper:.5f}]")
print(f"Sample Size: {preferred_n:,}")
print(f"P-value: {model2.pvalues['eligible_x_post']:.5f}")

interpretation = f"""
Interpretation:
The difference-in-differences estimate suggests that DACA eligibility is associated
with a {abs(preferred_coef)*100:.2f} percentage point {'increase' if preferred_coef > 0 else 'decrease'}
in the probability of full-time employment for eligible Mexican-born non-citizen individuals
compared to non-eligible individuals. This effect is {'statistically significant' if model2.pvalues['eligible_x_post'] < 0.05 else 'not statistically significant'}
at the 5% level (p = {model2.pvalues['eligible_x_post']:.4f}).
"""
print(interpretation)

# Save preferred estimate
with open('preferred_estimate.txt', 'w') as f:
    f.write("PREFERRED ESTIMATE\n")
    f.write("=" * 50 + "\n")
    f.write(f"Model: DiD with Demographic Controls\n")
    f.write(f"Effect Size: {preferred_coef:.5f}\n")
    f.write(f"Standard Error: {preferred_se:.5f}\n")
    f.write(f"95% CI: [{preferred_ci_lower:.5f}, {preferred_ci_upper:.5f}]\n")
    f.write(f"Sample Size: {preferred_n:,}\n")
    f.write(f"P-value: {model2.pvalues['eligible_x_post']:.5f}\n")
    f.write(interpretation)

print("\nSaved: preferred_estimate.txt")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files generated:")
print("  - summary_statistics.csv")
print("  - main_results.csv")
print("  - robustness_results.csv")
print("  - event_study_results.csv")
print("  - employment_trends.csv")
print("  - event_study_plot.png")
print("  - employment_trends_plot.png")
print("  - regression_output.txt")
print("  - preferred_estimate.txt")

"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican individuals born in Mexico

Author: Independent Replication
Date: 2025
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

print("=" * 80)
print("DACA Replication Study - Full Analysis")
print("=" * 80)

# Define file paths
DATA_PATH = "data/data.csv"
OUTPUT_PATH = "results/"

# Create results directory if it doesn't exist
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================================
# STEP 1: Load and filter data in chunks
# ============================================================================
print("\n[STEP 1] Loading and filtering data...")

# Define columns to use
usecols = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR', 'MARST',
           'HISPAN', 'HISPAND', 'BPL', 'BPLD', 'CITIZEN', 'YRIMMIG', 'YRSUSA1',
           'EDUC', 'EDUCD', 'EMPSTAT', 'UHRSWORK', 'STATEFIP']

# Read data in chunks
chunk_size = 1000000
chunks = []

print(f"Reading data in chunks of {chunk_size:,} rows...")

for i, chunk in enumerate(pd.read_csv(DATA_PATH, usecols=usecols, chunksize=chunk_size)):
    # Filter for Hispanic-Mexican ethnicity (HISPAN == 1)
    # and born in Mexico (BPL == 200)
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)
    print(f"  Processed chunk {i+1}: {len(filtered):,} Hispanic-Mexican born in Mexico")

df = pd.concat(chunks, ignore_index=True)
print(f"\nTotal Hispanic-Mexican individuals born in Mexico: {len(df):,}")

# ============================================================================
# STEP 2: Define DACA eligibility criteria
# ============================================================================
print("\n[STEP 2] Defining DACA eligibility criteria...")

"""
DACA Eligibility Requirements (announced June 15, 2012):
1. Arrived in US before 16th birthday
2. Under 31 years old as of June 15, 2012 (born after June 15, 1981)
3. Lived continuously in US since June 15, 2007 (at least 5 years by 2012)
4. Present in US on June 15, 2012
5. Not a citizen and not a lawful permanent resident

We cannot distinguish documented vs undocumented non-citizens in ACS.
Following instructions: assume non-citizens without papers are undocumented.
"""

# Create a copy for analysis
df_analysis = df.copy()

# Filter to non-citizens (CITIZEN == 3 means "Not a citizen")
# Also keep CITIZEN == 4, 5 (first papers/status unknown - likely non-naturalized)
df_analysis = df_analysis[df_analysis['CITIZEN'].isin([3, 4, 5])]
print(f"Non-citizen Hispanic-Mexicans born in Mexico: {len(df_analysis):,}")

# Calculate age at arrival
# YRSUSA1 gives years in the US
# Age at arrival = current age - years in US
df_analysis['age_at_arrival'] = df_analysis['AGE'] - df_analysis['YRSUSA1']

# For cases where YRSUSA1 is 0 (less than 1 year or N/A), use YRIMMIG if available
# YRIMMIG gives the year of immigration
df_analysis.loc[df_analysis['YRSUSA1'] == 0, 'age_at_arrival'] = np.nan
df_analysis.loc[df_analysis['YRIMMIG'] > 0, 'year_arrived'] = df_analysis['YRIMMIG']
df_analysis.loc[df_analysis['YRIMMIG'] > 0, 'age_at_arrival'] = \
    df_analysis['BIRTHYR'] - df_analysis['YRIMMIG']
# Recalculate properly: arrival age = year arrived - birth year
df_analysis.loc[df_analysis['YRIMMIG'] > 0, 'age_at_arrival'] = \
    df_analysis.loc[df_analysis['YRIMMIG'] > 0, 'YRIMMIG'] - \
    df_analysis.loc[df_analysis['YRIMMIG'] > 0, 'BIRTHYR']

# Define DACA-eligible group
# Criterion 1: Arrived before 16th birthday
arrived_young = df_analysis['age_at_arrival'] < 16

# Criterion 2: Born after June 15, 1981 (under 31 as of June 15, 2012)
# To be conservative, use born in 1982 or later (definitely under 31)
# Those born in 1981 might be eligible depending on birth month
# Use BIRTHQTR to refine: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
# June 15 is in Q2, so those born Q3-Q4 of 1981 would be under 31
born_after_cutoff = (df_analysis['BIRTHYR'] >= 1982) | \
                   ((df_analysis['BIRTHYR'] == 1981) & (df_analysis['BIRTHQTR'].isin([3, 4])))

# Criterion 3: In US since at least June 15, 2007 (5+ years by 2012)
# Use YRIMMIG: must have arrived by 2007
# For years after 2012, we still use 2007 as the cutoff
arrived_by_2007 = df_analysis['YRIMMIG'] <= 2007

# Criterion 4: Present in US (we assume everyone in ACS was present)

# Combined DACA eligibility
df_analysis['daca_eligible'] = arrived_young & born_after_cutoff & arrived_by_2007
df_analysis['daca_eligible'] = df_analysis['daca_eligible'].fillna(False).astype(int)

print(f"DACA-eligible individuals: {df_analysis['daca_eligible'].sum():,}")
print(f"Non-DACA-eligible individuals: {(~df_analysis['daca_eligible'].astype(bool)).sum():,}")

# ============================================================================
# STEP 3: Define outcome variable and time periods
# ============================================================================
print("\n[STEP 3] Defining outcome variable and time periods...")

# Outcome: Full-time employment (usually works 35+ hours per week)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Also create employed variable for robustness
df_analysis['employed'] = (df_analysis['EMPSTAT'] == 1).astype(int)

# Define treatment period
# DACA announced June 15, 2012; applications started August 15, 2012
# Pre-period: 2006-2011 (before DACA)
# Post-period: 2013-2016 (after DACA implementation)
# 2012 is ambiguous (some before, some after), so exclude for clean comparison

df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['pre'] = (df_analysis['YEAR'] <= 2011).astype(int)

# Create interaction term for DiD
df_analysis['daca_x_post'] = df_analysis['daca_eligible'] * df_analysis['post']

# Restrict to working-age population (16-64) for employment analysis
df_analysis = df_analysis[(df_analysis['AGE'] >= 16) & (df_analysis['AGE'] <= 64)]
print(f"Working-age (16-64) sample size: {len(df_analysis):,}")

# Exclude 2012 for clean pre/post comparison
df_clean = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"Sample excluding 2012: {len(df_clean):,}")

# ============================================================================
# STEP 4: Summary statistics
# ============================================================================
print("\n[STEP 4] Computing summary statistics...")

# Summary by DACA eligibility and time period
summary_stats = df_clean.groupby(['daca_eligible', 'post']).agg({
    'fulltime': ['mean', 'std', 'count'],
    'employed': ['mean', 'std'],
    'AGE': 'mean',
    'PERWT': 'sum'
}).round(4)

print("\nSummary Statistics by DACA Eligibility and Period:")
print(summary_stats)

# Detailed summary
print("\n" + "-" * 60)
print("Detailed Summary Statistics")
print("-" * 60)

for eligible in [0, 1]:
    for period in [0, 1]:
        subset = df_clean[(df_clean['daca_eligible'] == eligible) & (df_clean['post'] == period)]
        period_label = "Post (2013-2016)" if period == 1 else "Pre (2006-2011)"
        eligible_label = "DACA-Eligible" if eligible == 1 else "Non-Eligible"

        print(f"\n{eligible_label}, {period_label}:")
        print(f"  N (unweighted): {len(subset):,}")
        print(f"  N (weighted): {subset['PERWT'].sum():,.0f}")
        print(f"  Full-time rate: {subset['fulltime'].mean():.4f}")
        print(f"  Employment rate: {subset['employed'].mean():.4f}")
        print(f"  Mean age: {subset['AGE'].mean():.2f}")

# Save summary statistics
summary_stats.to_csv(OUTPUT_PATH + "summary_statistics.csv")

# ============================================================================
# STEP 5: Difference-in-Differences Analysis
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 5] Difference-in-Differences Analysis")
print("=" * 80)

# Simple 2x2 DiD calculation
print("\n--- Simple 2x2 DiD ---")

# Calculate means for each cell
mean_elig_pre = df_clean[(df_clean['daca_eligible']==1) & (df_clean['post']==0)]['fulltime'].mean()
mean_elig_post = df_clean[(df_clean['daca_eligible']==1) & (df_clean['post']==1)]['fulltime'].mean()
mean_nonelig_pre = df_clean[(df_clean['daca_eligible']==0) & (df_clean['post']==0)]['fulltime'].mean()
mean_nonelig_post = df_clean[(df_clean['daca_eligible']==0) & (df_clean['post']==1)]['fulltime'].mean()

did_simple = (mean_elig_post - mean_elig_pre) - (mean_nonelig_post - mean_nonelig_pre)

print(f"\nFull-time Employment Rates:")
print(f"{'':20} {'Pre (2006-2011)':>15} {'Post (2013-2016)':>15} {'Diff':>10}")
print(f"{'DACA-Eligible':20} {mean_elig_pre:>15.4f} {mean_elig_post:>15.4f} {mean_elig_post-mean_elig_pre:>10.4f}")
print(f"{'Non-Eligible':20} {mean_nonelig_pre:>15.4f} {mean_nonelig_post:>15.4f} {mean_nonelig_post-mean_nonelig_pre:>10.4f}")
print(f"\nDifference-in-Differences Estimate: {did_simple:.4f}")

# ============================================================================
# STEP 6: Regression Analysis
# ============================================================================
print("\n--- Regression DiD Analysis ---")

# Model 1: Basic DiD without controls
print("\nModel 1: Basic DiD (no controls)")
model1 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post', data=df_clean).fit()
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\nModel 2: DiD with demographic controls")
# Create control variables
df_clean['female'] = (df_clean['SEX'] == 2).astype(int)
df_clean['married'] = (df_clean['MARST'] == 1).astype(int)
df_clean['age_sq'] = df_clean['AGE'] ** 2

# Education categories
df_clean['educ_hs'] = (df_clean['EDUC'] >= 6).astype(int)  # HS or more
df_clean['educ_college'] = (df_clean['EDUC'] >= 10).astype(int)  # College or more

model2 = smf.ols('fulltime ~ daca_eligible + post + daca_x_post + AGE + age_sq + female + married + educ_hs',
                 data=df_clean).fit()
print(model2.summary().tables[1])

# Model 3: DiD with year fixed effects
print("\nModel 3: DiD with year fixed effects")
df_clean['year_factor'] = pd.Categorical(df_clean['YEAR'])
model3 = smf.ols('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + C(YEAR)',
                 data=df_clean).fit()
print("\nKey coefficients:")
print(f"  DACA eligible:     {model3.params['daca_eligible']:.4f} (SE: {model3.bse['daca_eligible']:.4f})")
print(f"  DACA x Post (DiD): {model3.params['daca_x_post']:.4f} (SE: {model3.bse['daca_x_post']:.4f})")

# Model 4: Full model with state fixed effects
print("\nModel 4: Full model with state and year fixed effects")
model4 = smf.ols('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df_clean).fit()
print("\nKey coefficients:")
print(f"  DACA eligible:     {model4.params['daca_eligible']:.4f} (SE: {model4.bse['daca_eligible']:.4f})")
print(f"  DACA x Post (DiD): {model4.params['daca_x_post']:.4f} (SE: {model4.bse['daca_x_post']:.4f})")

# ============================================================================
# STEP 7: Weighted regression (using PERWT)
# ============================================================================
print("\n--- Weighted Regression Analysis ---")

# Model 5: Weighted DiD with full controls
print("\nModel 5: Weighted DiD with full controls")
model5 = smf.wls('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                 data=df_clean, weights=df_clean['PERWT']).fit()
print("\nKey coefficients (weighted):")
print(f"  DACA eligible:     {model5.params['daca_eligible']:.4f} (SE: {model5.bse['daca_eligible']:.4f})")
print(f"  DACA x Post (DiD): {model5.params['daca_x_post']:.4f} (SE: {model5.bse['daca_x_post']:.4f})")

# ============================================================================
# STEP 8: Robustness checks
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 8] Robustness Checks")
print("=" * 80)

# Alternative outcome: Employment (any employment, not just full-time)
print("\n--- Alternative Outcome: Employment (any) ---")
model_emp = smf.ols('employed ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + C(YEAR) + C(STATEFIP)',
                    data=df_clean).fit()
print(f"  DACA x Post (DiD): {model_emp.params['daca_x_post']:.4f} (SE: {model_emp.bse['daca_x_post']:.4f})")

# Restrict to ages 18-30 (more likely DACA eligible)
print("\n--- Restricted Sample: Ages 18-30 ---")
df_young = df_clean[(df_clean['AGE'] >= 18) & (df_clean['AGE'] <= 30)]
model_young = smf.ols('fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + C(YEAR)',
                      data=df_young).fit()
print(f"  Sample size: {len(df_young):,}")
print(f"  DACA x Post (DiD): {model_young.params['daca_x_post']:.4f} (SE: {model_young.bse['daca_x_post']:.4f})")

# Event study: year-by-year effects
print("\n--- Event Study: Year-by-Year Effects ---")
df_clean['daca_2006'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2006)
df_clean['daca_2007'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2007)
df_clean['daca_2008'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2008)
df_clean['daca_2009'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2009)
df_clean['daca_2010'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2010)
# 2011 is reference year
df_clean['daca_2013'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2013)
df_clean['daca_2014'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2014)
df_clean['daca_2015'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2015)
df_clean['daca_2016'] = (df_clean['daca_eligible'] == 1) & (df_clean['YEAR'] == 2016)

# Convert to int
for col in ['daca_2006', 'daca_2007', 'daca_2008', 'daca_2009', 'daca_2010',
            'daca_2013', 'daca_2014', 'daca_2015', 'daca_2016']:
    df_clean[col] = df_clean[col].astype(int)

model_event = smf.ols('fulltime ~ daca_eligible + daca_2006 + daca_2007 + daca_2008 + daca_2009 + daca_2010 + '
                      'daca_2013 + daca_2014 + daca_2015 + daca_2016 + '
                      'AGE + age_sq + female + married + educ_hs + C(YEAR)',
                      data=df_clean).fit()

print("\nEvent Study Coefficients (relative to 2011):")
event_years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
for year in event_years:
    coef_name = f'daca_{year}'
    print(f"  {year}: {model_event.params[coef_name]:>8.4f} (SE: {model_event.bse[coef_name]:.4f})")

# ============================================================================
# STEP 9: Save results
# ============================================================================
print("\n" + "=" * 80)
print("[STEP 9] Saving Results")
print("=" * 80)

# Create results dictionary
results = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'State+Year FE', 'Weighted'],
    'DiD Coefficient': [model1.params['daca_x_post'], model2.params['daca_x_post'],
                        model3.params['daca_x_post'], model4.params['daca_x_post'],
                        model5.params['daca_x_post']],
    'Std Error': [model1.bse['daca_x_post'], model2.bse['daca_x_post'],
                  model3.bse['daca_x_post'], model4.bse['daca_x_post'],
                  model5.bse['daca_x_post']],
    't-statistic': [model1.tvalues['daca_x_post'], model2.tvalues['daca_x_post'],
                    model3.tvalues['daca_x_post'], model4.tvalues['daca_x_post'],
                    model5.tvalues['daca_x_post']],
    'p-value': [model1.pvalues['daca_x_post'], model2.pvalues['daca_x_post'],
                model3.pvalues['daca_x_post'], model4.pvalues['daca_x_post'],
                model5.pvalues['daca_x_post']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared, model5.rsquared]
}

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH + "regression_results.csv", index=False)
print("\nRegression Results Summary:")
print(results_df.to_string(index=False))

# Save event study results
event_results = {
    'Year': event_years,
    'Coefficient': [model_event.params[f'daca_{y}'] for y in event_years],
    'Std Error': [model_event.bse[f'daca_{y}'] for y in event_years],
    'p-value': [model_event.pvalues[f'daca_{y}'] for y in event_years]
}
event_df = pd.DataFrame(event_results)
event_df.to_csv(OUTPUT_PATH + "event_study_results.csv", index=False)
print("\nEvent Study Results:")
print(event_df.to_string(index=False))

# ============================================================================
# STEP 10: Final summary
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"""
Research Question: Effect of DACA eligibility on full-time employment
                   among Hispanic-Mexican individuals born in Mexico

Sample: Non-citizen Hispanic-Mexican individuals born in Mexico, ages 16-64
        Years: 2006-2011 (pre) and 2013-2016 (post), excluding 2012

DACA Eligibility Criteria:
  - Arrived in US before age 16
  - Born after June 15, 1981 (under 31 as of June 15, 2012)
  - In US since at least 2007 (5+ years by 2012)
  - Non-citizen

Outcome: Full-time employment (35+ hours per week)

MAIN RESULTS (Preferred Specification: Model 4 - State + Year FE):
  DiD Estimate: {model4.params['daca_x_post']:.4f}
  Standard Error: {model4.bse['daca_x_post']:.4f}
  95% CI: [{model4.params['daca_x_post'] - 1.96*model4.bse['daca_x_post']:.4f}, {model4.params['daca_x_post'] + 1.96*model4.bse['daca_x_post']:.4f}]
  p-value: {model4.pvalues['daca_x_post']:.4f}
  Sample Size: {int(model4.nobs):,}

Interpretation: DACA eligibility is associated with a {abs(model4.params['daca_x_post'])*100:.2f} percentage point
{"increase" if model4.params['daca_x_post'] > 0 else "decrease"} in the probability of full-time employment.
""")

# Save full regression output
with open(OUTPUT_PATH + "full_regression_output.txt", 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("DACA Replication Study - Full Regression Output\n")
    f.write("=" * 80 + "\n\n")

    f.write("Model 1: Basic DiD\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("Model 2: DiD with Demographic Controls\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("Model 3: DiD with Year Fixed Effects\n")
    f.write(str(model3.summary()) + "\n\n")

    f.write("Model 4: Full Model (State + Year FE)\n")
    f.write(str(model4.summary()) + "\n\n")

    f.write("Model 5: Weighted Full Model\n")
    f.write(str(model5.summary()) + "\n\n")

    f.write("Event Study Model\n")
    f.write(str(model_event.summary()) + "\n")

print("\nResults saved to:", OUTPUT_PATH)
print("Analysis complete!")

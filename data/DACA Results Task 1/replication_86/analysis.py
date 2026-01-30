"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Author: Replication ID 86
Date: 2026-01-25
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 60)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")
df = pd.read_csv('data/data.csv')
print(f"    Total observations loaded: {len(df):,}")
print(f"    Years in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. SAMPLE SELECTION
# =============================================================================
print("\n[2] Sample selection...")

# Step 2a: Hispanic-Mexican ethnicity (HISPAN == 1)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"    After Hispanic-Mexican filter: {len(df_mex):,}")

# Step 2b: Born in Mexico (BPL == 200)
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"    After Mexico birthplace filter: {len(df_mex):,}")

# Step 2c: Non-citizens (CITIZEN == 3) - proxy for undocumented
# Per instructions: "Assume that anyone who is not a citizen and who has
# not received immigration papers is undocumented for DACA purposes"
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"    After non-citizen filter: {len(df_mex):,}")

# Step 2d: Exclude 2012 (cannot distinguish pre/post DACA within year)
df_mex = df_mex[df_mex['YEAR'] != 2012].copy()
print(f"    After excluding 2012: {len(df_mex):,}")

# Step 2e: Working-age population (16-64)
df_mex = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 64)].copy()
print(f"    After working-age (16-64) filter: {len(df_mex):,}")

# =============================================================================
# 3. DEFINE DACA ELIGIBILITY
# =============================================================================
print("\n[3] Defining DACA eligibility...")

# Calculate age at arrival
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']

# Age at arrival < 16 (came before 16th birthday)
df_mex['arrived_before_16'] = df_mex['age_at_arrival'] < 16

# Born after June 15, 1981 (under 31 on June 15, 2012)
# Conservative: if BIRTHYR >= 1982, definitely eligible
# If BIRTHYR == 1981 and BIRTHQTR >= 3 (July or later), also eligible
df_mex['born_after_cutoff'] = (
    (df_mex['BIRTHYR'] >= 1982) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
)

# In US since June 15, 2007 (continuous presence requirement)
# Must have arrived by 2007 at the latest
df_mex['in_us_since_2007'] = df_mex['YRIMMIG'] <= 2007

# DACA eligible: meets all criteria
df_mex['daca_eligible'] = (
    df_mex['arrived_before_16'] &
    df_mex['born_after_cutoff'] &
    df_mex['in_us_since_2007']
)

print(f"    Arrived before age 16: {df_mex['arrived_before_16'].sum():,}")
print(f"    Born after cutoff (1981 mid-year): {df_mex['born_after_cutoff'].sum():,}")
print(f"    In US since 2007: {df_mex['in_us_since_2007'].sum():,}")
print(f"    DACA eligible (all criteria): {df_mex['daca_eligible'].sum():,}")

# =============================================================================
# 4. CREATE TREATMENT AND OUTCOME VARIABLES
# =============================================================================
print("\n[4] Creating treatment and outcome variables...")

# Post-DACA indicator (2013-2016)
df_mex['post'] = (df_mex['YEAR'] >= 2013).astype(int)

# Treatment indicator
df_mex['eligible'] = df_mex['daca_eligible'].astype(int)

# Interaction term for DiD
df_mex['eligible_x_post'] = df_mex['eligible'] * df_mex['post']

# Outcome: Full-time employment (35+ hours per week)
df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Alternative outcome: Employed (any work)
df_mex['employed'] = (df_mex['EMPSTAT'] == 1).astype(int)

print(f"    Pre-period observations: {(df_mex['post'] == 0).sum():,}")
print(f"    Post-period observations: {(df_mex['post'] == 1).sum():,}")
print(f"    Eligible observations: {df_mex['eligible'].sum():,}")
print(f"    Not eligible observations: {(df_mex['eligible'] == 0).sum():,}")

# =============================================================================
# 5. SUMMARY STATISTICS
# =============================================================================
print("\n[5] Summary statistics...")

def weighted_mean(data, weights):
    return np.average(data, weights=weights)

def weighted_std(data, weights):
    average = np.average(data, weights=weights)
    variance = np.average((data - average)**2, weights=weights)
    return np.sqrt(variance)

# Overall sample
print("\n    Overall sample characteristics:")
print(f"    Mean age: {weighted_mean(df_mex['AGE'], df_mex['PERWT']):.1f}")
print(f"    Male share: {weighted_mean(df_mex['SEX'] == 1, df_mex['PERWT']):.3f}")
print(f"    Full-time employment rate: {weighted_mean(df_mex['fulltime'], df_mex['PERWT']):.3f}")
print(f"    Employment rate: {weighted_mean(df_mex['employed'], df_mex['PERWT']):.3f}")

# By eligibility status
print("\n    By DACA eligibility:")
for elig in [0, 1]:
    subset = df_mex[df_mex['eligible'] == elig]
    label = "Eligible" if elig == 1 else "Not Eligible"
    print(f"\n    {label}:")
    print(f"      N: {len(subset):,}")
    print(f"      Mean age: {weighted_mean(subset['AGE'], subset['PERWT']):.1f}")
    print(f"      Full-time rate: {weighted_mean(subset['fulltime'], subset['PERWT']):.3f}")

# =============================================================================
# 6. DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n[6] Difference-in-Differences analysis...")

# Create 2x2 table for basic DiD
print("\n    2x2 DiD Table (Full-time employment rates):")
did_table = df_mex.groupby(['eligible', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
did_table.columns = ['Pre', 'Post']
did_table.index = ['Not Eligible', 'Eligible']
print(did_table)

# Calculate simple DiD
diff_eligible = did_table.loc['Eligible', 'Post'] - did_table.loc['Eligible', 'Pre']
diff_not_eligible = did_table.loc['Not Eligible', 'Post'] - did_table.loc['Not Eligible', 'Pre']
simple_did = diff_eligible - diff_not_eligible

print(f"\n    Change for eligible: {diff_eligible:.4f}")
print(f"    Change for not eligible: {diff_not_eligible:.4f}")
print(f"    Simple DiD estimate: {simple_did:.4f}")

# =============================================================================
# 7. REGRESSION ANALYSIS
# =============================================================================
print("\n[7] Regression analysis...")

# Model 1: Basic DiD without controls
print("\n    Model 1: Basic DiD (no controls)")
model1 = smf.wls(
    'fulltime ~ eligible + post + eligible_x_post',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Store preferred estimate
preferred_estimate = model1.params['eligible_x_post']
preferred_se = model1.bse['eligible_x_post']
preferred_ci = model1.conf_int().loc['eligible_x_post']

# Add control variables
df_mex['age_sq'] = df_mex['AGE'] ** 2
df_mex['male'] = (df_mex['SEX'] == 1).astype(int)
df_mex['married'] = (df_mex['MARST'].isin([1, 2])).astype(int)

# Education categories
df_mex['educ_hs'] = (df_mex['EDUC'] >= 6).astype(int)  # High school or more
df_mex['educ_college'] = (df_mex['EDUC'] >= 10).astype(int)  # Some college or more

# Model 2: DiD with demographic controls
print("\n    Model 2: DiD with demographic controls")
model2 = smf.wls(
    'fulltime ~ eligible + post + eligible_x_post + AGE + age_sq + male + married',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n    Model 3: DiD with demographic + education controls")
model3 = smf.wls(
    'fulltime ~ eligible + post + eligible_x_post + AGE + age_sq + male + married + educ_hs + educ_college',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: Add year fixed effects
print("\n    Model 4: DiD with year fixed effects")
df_mex['year_factor'] = df_mex['YEAR'].astype(str)
model4 = smf.wls(
    'fulltime ~ eligible + eligible_x_post + AGE + age_sq + male + married + educ_hs + educ_college + C(year_factor)',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
# Print only key coefficients
print(f"    eligible_x_post: {model4.params['eligible_x_post']:.4f} (SE: {model4.bse['eligible_x_post']:.4f})")

# Model 5: Add state fixed effects
print("\n    Model 5: DiD with year + state fixed effects")
model5 = smf.wls(
    'fulltime ~ eligible + eligible_x_post + AGE + age_sq + male + married + educ_hs + educ_college + C(year_factor) + C(STATEFIP)',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
print(f"    eligible_x_post: {model5.params['eligible_x_post']:.4f} (SE: {model5.bse['eligible_x_post']:.4f})")

# =============================================================================
# 8. ROBUSTNESS CHECKS
# =============================================================================
print("\n[8] Robustness checks...")

# Alternative outcome: Any employment
print("\n    Alternative outcome: Any employment (vs full-time)")
model_emp = smf.wls(
    'employed ~ eligible + post + eligible_x_post + AGE + age_sq + male + married',
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')
print(f"    eligible_x_post (employment): {model_emp.params['eligible_x_post']:.4f} (SE: {model_emp.bse['eligible_x_post']:.4f})")

# Gender-specific effects
print("\n    Gender-specific effects:")
for gender, label in [(1, 'Male'), (2, 'Female')]:
    subset = df_mex[df_mex['SEX'] == gender]
    model_gender = smf.wls(
        'fulltime ~ eligible + post + eligible_x_post + AGE + age_sq + married',
        data=subset,
        weights=subset['PERWT']
    ).fit(cov_type='HC1')
    print(f"    {label}: {model_gender.params['eligible_x_post']:.4f} (SE: {model_gender.bse['eligible_x_post']:.4f})")

# Event study (year-by-year effects)
print("\n    Event study (year-by-year effects):")
df_mex['year_x_eligible'] = df_mex['YEAR'].astype(str) + '_x_' + df_mex['eligible'].astype(str)
years = sorted(df_mex['YEAR'].unique())
event_results = []
for year in years:
    if year != 2011:  # Reference year
        df_mex[f'year_{year}'] = (df_mex['YEAR'] == year).astype(int)
        df_mex[f'eligible_x_year_{year}'] = df_mex['eligible'] * df_mex[f'year_{year}']

# Reference year is 2011 (last pre-treatment year)
event_formula = 'fulltime ~ eligible + ' + ' + '.join([f'eligible_x_year_{y}' for y in years if y != 2011]) + ' + AGE + age_sq + male + married + C(year_factor)'
model_event = smf.wls(
    event_formula,
    data=df_mex,
    weights=df_mex['PERWT']
).fit(cov_type='HC1')

print("    Year   Coefficient   Std.Err.")
print("    " + "-" * 35)
for year in years:
    if year == 2011:
        print(f"    {year}     0.0000      (reference)")
    else:
        coef = model_event.params[f'eligible_x_year_{year}']
        se = model_event.bse[f'eligible_x_year_{year}']
        print(f"    {year}     {coef:.4f}     ({se:.4f})")

# =============================================================================
# 9. SAMPLE SIZE SUMMARIES
# =============================================================================
print("\n[9] Sample sizes...")

print("\n    By year and eligibility:")
sample_table = df_mex.groupby(['YEAR', 'eligible']).size().unstack()
sample_table.columns = ['Not Eligible', 'Eligible']
print(sample_table)

print(f"\n    Total sample size: {len(df_mex):,}")
print(f"    Total eligible: {df_mex['eligible'].sum():,}")
print(f"    Total not eligible: {(df_mex['eligible'] == 0).sum():,}")

# =============================================================================
# 10. SAVE RESULTS
# =============================================================================
print("\n[10] Saving results...")

# Create results dictionary
results = {
    'preferred_estimate': preferred_estimate,
    'preferred_se': preferred_se,
    'preferred_ci_lower': preferred_ci[0],
    'preferred_ci_upper': preferred_ci[1],
    'sample_size': len(df_mex),
    'n_eligible': df_mex['eligible'].sum(),
    'n_not_eligible': (df_mex['eligible'] == 0).sum(),
    'simple_did': simple_did,
    'model2_coef': model2.params['eligible_x_post'],
    'model2_se': model2.bse['eligible_x_post'],
    'model3_coef': model3.params['eligible_x_post'],
    'model3_se': model3.bse['eligible_x_post'],
    'model5_coef': model5.params['eligible_x_post'],
    'model5_se': model5.bse['eligible_x_post'],
}

# Save to file
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)

# Print final summary
print("\n" + "=" * 60)
print("PREFERRED ESTIMATE SUMMARY")
print("=" * 60)
print(f"Effect of DACA eligibility on full-time employment (DiD)")
print(f"Estimate: {preferred_estimate:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% CI: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"Sample Size: {len(df_mex):,}")
print(f"t-statistic: {preferred_estimate / preferred_se:.2f}")
print(f"p-value: {2 * (1 - stats.t.cdf(abs(preferred_estimate / preferred_se), len(df_mex) - 4)):.4f}")
print("=" * 60)

# Save detailed model results for report
with open('model_results.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - DETAILED RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("Model 1: Basic DiD\n")
    f.write(str(model1.summary()) + "\n\n")

    f.write("Model 2: DiD with demographic controls\n")
    f.write(str(model2.summary()) + "\n\n")

    f.write("Model 3: DiD with demographic + education controls\n")
    f.write(str(model3.summary()) + "\n\n")

print("\nAnalysis complete. Results saved to results_summary.csv and model_results.txt")

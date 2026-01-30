"""
DACA Replication Analysis - Replication 91
Research Question: Effect of DACA eligibility on full-time employment

Using Difference-in-Differences design comparing:
- Treatment: Ages 26-30 at time of policy (ELIGIBLE=1)
- Control: Ages 31-35 at time of policy (ELIGIBLE=0)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
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
print("DACA REPLICATION ANALYSIS - REPLICATION 91")
print("="*80)

# Load data
print("\n### Loading Data ###")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Basic data structure
print("\n### Data Structure ###")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"AFTER values: {df['AFTER'].unique()}")
print(f"FT values: {df['FT'].unique()}")

# Check sample sizes by group
print("\n### Sample Sizes by Group ###")
sample_table = df.groupby(['ELIGIBLE', 'AFTER']).agg(
    N=('FT', 'count'),
    FT_mean=('FT', 'mean'),
    PERWT_sum=('PERWT', 'sum')
).reset_index()
print(sample_table)

# Sample sizes by year
print("\n### Sample Sizes by Year ###")
year_table = df.groupby(['YEAR', 'ELIGIBLE']).agg(
    N=('FT', 'count'),
    FT_rate=('FT', 'mean')
).reset_index()
print(year_table.pivot(index='YEAR', columns='ELIGIBLE', values=['N', 'FT_rate']))

# Descriptive statistics
print("\n### Descriptive Statistics ###")
print("\n--- Full-Time Employment Rates ---")
print(df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count']))

# Weighted means
print("\n--- Weighted Full-Time Employment Rates ---")
def weighted_mean(x, weights):
    return np.average(x, weights=weights)

for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        wt_mean = weighted_mean(subset['FT'], subset['PERWT'])
        print(f"ELIGIBLE={eligible}, AFTER={after}: Weighted FT rate = {wt_mean:.4f} (N={len(subset):,})")

# ============================================================================
# MAIN ANALYSIS: DIFFERENCE-IN-DIFFERENCES
# ============================================================================

print("\n" + "="*80)
print("MAIN ANALYSIS: DIFFERENCE-IN-DIFFERENCES")
print("="*80)

# Simple 2x2 DiD calculation
print("\n### Simple 2x2 DiD (Unweighted) ###")
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
print("\nFull-time employment rates:")
print(means)

# DiD calculation
diff_treated = means.loc[1, 1] - means.loc[1, 0]
diff_control = means.loc[0, 1] - means.loc[0, 0]
did_simple = diff_treated - diff_control

print(f"\nChange for Treatment group (ELIGIBLE=1): {diff_treated:.4f}")
print(f"Change for Control group (ELIGIBLE=0): {diff_control:.4f}")
print(f"Difference-in-Differences estimate: {did_simple:.4f}")

# Weighted 2x2 DiD
print("\n### Simple 2x2 DiD (Weighted) ###")
weighted_means = {}
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        weighted_means[(eligible, after)] = weighted_mean(subset['FT'], subset['PERWT'])

print("Weighted Full-time employment rates:")
print(f"  ELIGIBLE=0, AFTER=0: {weighted_means[(0,0)]:.4f}")
print(f"  ELIGIBLE=0, AFTER=1: {weighted_means[(0,1)]:.4f}")
print(f"  ELIGIBLE=1, AFTER=0: {weighted_means[(1,0)]:.4f}")
print(f"  ELIGIBLE=1, AFTER=1: {weighted_means[(1,1)]:.4f}")

diff_treated_wt = weighted_means[(1,1)] - weighted_means[(1,0)]
diff_control_wt = weighted_means[(0,1)] - weighted_means[(0,0)]
did_weighted = diff_treated_wt - diff_control_wt

print(f"\nWeighted change for Treatment group: {diff_treated_wt:.4f}")
print(f"Weighted change for Control group: {diff_control_wt:.4f}")
print(f"Weighted DiD estimate: {did_weighted:.4f}")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("REGRESSION ANALYSIS")
print("="*80)

# Model 1: Basic DiD regression (unweighted)
print("\n### Model 1: Basic DiD Regression (Unweighted) ###")
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: Basic DiD regression (weighted)
print("\n### Model 2: Basic DiD Regression (Weighted) ###")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with robust standard errors
print("\n### Model 3: Basic DiD with Robust Standard Errors ###")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year fixed effects
print("\n### Model 4: DiD with Year Fixed Effects (Weighted) ###")
df['YEAR_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with covariates
print("\n### Model 5: DiD with Covariates (Weighted) ###")
# Prepare covariates
df['SEX_female'] = (df['SEX'] == 2).astype(int)
df['MARST_married'] = (df['MARST'] <= 2).astype(int)  # Currently married (spouse present or absent)

# Education recoding
# From data dict: 1=No High School, 2=High School, 3=Some College, 4=Associate, 5=BA+
# Let's use the labelled version to see labels if needed
df_labels = pd.read_csv('data/prepared_data_labelled_version.csv')
print("\nEducation recode values:", df_labels['EDUC_RECODE'].unique())

# Create education dummies
df['educ_hs'] = (df_labels['EDUC_RECODE'] == 'High School Degree').astype(int)
df['educ_somecoll'] = (df_labels['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_assoc'] = (df_labels['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba'] = (df_labels['EDUC_RECODE'] == 'BA+').astype(int)

# Nchild
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# State fixed effects
df['STATE_factor'] = pd.Categorical(df['STATEFIP'])

model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + SEX_female + MARST_married + '
                 'educ_hs + educ_somecoll + educ_assoc + educ_ba + has_children + AGE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Model 6: Full model with state fixed effects
print("\n### Model 6: DiD with Covariates and State Fixed Effects ###")
model6 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + SEX_female + MARST_married + '
                 'educ_hs + educ_somecoll + educ_assoc + educ_ba + has_children + AGE',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
# Only print key coefficients
print("\nKey coefficients from Model 6 (Full model with state FE):")
print(f"ELIGIBLE_AFTER: {model6.params['ELIGIBLE_AFTER']:.5f} (SE: {model6.bse['ELIGIBLE_AFTER']:.5f}, p={model6.pvalues['ELIGIBLE_AFTER']:.4f})")
print(f"ELIGIBLE: {model6.params['ELIGIBLE']:.5f} (SE: {model6.bse['ELIGIBLE']:.5f})")
print(f"SEX_female: {model6.params['SEX_female']:.5f}")
print(f"MARST_married: {model6.params['MARST_married']:.5f}")
print(f"AGE: {model6.params['AGE']:.5f}")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs):,}")

# ============================================================================
# EVENT STUDY / YEAR-BY-YEAR EFFECTS
# ============================================================================

print("\n" + "="*80)
print("EVENT STUDY ANALYSIS")
print("="*80)

# Create year dummies interacted with treatment
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Use 2011 as reference year (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]

# Event study regression
formula_event = 'FT ~ ELIGIBLE + C(YEAR) + ' + ' + '.join(year_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
print("-" * 50)
for year in sorted(df['YEAR'].unique()):
    if year == 2011:
        print(f"Year {year}: 0.0000 (reference)")
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        coef = model_event.params[var]
        se = model_event.bse[var]
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        print(f"Year {year}: {coef:.4f} (SE: {se:.4f}), 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# ============================================================================
# SUMMARY TABLE FOR REPORT
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF RESULTS FOR REPORT")
print("="*80)

print("\n### Table: DiD Estimates Across Specifications ###")
print("-" * 100)
print(f"{'Specification':<50} {'Estimate':>12} {'Std.Error':>12} {'95% CI':>25}")
print("-" * 100)

models_summary = [
    ("(1) Simple DiD (unweighted)", model1.params['ELIGIBLE_AFTER'], model1.bse['ELIGIBLE_AFTER']),
    ("(2) Simple DiD (weighted)", model2.params['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER']),
    ("(3) Simple DiD (weighted, robust SE)", model3.params['ELIGIBLE_AFTER'], model3.bse['ELIGIBLE_AFTER']),
    ("(4) Year FE (weighted, robust SE)", model4.params['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER']),
    ("(5) Year FE + covariates (weighted)", model5.params['ELIGIBLE_AFTER'], model5.bse['ELIGIBLE_AFTER']),
    ("(6) Year + State FE + covariates (weighted)", model6.params['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER']),
]

for name, est, se in models_summary:
    ci_low = est - 1.96 * se
    ci_high = est + 1.96 * se
    print(f"{name:<50} {est:>12.5f} {se:>12.5f} [{ci_low:>10.5f}, {ci_high:>10.5f}]")

print("-" * 100)

# Preferred specification
print("\n### PREFERRED SPECIFICATION ###")
print("Model 5: DiD with year fixed effects and individual covariates")
print(f"  Effect estimate: {model5.params['ELIGIBLE_AFTER']:.5f}")
print(f"  Standard error: {model5.bse['ELIGIBLE_AFTER']:.5f}")
ci_low = model5.params['ELIGIBLE_AFTER'] - 1.96 * model5.bse['ELIGIBLE_AFTER']
ci_high = model5.params['ELIGIBLE_AFTER'] + 1.96 * model5.bse['ELIGIBLE_AFTER']
print(f"  95% CI: [{ci_low:.5f}, {ci_high:.5f}]")
print(f"  Sample size: {int(model5.nobs):,}")

# ============================================================================
# BALANCE CHECKS
# ============================================================================

print("\n" + "="*80)
print("BALANCE CHECKS")
print("="*80)

# Check pre-treatment characteristics
pre_data = df[df['AFTER'] == 0]

print("\n### Pre-Treatment Balance (2008-2011) ###")
balance_vars = ['AGE', 'SEX_female', 'MARST_married', 'NCHILD', 'has_children']

print("-" * 80)
print(f"{'Variable':<20} {'Control Mean':>15} {'Treated Mean':>15} {'Difference':>15} {'P-value':>10}")
print("-" * 80)

for var in balance_vars:
    control = pre_data[pre_data['ELIGIBLE'] == 0][var]
    treated = pre_data[pre_data['ELIGIBLE'] == 1][var]

    control_mean = np.average(control, weights=pre_data[pre_data['ELIGIBLE'] == 0]['PERWT'])
    treated_mean = np.average(treated, weights=pre_data[pre_data['ELIGIBLE'] == 1]['PERWT'])
    diff = treated_mean - control_mean

    # t-test (unweighted for simplicity)
    t_stat, p_val = stats.ttest_ind(treated, control)

    print(f"{var:<20} {control_mean:>15.4f} {treated_mean:>15.4f} {diff:>15.4f} {p_val:>10.4f}")

print("-" * 80)

# Pre-treatment outcome trends
print("\n### Pre-Treatment Outcome Trends ###")
pre_trends = df[df['AFTER'] == 0].groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
print(pre_trends)

# ============================================================================
# SAVE RESULTS FOR REPORT
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save key results to file
results_summary = {
    'preferred_estimate': model5.params['ELIGIBLE_AFTER'],
    'preferred_se': model5.bse['ELIGIBLE_AFTER'],
    'preferred_ci_low': model5.params['ELIGIBLE_AFTER'] - 1.96 * model5.bse['ELIGIBLE_AFTER'],
    'preferred_ci_high': model5.params['ELIGIBLE_AFTER'] + 1.96 * model5.bse['ELIGIBLE_AFTER'],
    'sample_size': int(model5.nobs),
    'simple_did_weighted': did_weighted,
    'simple_did_unweighted': did_simple
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to analysis_results.json")
print("\nAnalysis complete!")

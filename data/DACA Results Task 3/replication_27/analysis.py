"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("="*80)
print("DACA REPLICATION ANALYSIS - LOADING DATA")
print("="*80)

df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nTotal observations: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"\nKey variables check:")
print(f"  ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"  AFTER values: {df['AFTER'].unique()}")
print(f"  FT values: {df['FT'].unique()}")

# Summary of treatment and control groups
print("\n" + "="*80)
print("SAMPLE COMPOSITION")
print("="*80)

print("\nBy ELIGIBLE status:")
print(df['ELIGIBLE'].value_counts())

print("\nBy AFTER status:")
print(df['AFTER'].value_counts())

print("\nCross-tabulation ELIGIBLE x AFTER:")
print(pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True))

# Check AGE_IN_JUNE_2012 distribution
print("\n" + "="*80)
print("AGE DISTRIBUTION (as of June 2012)")
print("="*80)
print("\nAGE_IN_JUNE_2012 distribution:")
print(df['AGE_IN_JUNE_2012'].describe())

print("\nAGE_IN_JUNE_2012 by ELIGIBLE status:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# Verify age ranges
print("\nAge ranges by ELIGIBLE status:")
for elig in [0, 1]:
    subset = df[df['ELIGIBLE'] == elig]
    print(f"  ELIGIBLE={elig}: Age range {subset['AGE_IN_JUNE_2012'].min()}-{subset['AGE_IN_JUNE_2012'].max()}")

# FT (Full-time employment) outcome
print("\n" + "="*80)
print("OUTCOME VARIABLE: FT (Full-time employment)")
print("="*80)
print("\nFT distribution overall:")
print(df['FT'].value_counts(normalize=True))

print("\nFT by ELIGIBLE and AFTER:")
ft_by_group = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print(ft_by_group)

# Calculate raw difference-in-differences
print("\n" + "="*80)
print("RAW DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("="*80)

# Create group means
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
print("\nMean FT by group:")
print(means)

# Calculate differences
diff_treated = means.loc[1, 1] - means.loc[1, 0]
diff_control = means.loc[0, 1] - means.loc[0, 0]
did = diff_treated - diff_control

print(f"\nChange for Treated (ELIGIBLE=1): {means.loc[1, 0]:.4f} -> {means.loc[1, 1]:.4f} = {diff_treated:.4f}")
print(f"Change for Control (ELIGIBLE=0): {means.loc[0, 0]:.4f} -> {means.loc[0, 1]:.4f} = {diff_control:.4f}")
print(f"Difference-in-Differences: {did:.4f}")

# Simple DiD regression without controls
print("\n" + "="*80)
print("MODEL 1: SIMPLE DIFFERENCE-IN-DIFFERENCES (No Controls)")
print("="*80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# OLS regression
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary())

# DiD with robust standard errors clustered by state
print("\n" + "="*80)
print("MODEL 2: DiD WITH STATE CLUSTERING")
print("="*80)

# Using state clustering
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model2.summary())

# Model with year fixed effects
print("\n" + "="*80)
print("MODEL 3: DiD WITH YEAR FIXED EFFECTS")
print("="*80)

# Create year dummies
df['YEAR_factor'] = df['YEAR'].astype(str)
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model3.summary())

# Model with state fixed effects
print("\n" + "="*80)
print("MODEL 4: DiD WITH STATE AND YEAR FIXED EFFECTS")
print("="*80)

model4 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("\nCoefficient on ELIGIBLE_AFTER (DiD estimate):")
print(f"  Estimate: {model4.params['ELIGIBLE_AFTER']:.6f}")
print(f"  Std Error: {model4.bse['ELIGIBLE_AFTER']:.6f}")
print(f"  t-stat: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model4.pvalues['ELIGIBLE_AFTER']:.6f}")
ci = model4.conf_int().loc['ELIGIBLE_AFTER']
print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")

# Model with individual-level covariates
print("\n" + "="*80)
print("MODEL 5: DiD WITH COVARIATES (Preferred Specification)")
print("="*80)

# Prepare covariates
# SEX: 1=Male, 2=Female in IPUMS coding
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: Married status
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Education recode - convert to numeric categories
print("\nEducation distribution (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts())

# Create a numeric education variable
educ_map = {
    'Less than High School': 1,
    'High School Degree': 2,
    'Some College': 3,
    'Two-Year Degree': 4,
    'BA+': 5
}
df['EDUC_NUM'] = df['EDUC_RECODE'].map(educ_map)
print(f"\nEDUC_NUM missing: {df['EDUC_NUM'].isna().sum()}")

# NCHILD - number of children
print(f"\nNCHILD range: {df['NCHILD'].min()} - {df['NCHILD'].max()}")

# For model with covariates, we need to handle the categorical education variable properly
# First check for missing values
print(f"\nMissing values check:")
print(f"  FT: {df['FT'].isna().sum()}")
print(f"  ELIGIBLE: {df['ELIGIBLE'].isna().sum()}")
print(f"  AFTER: {df['AFTER'].isna().sum()}")
print(f"  FEMALE: {df['FEMALE'].isna().sum()}")
print(f"  MARRIED: {df['MARRIED'].isna().sum()}")
print(f"  NCHILD: {df['NCHILD'].isna().sum()}")
print(f"  EDUC_NUM: {df['EDUC_NUM'].isna().sum()}")
print(f"  STATEFIP: {df['STATEFIP'].isna().sum()}")
print(f"  YEAR: {df['YEAR'].isna().sum()}")

# Create analysis dataset without missing values
df_analysis = df.dropna(subset=['FT', 'ELIGIBLE', 'AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC_NUM', 'STATEFIP', 'YEAR'])
print(f"\nObservations after dropping missing: {len(df_analysis):,} (dropped {len(df) - len(df_analysis):,})")

# Create the interaction term in the analysis dataset
df_analysis = df_analysis.copy()
df_analysis['ELIGIBLE_AFTER'] = df_analysis['ELIGIBLE'] * df_analysis['AFTER']

# Full model with covariates
model5 = smf.ols('''FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER
                    + FEMALE + MARRIED + NCHILD + C(EDUC_NUM)''', data=df_analysis).fit(
    cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

print("\nKey coefficients from preferred specification:")
print(f"\nELIGIBLE_AFTER (DiD estimate):")
print(f"  Estimate: {model5.params['ELIGIBLE_AFTER']:.6f}")
print(f"  Std Error: {model5.bse['ELIGIBLE_AFTER']:.6f}")
print(f"  t-stat: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model5.pvalues['ELIGIBLE_AFTER']:.6f}")
ci5 = model5.conf_int().loc['ELIGIBLE_AFTER']
print(f"  95% CI: [{ci5[0]:.6f}, {ci5[1]:.6f}]")

print(f"\nFEMALE coefficient: {model5.params['FEMALE']:.6f} (SE: {model5.bse['FEMALE']:.6f})")
print(f"MARRIED coefficient: {model5.params['MARRIED']:.6f} (SE: {model5.bse['MARRIED']:.6f})")
print(f"NCHILD coefficient: {model5.params['NCHILD']:.6f} (SE: {model5.bse['NCHILD']:.6f})")

print(f"\nModel R-squared: {model5.rsquared:.4f}")
print(f"Number of observations: {model5.nobs:,.0f}")

# Robustness check: Using survey weights
print("\n" + "="*80)
print("ROBUSTNESS CHECK: WEIGHTED REGRESSION")
print("="*80)

# Check if PERWT exists
if 'PERWT' in df_analysis.columns:
    print(f"Person weights (PERWT) available. Range: {df_analysis['PERWT'].min():.0f} - {df_analysis['PERWT'].max():.0f}")

    # Weighted regression using WLS
    model_weighted = smf.wls('''FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER
                        + FEMALE + MARRIED + NCHILD + C(EDUC_NUM)''',
                        data=df_analysis, weights=df_analysis['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

    print(f"\nWeighted DiD estimate (ELIGIBLE_AFTER):")
    print(f"  Estimate: {model_weighted.params['ELIGIBLE_AFTER']:.6f}")
    print(f"  Std Error: {model_weighted.bse['ELIGIBLE_AFTER']:.6f}")
    ci_w = model_weighted.conf_int().loc['ELIGIBLE_AFTER']
    print(f"  95% CI: [{ci_w[0]:.6f}, {ci_w[1]:.6f}]")
    print(f"  p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.6f}")

# Subgroup analysis by gender
print("\n" + "="*80)
print("HETEROGENEITY: BY GENDER")
print("="*80)

for sex, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df_analysis[df_analysis['SEX'] == sex].copy()
    model_sub = smf.ols('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + MARRIED + NCHILD + C(EDUC_NUM)',
                        data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['STATEFIP']})
    print(f"\n{label} (N={len(df_sub):,}):")
    print(f"  DiD estimate: {model_sub.params['ELIGIBLE_AFTER']:.6f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.6f})")
    ci_sub = model_sub.conf_int().loc['ELIGIBLE_AFTER']
    print(f"  95% CI: [{ci_sub[0]:.6f}, {ci_sub[1]:.6f}]")
    print(f"  p-value: {model_sub.pvalues['ELIGIBLE_AFTER']:.6f}")

# Event study / Dynamic effects
print("\n" + "="*80)
print("EVENT STUDY: YEAR-BY-YEAR EFFECTS")
print("="*80)

# Create year-specific treatment indicators
years = sorted(df_analysis['YEAR'].unique())
df_event = df_analysis.copy()
for year in years:
    df_event[f'ELIGIBLE_YEAR_{year}'] = ((df_event['ELIGIBLE'] == 1) & (df_event['YEAR'] == year)).astype(int)

# Reference year: 2011 (last pre-treatment year)
year_vars = [f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011]
formula_event = 'FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ' + ' + '.join(year_vars)
model_event = smf.ols(formula_event, data=df_event).fit(cov_type='cluster', cov_kwds={'groups': df_event['STATEFIP']})

print("\nYear-specific treatment effects (reference: 2011):")
for year in years:
    if year != 2011:
        var = f'ELIGIBLE_YEAR_{year}'
        if var in model_event.params:
            coef = model_event.params[var]
            se = model_event.bse[var]
            pval = model_event.pvalues[var]
            ci_e = model_event.conf_int().loc[var]
            print(f"  {year}: {coef:+.4f} (SE: {se:.4f}, 95% CI: [{ci_e[0]:.4f}, {ci_e[1]:.4f}], p={pval:.4f})")

# Parallel trends check
print("\n" + "="*80)
print("PARALLEL TRENDS: PRE-TREATMENT PERIOD CHECK")
print("="*80)

df_pre = df_analysis[df_analysis['AFTER'] == 0].copy()
df_pre['YEAR_centered'] = df_pre['YEAR'] - 2011
df_pre['ELIGIBLE_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_centered']

model_trends = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_TREND', data=df_pre).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']})

print("\nTest for differential pre-trends:")
print(f"  ELIGIBLE x Year interaction: {model_trends.params['ELIGIBLE_TREND']:.6f}")
print(f"  Standard Error: {model_trends.bse['ELIGIBLE_TREND']:.6f}")
print(f"  p-value: {model_trends.pvalues['ELIGIBLE_TREND']:.4f}")

if model_trends.pvalues['ELIGIBLE_TREND'] > 0.05:
    print("  -> Cannot reject null of parallel pre-trends at 5% level")
else:
    print("  -> Evidence of differential pre-trends")

# Descriptive statistics table
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS BY GROUP")
print("="*80)

desc_vars = ['FT', 'AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'UHRSWORK']
for var in desc_vars:
    if var in df_analysis.columns:
        print(f"\n{var}:")
        desc = df_analysis.groupby(['ELIGIBLE', 'AFTER'])[var].agg(['mean', 'std', 'count'])
        print(desc)

# Summary statistics for covariates
print("\n" + "="*80)
print("COVARIATE BALANCE: TREATED vs CONTROL (Pre-period)")
print("="*80)

df_pre_balance = df_analysis[df_analysis['AFTER'] == 0].copy()
balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'NCHILD']

print("\nPre-period means by ELIGIBLE status:")
print(f"{'Variable':<15} {'Control':<12} {'Treated':<12} {'Diff':<12} {'p-value':<10}")
print("-" * 60)

for var in balance_vars:
    if var in df_pre_balance.columns:
        control = df_pre_balance[df_pre_balance['ELIGIBLE'] == 0][var]
        treated = df_pre_balance[df_pre_balance['ELIGIBLE'] == 1][var]
        diff = treated.mean() - control.mean()
        tstat, pval = stats.ttest_ind(treated, control)
        print(f"{var:<15} {control.mean():<12.4f} {treated.mean():<12.4f} {diff:<12.4f} {pval:<10.4f}")

# Education balance
print("\nEducation distribution by ELIGIBLE status (pre-period):")
educ_cross = pd.crosstab(df_pre_balance['ELIGIBLE'], df_pre_balance['EDUC_NUM'], normalize='index')
educ_labels = {1: 'LessHS', 2: 'HS', 3: 'SomeColl', 4: '2Yr', 5: 'BA+'}
educ_cross.columns = [educ_labels.get(c, c) for c in educ_cross.columns]
print(educ_cross)

# Final summary
print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)

print("\n=== PREFERRED ESTIMATE (Model 5 - with covariates) ===")
print(f"DiD Estimate: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{ci5[0]:.4f}, {ci5[1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {int(model5.nobs):,}")

print("\n=== WEIGHTED ESTIMATE ===")
print(f"DiD Estimate: {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{ci_w[0]:.4f}, {ci_w[1]:.4f}]")
print(f"p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\n=== INTERPRETATION ===")
if model5.params['ELIGIBLE_AFTER'] > 0:
    print(f"DACA eligibility is associated with a {model5.params['ELIGIBLE_AFTER']*100:.2f} percentage point")
    print("INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(model5.params['ELIGIBLE_AFTER'])*100:.2f} percentage point")
    print("DECREASE in the probability of full-time employment.")

if model5.pvalues['ELIGIBLE_AFTER'] < 0.01:
    print("This effect is STATISTICALLY SIGNIFICANT at the 1% level.")
elif model5.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("This effect is STATISTICALLY SIGNIFICANT at the 5% level.")
elif model5.pvalues['ELIGIBLE_AFTER'] < 0.10:
    print("This effect is STATISTICALLY SIGNIFICANT at the 10% level.")
else:
    print("This effect is NOT STATISTICALLY SIGNIFICANT at conventional levels.")

# Save key results for report
results_dict = {
    'did_estimate': model5.params['ELIGIBLE_AFTER'],
    'did_se': model5.bse['ELIGIBLE_AFTER'],
    'did_pvalue': model5.pvalues['ELIGIBLE_AFTER'],
    'did_ci_lower': ci5[0],
    'did_ci_upper': ci5[1],
    'sample_size': int(model5.nobs),
    'r_squared': model5.rsquared,
    'weighted_did_estimate': model_weighted.params['ELIGIBLE_AFTER'],
    'weighted_did_se': model_weighted.bse['ELIGIBLE_AFTER'],
    'weighted_did_pvalue': model_weighted.pvalues['ELIGIBLE_AFTER'],
    'weighted_did_ci_lower': ci_w[0],
    'weighted_did_ci_upper': ci_w[1],
}

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# Export results summary for the report
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS RESULTS\n")
    f.write("="*50 + "\n\n")
    f.write(f"Sample Size: {int(model5.nobs):,}\n")
    f.write(f"Preferred DiD Estimate: {model5.params['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"95% CI: [{ci5[0]:.4f}, {ci5[1]:.4f}]\n")
    f.write(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}\n\n")
    f.write(f"Weighted DiD Estimate: {model_weighted.params['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"Weighted SE: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}\n")
    f.write(f"Weighted 95% CI: [{ci_w[0]:.4f}, {ci_w[1]:.4f}]\n")
    f.write(f"Weighted p-value: {model_weighted.pvalues['ELIGIBLE_AFTER']:.4f}\n")

print("\nResults saved to analysis_results.txt")

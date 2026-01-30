"""
DACA Replication Study - Analysis Script
Examining the effect of DACA eligibility on full-time employment
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# Load data
print("\n1. LOADING DATA")
print("-" * 40)
df = pd.read_csv('data/prepared_data_labelled_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Variables: {len(df.columns)}")

# Basic variable checks
print("\n2. DATA VALIDATION")
print("-" * 40)
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE values: {df['ELIGIBLE'].unique()}")
print(f"AFTER values: {df['AFTER'].unique()}")
print(f"FT values: {df['FT'].unique()}")

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Group counts
print("\n3. SAMPLE DISTRIBUTION")
print("-" * 40)
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
print("Observations by Group:")
print(group_counts)
print(f"\nTotal: {len(df):,}")

# Weighted group counts
print("\nWeighted Population (using PERWT):")
weighted_counts = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
print(weighted_counts)

# Descriptive statistics
print("\n4. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Full-time employment rates by group
print("\nFull-Time Employment Rates (Weighted):")
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        n = len(subset)
        group_name = f"{'Treatment' if eligible==1 else 'Control'}, {'Post' if after==1 else 'Pre'}"
        print(f"  {group_name}: {weighted_mean:.4f} (n={n:,})")

# Calculate simple DiD
print("\n5. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Calculate means for each cell
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

means = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean)
print("\nCell Means (Weighted):")
print(f"  Treatment, Pre:  {means[(1, 0)]:.4f}")
print(f"  Treatment, Post: {means[(1, 1)]:.4f}")
print(f"  Control, Pre:    {means[(0, 0)]:.4f}")
print(f"  Control, Post:   {means[(0, 1)]:.4f}")

# DiD calculation
treatment_diff = means[(1, 1)] - means[(1, 0)]
control_diff = means[(0, 1)] - means[(0, 0)]
did_estimate = treatment_diff - control_diff

print(f"\nFirst Differences:")
print(f"  Treatment group change: {treatment_diff:.4f}")
print(f"  Control group change:   {control_diff:.4f}")
print(f"\nDiD Estimate: {did_estimate:.4f}")

# Regression Analysis
print("\n6. REGRESSION ANALYSIS")
print("-" * 40)

# Model 1: Basic DiD (unweighted)
print("\nModel 1: Basic DiD (Unweighted OLS)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 2: Weighted DiD
print("\nModel 2: Basic DiD (Weighted)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 3: With clustered standard errors at state level
print("\nModel 3: Weighted DiD with State-Clustered SE")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 4: With demographic controls
print("\nModel 4: With Demographic Controls (Weighted, Clustered SE)")

# Prepare control variables
df['FEMALE'] = (df['SEX'] == 'Female').astype(int)
df['MARRIED'] = df['MARST'].isin(['Married, spouse present', 'Married, spouse absent']).astype(int)

# Education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 5: With state fixed effects
print("\nModel 5: With State Fixed Effects (Weighted, Clustered SE)")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN + C(STATEFIP)'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Model 6: With year fixed effects
print("\nModel 6: With State and Year Fixed Effects (Weighted, Clustered SE)")
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN + C(STATEFIP) + C(YEAR)'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"  DiD coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Parallel Trends Analysis
print("\n7. PARALLEL TRENDS ANALYSIS")
print("-" * 40)

# Year-by-year means
print("\nFull-Time Employment by Year and Group (Weighted):")
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean).unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_means.round(4))

# Pre-treatment trends
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_CENTERED'] = pre_data['YEAR'] - 2008

print("\nPre-treatment trend test:")
pre_data['YEAR_ELIGIBLE'] = pre_data['YEAR_CENTERED'] * pre_data['ELIGIBLE']
trend_model = smf.wls('FT ~ YEAR_CENTERED + ELIGIBLE + YEAR_ELIGIBLE',
                       data=pre_data, weights=pre_data['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': pre_data['STATEFIP']})
print(f"  Differential pre-trend coefficient: {trend_model.params['YEAR_ELIGIBLE']:.4f}")
print(f"  SE: {trend_model.bse['YEAR_ELIGIBLE']:.4f}")
print(f"  p-value: {trend_model.pvalues['YEAR_ELIGIBLE']:.4f}")

# Heterogeneity Analysis
print("\n8. HETEROGENEITY ANALYSIS")
print("-" * 40)

# By sex
print("\nEffect by Sex:")
for sex in ['Male', 'Female']:
    subset = df[df['SEX'] == sex]
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset, weights=subset['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
    print(f"  {sex}: DiD = {model.params['ELIGIBLE_AFTER']:.4f} (SE = {model.bse['ELIGIBLE_AFTER']:.4f})")

# By education
print("\nEffect by Education:")
for educ in df['EDUC_RECODE'].unique():
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=subset, weights=subset['PERWT']).fit()
        print(f"  {educ}: DiD = {model.params['ELIGIBLE_AFTER']:.4f} (n={len(subset)})")

# Summary statistics for covariate balance
print("\n9. COVARIATE BALANCE")
print("-" * 40)

print("\nPre-treatment means by group:")
pre_df = df[df['AFTER'] == 0]
for var in ['AGE', 'FEMALE', 'MARRIED', 'HAS_CHILDREN']:
    treat_mean = np.average(pre_df[pre_df['ELIGIBLE']==1][var], weights=pre_df[pre_df['ELIGIBLE']==1]['PERWT'])
    ctrl_mean = np.average(pre_df[pre_df['ELIGIBLE']==0][var], weights=pre_df[pre_df['ELIGIBLE']==0]['PERWT'])
    print(f"  {var}: Treatment={treat_mean:.3f}, Control={ctrl_mean:.3f}, Diff={treat_mean-ctrl_mean:.3f}")

# Final Summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print("\nPREFERRED SPECIFICATION: Model 6 (State + Year FE, Demographic Controls)")
print(f"  Sample Size: {len(df):,}")
print(f"  DiD Estimate: {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error (Clustered): {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

print("\nINTERPRETATION:")
effect = model6.params['ELIGIBLE_AFTER']
if effect > 0:
    print(f"  DACA eligibility is associated with a {effect*100:.2f} percentage point")
    print(f"  INCREASE in full-time employment probability.")
else:
    print(f"  DACA eligibility is associated with a {abs(effect)*100:.2f} percentage point")
    print(f"  DECREASE in full-time employment probability.")

if model6.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("  This effect is statistically significant at the 5% level.")
elif model6.pvalues['ELIGIBLE_AFTER'] < 0.10:
    print("  This effect is statistically significant at the 10% level.")
else:
    print("  This effect is NOT statistically significant at conventional levels.")

# Save regression tables for LaTeX
print("\n10. GENERATING OUTPUT FOR LATEX")
print("-" * 40)

# Create summary table
results_data = {
    'Model': ['(1) Basic', '(2) Weighted', '(3) Clustered SE', '(4) + Controls', '(5) + State FE', '(6) + Year FE'],
    'DiD Estimate': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model6.params['ELIGIBLE_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model6.bse['ELIGIBLE_AFTER']
    ],
    'p-value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER'],
        model6.pvalues['ELIGIBLE_AFTER']
    ]
}
results_df = pd.DataFrame(results_data)
results_df.to_csv('regression_results.csv', index=False)
print("Saved regression_results.csv")

# Save yearly means for plotting
yearly_means.to_csv('yearly_means.csv')
print("Saved yearly_means.csv")

# Save group counts
group_summary = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['count', 'mean', 'std'],
    'PERWT': 'sum'
}).round(4)
group_summary.to_csv('group_summary.csv')
print("Saved group_summary.csv")

print("\nAnalysis complete!")

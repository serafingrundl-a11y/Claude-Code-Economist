"""
DACA Full-Time Employment Replication Analysis
Difference-in-Differences Estimation

Research Question: Effect of DACA eligibility on full-time employment
Treatment: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
Control: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
Pre-period: 2008-2011 (AFTER=0)
Post-period: 2013-2016 (AFTER=1)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"Total observations: {len(df):,}")
print(f"Columns: {df.shape[1]}")

# Basic data exploration
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

# Key variables summary
print("\nKey Variables Summary:")
print(f"ELIGIBLE values: {df['ELIGIBLE'].value_counts().sort_index().to_dict()}")
print(f"AFTER values: {df['AFTER'].value_counts().sort_index().to_dict()}")
print(f"FT values: {df['FT'].value_counts().sort_index().to_dict()}")

# Years in data
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")

# Sample sizes by group
print("\n" + "="*60)
print("SAMPLE SIZES BY GROUP")
print("="*60)
group_counts = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
print(group_counts)

# Verify age distribution
print("\n" + "="*60)
print("AGE AT JUNE 2012 DISTRIBUTION BY ELIGIBLE STATUS")
print("="*60)
print("\nELIGIBLE=1 (Treatment - should be ages 26-30):")
print(df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].describe())
print(f"Unique values: {sorted(df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].unique())}")

print("\nELIGIBLE=0 (Control - should be ages 31-35):")
print(df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].describe())
print(f"Unique values: {sorted(df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].unique())}")

# Full-time employment rates by group
print("\n" + "="*60)
print("FULL-TIME EMPLOYMENT RATES BY GROUP (UNWEIGHTED)")
print("="*60)
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(ft_rates.round(4))

# Calculate naive DiD
did_naive = (ft_rates.loc['Treatment (26-30)', 'Post (2013-2016)'] -
             ft_rates.loc['Treatment (26-30)', 'Pre (2008-2011)']) - \
            (ft_rates.loc['Control (31-35)', 'Post (2013-2016)'] -
             ft_rates.loc['Control (31-35)', 'Pre (2008-2011)'])
print(f"\nNaive DiD estimate (unweighted): {did_naive:.4f}")

# Weighted full-time employment rates
print("\n" + "="*60)
print("FULL-TIME EMPLOYMENT RATES BY GROUP (WEIGHTED)")
print("="*60)

def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_rates_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_rates_weighted.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(ft_rates_weighted.round(4))

did_weighted = (ft_rates_weighted.loc['Treatment (26-30)', 'Post (2013-2016)'] -
                ft_rates_weighted.loc['Treatment (26-30)', 'Pre (2008-2011)']) - \
               (ft_rates_weighted.loc['Control (31-35)', 'Post (2013-2016)'] -
                ft_rates_weighted.loc['Control (31-35)', 'Pre (2008-2011)'])
print(f"\nWeighted DiD estimate: {did_weighted:.4f}")

# Year-by-year trends
print("\n" + "="*60)
print("YEAR-BY-YEAR FULL-TIME EMPLOYMENT RATES (WEIGHTED)")
print("="*60)

yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean).unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly_rates.round(4))

# Save summary statistics
summary_stats = {
    'Total_N': len(df),
    'Treatment_N': len(df[df['ELIGIBLE']==1]),
    'Control_N': len(df[df['ELIGIBLE']==0]),
    'Pre_N': len(df[df['AFTER']==0]),
    'Post_N': len(df[df['AFTER']==1])
}

print("\n" + "="*60)
print("REGRESSION ANALYSIS")
print("="*60)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\nModel 1: Basic DiD (Unweighted OLS)")
print("-"*40)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model1.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {model1.nobs:.0f}")

# Model 2: Basic DiD (weighted)
print("\nModel 2: Basic DiD (Weighted OLS)")
print("-"*40)
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with year fixed effects (weighted)
print("\nModel 3: DiD with Year Fixed Effects (Weighted)")
print("-"*40)
df['YEAR_factor'] = df['YEAR'].astype(str)
model3 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state fixed effects (weighted)
print("\nModel 4: DiD with State Fixed Effects (Weighted)")
print("-"*40)
df['STATE_factor'] = df['STATEFIP'].astype(str)
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATE_factor)', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: DiD with year and state fixed effects (weighted)
print("\nModel 5: DiD with Year and State Fixed Effects (Weighted)")
print("-"*40)
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + C(STATE_factor) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model5.rsquared:.4f}")

# Model 6: DiD with covariates
print("\nModel 6: DiD with Covariates (Weighted)")
print("-"*40)
# Add demographic controls
# SEX: 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST: 1=Married spouse present, 2=Married spouse absent, 3=Separated, 4=Divorced, 5=Widowed, 6=Never married
df['MARRIED'] = (df['MARST'] <= 2).astype(int)

# Check for education recode variable
if 'EDUC_RECODE' in df.columns:
    print("Using EDUC_RECODE for education controls")
else:
    print("Creating education categories from EDUC")
    df['EDUC_RECODE'] = pd.cut(df['EDUC'], bins=[-1, 5, 6, 9, 10, 20],
                               labels=['Less than HS', 'HS Diploma', 'Some College', 'Two-Year', 'BA+'])

# Convert EDUC_RECODE to factor if not already
df['EDUC_factor'] = df['EDUC_RECODE'].astype(str)

# Formula with demographic controls
formula6 = 'FT ~ ELIGIBLE + C(YEAR_factor) + C(STATE_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_factor) + NCHILD'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model6.rsquared:.4f}")

# Print covariate effects for context
print("\nSelected covariate effects:")
print(f"  FEMALE: {model6.params['FEMALE']:.4f} (SE: {model6.bse['FEMALE']:.4f})")
print(f"  MARRIED: {model6.params['MARRIED']:.4f} (SE: {model6.bse['MARRIED']:.4f})")
print(f"  NCHILD: {model6.params['NCHILD']:.4f} (SE: {model6.bse['NCHILD']:.4f})")

# Model 7: Full model with state policies
print("\nModel 7: DiD with Covariates and State Policies (Weighted)")
print("-"*40)
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'EVERIFY']
available_policies = [v for v in policy_vars if v in df.columns]
print(f"Available state policy variables: {available_policies}")

formula7 = 'FT ~ ELIGIBLE + C(YEAR_factor) + C(STATE_factor) + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_factor) + NCHILD'
if available_policies:
    formula7 += ' + ' + ' + '.join(available_policies)

model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"R-squared: {model7.rsquared:.4f}")

# Robustness check: Clustered standard errors at state level
print("\n" + "="*60)
print("ROBUSTNESS: CLUSTERED STANDARD ERRORS")
print("="*60)

# Find index of ELIGIBLE_AFTER in model parameters
ea_idx = list(model5.params.index).index('ELIGIBLE_AFTER')

print("\nModel 5 with State-Clustered Standard Errors:")
model5_cluster = model5.get_robustcov_results(cov_type='cluster', groups=df['STATEFIP'])
cluster_se = model5_cluster.bse[ea_idx]
cluster_t = model5_cluster.tvalues[ea_idx]
cluster_p = model5_cluster.pvalues[ea_idx]
cluster_ci = model5_cluster.conf_int()[ea_idx]
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {cluster_se:.4f}")
print(f"t-statistic: {cluster_t:.4f}")
print(f"p-value: {cluster_p:.4f}")
print(f"95% CI: [{cluster_ci[0]:.4f}, {cluster_ci[1]:.4f}]")

# Heteroscedasticity-robust standard errors
print("\nModel 5 with Robust (HC1) Standard Errors:")
model5_robust = model5.get_robustcov_results(cov_type='HC1')
robust_se = model5_robust.bse[ea_idx]
robust_t = model5_robust.tvalues[ea_idx]
robust_p = model5_robust.pvalues[ea_idx]
robust_ci = model5_robust.conf_int()[ea_idx]
print(f"DiD Coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {robust_se:.4f}")
print(f"t-statistic: {robust_t:.4f}")
print(f"p-value: {robust_p:.4f}")
print(f"95% CI: [{robust_ci[0]:.4f}, {robust_ci[1]:.4f}]")

# Summary of all models
print("\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)
print(f"{'Model':<50} {'DiD Est':>10} {'SE':>10} {'p-value':>10}")
print("-"*80)
print(f"{'1. Basic DiD (Unweighted)':<50} {model1.params['ELIGIBLE_AFTER']:>10.4f} {model1.bse['ELIGIBLE_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'2. Basic DiD (Weighted)':<50} {model2.params['ELIGIBLE_AFTER']:>10.4f} {model2.bse['ELIGIBLE_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'3. Year FE (Weighted)':<50} {model3.params['ELIGIBLE_AFTER']:>10.4f} {model3.bse['ELIGIBLE_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'4. State FE (Weighted)':<50} {model4.params['ELIGIBLE_AFTER']:>10.4f} {model4.bse['ELIGIBLE_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. Year + State FE (Weighted)':<50} {model5.params['ELIGIBLE_AFTER']:>10.4f} {model5.bse['ELIGIBLE_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'6. + Demographics (Weighted)':<50} {model6.params['ELIGIBLE_AFTER']:>10.4f} {model6.bse['ELIGIBLE_AFTER']:>10.4f} {model6.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'7. + State Policies (Weighted)':<50} {model7.params['ELIGIBLE_AFTER']:>10.4f} {model7.bse['ELIGIBLE_AFTER']:>10.4f} {model7.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. Clustered SE (State)':<50} {model5.params['ELIGIBLE_AFTER']:>10.4f} {cluster_se:>10.4f} {cluster_p:>10.4f}")
print(f"{'5. Robust SE (HC1)':<50} {model5.params['ELIGIBLE_AFTER']:>10.4f} {robust_se:>10.4f} {robust_p:>10.4f}")

# Save key results to file for LaTeX
results_dict = {
    'model': ['Basic DiD (Unweighted)', 'Basic DiD (Weighted)', 'Year FE', 'State FE',
              'Year + State FE', 'With Demographics', 'With State Policies',
              'Clustered SE', 'Robust SE'],
    'estimate': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                 model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                 model5.params['ELIGIBLE_AFTER'], model6.params['ELIGIBLE_AFTER'],
                 model7.params['ELIGIBLE_AFTER'], model5.params['ELIGIBLE_AFTER'],
                 model5.params['ELIGIBLE_AFTER']],
    'se': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'],
           model7.bse['ELIGIBLE_AFTER'], cluster_se, robust_se],
    'pvalue': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
               model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
               model5.pvalues['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER'],
               model7.pvalues['ELIGIBLE_AFTER'], cluster_p, robust_p]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_summary.csv', index=False)

# Save detailed model output for Model 5 (preferred specification)
print("\n" + "="*60)
print("PREFERRED MODEL DETAILS (Model 5: Year + State FE)")
print("="*60)
print(model5.summary())

print("\n\nAnalysis complete. Results saved to results_summary.csv")
print(f"\nPreferred estimate: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {int(model5.nobs):,}")

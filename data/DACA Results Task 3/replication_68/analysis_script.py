"""
DACA Full-Time Employment Analysis - Replication 68
====================================================
Independent replication analyzing the causal impact of DACA eligibility
on full-time employment among Hispanic-Mexican, Mexican-born individuals.

Research Design: Difference-in-Differences
- Treatment: ELIGIBLE=1 (ages 26-30 at June 2012)
- Control: ELIGIBLE=0 (ages 31-35 at June 2012)
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
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

print("=" * 80)
print("DACA FULL-TIME EMPLOYMENT ANALYSIS - REPLICATION 68")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING")
print("=" * 80)

data_path = "data/prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df):,} observations")
print(f"Number of variables: {df.shape[1]}")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")

# =============================================================================
# 2. DATA VERIFICATION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: DATA VERIFICATION")
print("=" * 80)

# Check key variables
print("\n--- Key Variable Distributions ---")
print(f"\nELIGIBLE (treatment indicator):")
print(df['ELIGIBLE'].value_counts().sort_index())

print(f"\nAFTER (post-treatment indicator):")
print(df['AFTER'].value_counts().sort_index())

print(f"\nFT (full-time employment outcome):")
print(df['FT'].value_counts().sort_index())

# Check for missing values in key variables
print("\n--- Missing Values in Key Variables ---")
key_vars = ['ELIGIBLE', 'AFTER', 'FT', 'PERWT', 'YEAR', 'AGE', 'SEX']
for var in key_vars:
    if var in df.columns:
        missing = df[var].isna().sum()
        print(f"{var}: {missing} missing ({100*missing/len(df):.2f}%)")

# =============================================================================
# 3. SAMPLE CHARACTERISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: SAMPLE CHARACTERISTICS")
print("=" * 80)

# Sample sizes by group
print("\n--- Sample Sizes by Treatment and Period ---")
sample_table = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
sample_table.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
sample_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
print(sample_table)

# Sample sizes by year
print("\n--- Sample Size by Year ---")
print(df.groupby('YEAR').size())

# Weighted sample sizes
print("\n--- Weighted Population by Treatment and Period ---")
weighted_table = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
weighted_table.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_table.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(weighted_table.round(0))

# =============================================================================
# 4. OUTCOME VARIABLE ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: FULL-TIME EMPLOYMENT RATES BY GROUP")
print("=" * 80)

# Unweighted means
print("\n--- Unweighted Full-Time Employment Rates ---")
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
ft_means.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(ft_means.round(4))

# Calculate simple DiD
diff_treatment = ft_means.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_means.loc['Treatment (26-30)', 'Pre (2008-2011)']
diff_control = ft_means.loc['Control (31-35)', 'Post (2013-2016)'] - ft_means.loc['Control (31-35)', 'Pre (2008-2011)']
simple_did = diff_treatment - diff_control

print(f"\nSimple DiD calculation (unweighted):")
print(f"  Treatment group change: {diff_treatment:.4f}")
print(f"  Control group change: {diff_control:.4f}")
print(f"  Difference-in-differences: {simple_did:.4f}")

# Weighted means
print("\n--- Weighted Full-Time Employment Rates ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_weighted.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
print(ft_weighted.round(4))

# Weighted DiD
diff_treatment_w = ft_weighted.loc['Treatment (26-30)', 'Post (2013-2016)'] - ft_weighted.loc['Treatment (26-30)', 'Pre (2008-2011)']
diff_control_w = ft_weighted.loc['Control (31-35)', 'Post (2013-2016)'] - ft_weighted.loc['Control (31-35)', 'Pre (2008-2011)']
weighted_did = diff_treatment_w - diff_control_w

print(f"\nSimple DiD calculation (weighted):")
print(f"  Treatment group change: {diff_treatment_w:.4f}")
print(f"  Control group change: {diff_control_w:.4f}")
print(f"  Difference-in-differences: {weighted_did:.4f}")

# =============================================================================
# 5. DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (OLS, Unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"Coefficient on ELIGIBLE x AFTER: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (WLS, Weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"Coefficient on ELIGIBLE x AFTER: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with robust standard errors
print("\n--- Model 3: Basic DiD (Weighted, HC1 Robust SE) ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Coefficient on ELIGIBLE x AFTER: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 6. COVARIATE-ADJUSTED MODELS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: COVARIATE-ADJUSTED MODELS")
print("=" * 80)

# Prepare categorical variables for regression
# SEX: 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# MARST categories
df['MARRIED'] = (df['MARST'] == 1).astype(int)  # Married, spouse present

# Education categories (using EDUC_RECODE if available, otherwise create)
if 'EDUC_RECODE' in df.columns:
    print("\nEducation distribution (EDUC_RECODE):")
    print(df['EDUC_RECODE'].value_counts())
    # Create dummy variables
    df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
    df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
    df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
    df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Check for children
if 'NCHILD' in df.columns:
    df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Model 4: DiD with demographic controls
print("\n--- Model 4: DiD with Demographics (Weighted, Robust SE) ---")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Coefficient on ELIGIBLE x AFTER: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 5: DiD with demographics + education
print("\n--- Model 5: DiD with Demographics + Education (Weighted, Robust SE) ---")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Coefficient on ELIGIBLE x AFTER: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 6: Add year fixed effects
print("\n--- Model 6: DiD with Year Fixed Effects (Weighted, Robust SE) ---")
df['YEAR_cat'] = pd.Categorical(df['YEAR'])
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)
year_cols = [c for c in year_dummies.columns]
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + ' + ' + '.join(year_cols)
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Coefficient on ELIGIBLE x AFTER: {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 7: Add state fixed effects
print("\n--- Model 7: DiD with Year + State Fixed Effects (Weighted, Robust SE) ---")
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)
state_cols = [c for c in state_dummies.columns]
formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"Coefficient on ELIGIBLE x AFTER: {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# 7. FULL-TIME EMPLOYMENT BY YEAR (FOR PARALLEL TRENDS)
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: YEARLY TRENDS (FOR PARALLEL TRENDS ASSESSMENT)")
print("=" * 80)

# Calculate weighted FT rate by year and group
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nWeighted Full-Time Employment Rate by Year:")
print(yearly_rates.round(4))

# Calculate yearly sample sizes
yearly_n = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
yearly_n.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nSample Size by Year:")
print(yearly_n)

# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SUMMARY OF ALL MODELS")
print("=" * 80)

print("\n" + "-" * 100)
print(f"{'Model':<50} {'DiD Estimate':>12} {'Std. Error':>12} {'p-value':>10}")
print("-" * 100)
print(f"{'1. Basic OLS (unweighted)':<50} {model1.params['ELIGIBLE_AFTER']:>12.4f} {model1.bse['ELIGIBLE_AFTER']:>12.4f} {model1.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'2. Basic WLS (weighted)':<50} {model2.params['ELIGIBLE_AFTER']:>12.4f} {model2.bse['ELIGIBLE_AFTER']:>12.4f} {model2.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'3. WLS with robust SE':<50} {model3.params['ELIGIBLE_AFTER']:>12.4f} {model3.bse['ELIGIBLE_AFTER']:>12.4f} {model3.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'4. + Demographics':<50} {model4.params['ELIGIBLE_AFTER']:>12.4f} {model4.bse['ELIGIBLE_AFTER']:>12.4f} {model4.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'5. + Education':<50} {model5.params['ELIGIBLE_AFTER']:>12.4f} {model5.bse['ELIGIBLE_AFTER']:>12.4f} {model5.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'6. + Year FE':<50} {model6.params['ELIGIBLE_AFTER']:>12.4f} {model6.bse['ELIGIBLE_AFTER']:>12.4f} {model6.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print(f"{'7. + Year + State FE (PREFERRED)':<50} {model7.params['ELIGIBLE_AFTER']:>12.4f} {model7.bse['ELIGIBLE_AFTER']:>12.4f} {model7.pvalues['ELIGIBLE_AFTER']:>10.4f}")
print("-" * 100)

# =============================================================================
# 9. PREFERRED ESTIMATE SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 9: PREFERRED ESTIMATE")
print("=" * 80)

# Using Model 7 as preferred (full controls with year and state FE)
preferred_model = model7
preferred_coef = preferred_model.params['ELIGIBLE_AFTER']
preferred_se = preferred_model.bse['ELIGIBLE_AFTER']
preferred_ci = preferred_model.conf_int().loc['ELIGIBLE_AFTER']
preferred_pval = preferred_model.pvalues['ELIGIBLE_AFTER']

print(f"\nPreferred Model: DiD with Demographics, Education, Year FE, and State FE")
print(f"  Effect Estimate: {preferred_coef:.4f}")
print(f"  Standard Error:  {preferred_se:.4f}")
print(f"  95% CI:          [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"  p-value:         {preferred_pval:.4f}")
print(f"  Sample Size:     {int(preferred_model.nobs):,}")

print(f"\nInterpretation:")
print(f"  DACA eligibility is associated with a {preferred_coef:.1%} {'increase' if preferred_coef > 0 else 'decrease'}")
print(f"  in the probability of full-time employment.")
if preferred_pval < 0.05:
    print(f"  This effect is statistically significant at the 5% level.")
else:
    print(f"  This effect is NOT statistically significant at the 5% level.")

# =============================================================================
# 10. COVARIATE BALANCE CHECK
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 10: COVARIATE BALANCE (Pre-Period)")
print("=" * 80)

# Check balance in pre-period
pre_df = df[df['AFTER'] == 0]

def weighted_stats(group, var, weight='PERWT'):
    """Calculate weighted mean and std"""
    mean = np.average(group[var], weights=group[weight])
    variance = np.average((group[var] - mean)**2, weights=group[weight])
    return pd.Series({'mean': mean, 'std': np.sqrt(variance)})

balance_vars = ['FEMALE', 'MARRIED', 'HAS_CHILDREN', 'AGE']
print("\n--- Pre-Period Covariate Means by Treatment Status ---")
print(f"{'Variable':<20} {'Treatment':>12} {'Control':>12} {'Difference':>12}")
print("-" * 60)

for var in balance_vars:
    if var in pre_df.columns:
        treat_mean = np.average(pre_df[pre_df['ELIGIBLE']==1][var], weights=pre_df[pre_df['ELIGIBLE']==1]['PERWT'])
        ctrl_mean = np.average(pre_df[pre_df['ELIGIBLE']==0][var], weights=pre_df[pre_df['ELIGIBLE']==0]['PERWT'])
        diff = treat_mean - ctrl_mean
        print(f"{var:<20} {treat_mean:>12.3f} {ctrl_mean:>12.3f} {diff:>12.3f}")

# =============================================================================
# 11. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 11: SAVING RESULTS")
print("=" * 80)

# Save key results to CSV for the report
results_dict = {
    'Model': ['Basic OLS', 'Basic WLS', 'WLS Robust SE', '+ Demographics', '+ Education', '+ Year FE', '+ Year + State FE'],
    'DiD_Estimate': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                     model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                     model5.params['ELIGIBLE_AFTER'], model6.params['ELIGIBLE_AFTER'],
                     model7.params['ELIGIBLE_AFTER']],
    'Std_Error': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
                  model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
                  model5.bse['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'],
                  model7.bse['ELIGIBLE_AFTER']],
    'CI_Lower': [model1.conf_int().loc['ELIGIBLE_AFTER', 0], model2.conf_int().loc['ELIGIBLE_AFTER', 0],
                 model3.conf_int().loc['ELIGIBLE_AFTER', 0], model4.conf_int().loc['ELIGIBLE_AFTER', 0],
                 model5.conf_int().loc['ELIGIBLE_AFTER', 0], model6.conf_int().loc['ELIGIBLE_AFTER', 0],
                 model7.conf_int().loc['ELIGIBLE_AFTER', 0]],
    'CI_Upper': [model1.conf_int().loc['ELIGIBLE_AFTER', 1], model2.conf_int().loc['ELIGIBLE_AFTER', 1],
                 model3.conf_int().loc['ELIGIBLE_AFTER', 1], model4.conf_int().loc['ELIGIBLE_AFTER', 1],
                 model5.conf_int().loc['ELIGIBLE_AFTER', 1], model6.conf_int().loc['ELIGIBLE_AFTER', 1],
                 model7.conf_int().loc['ELIGIBLE_AFTER', 1]],
    'p_value': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
                model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER'],
                model7.pvalues['ELIGIBLE_AFTER']]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save yearly trends
yearly_rates.to_csv('yearly_ft_rates.csv')
print("Yearly trends saved to yearly_ft_rates.csv")

# Save full model summary for preferred specification
with open('preferred_model_summary.txt', 'w') as f:
    f.write(model7.summary().as_text())
print("Preferred model summary saved to preferred_model_summary.txt")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

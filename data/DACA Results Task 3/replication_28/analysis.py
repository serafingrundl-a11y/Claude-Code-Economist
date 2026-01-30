"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
Method: Difference-in-Differences
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================
print("\n" + "="*80)
print("1. DATA LOADING AND EXPLORATION")
print("="*80)

# Load the numeric version of the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Number of observations: {len(df):,}")
print(f"Number of variables: {df.shape[1]}")

# Check key variables
print("\n--- Key Variables Summary ---")
print(f"\nFT (Full-time employment):")
print(df['FT'].value_counts().sort_index())
print(f"Mean FT: {df['FT'].mean():.4f}")

print(f"\nELIGIBLE (Treatment group indicator):")
print(df['ELIGIBLE'].value_counts().sort_index())

print(f"\nAFTER (Post-DACA period indicator):")
print(df['AFTER'].value_counts().sort_index())

print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# =============================================================================
# 2. SAMPLE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("2. SAMPLE STATISTICS BY GROUP")
print("="*80)

# Create group variable
df['group'] = df['ELIGIBLE'].map({1: 'Treatment (26-30)', 0: 'Control (31-35)'})
df['period'] = df['AFTER'].map({1: 'Post-DACA (2013-2016)', 0: 'Pre-DACA (2008-2011)'})

# Sample sizes by group and period
print("\n--- Sample Sizes ---")
sample_table = pd.crosstab(df['group'], df['period'], margins=True)
print(sample_table)

# Mean FT by group and period (unweighted)
print("\n--- Mean Full-Time Employment (Unweighted) ---")
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
ft_means.columns = ['Pre-DACA', 'Post-DACA']
print(ft_means)

# Calculate simple DiD
did_simple = (ft_means.loc['Treatment (26-30)', 'Post-DACA'] - ft_means.loc['Treatment (26-30)', 'Pre-DACA']) - \
             (ft_means.loc['Control (31-35)', 'Post-DACA'] - ft_means.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate (unweighted): {did_simple:.4f}")

# Weighted means
print("\n--- Mean Full-Time Employment (Weighted by PERWT) ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
ft_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_weighted.columns = ['Pre-DACA', 'Post-DACA']
print(ft_weighted)

did_weighted = (ft_weighted.loc['Treatment (26-30)', 'Post-DACA'] - ft_weighted.loc['Treatment (26-30)', 'Pre-DACA']) - \
               (ft_weighted.loc['Control (31-35)', 'Post-DACA'] - ft_weighted.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate (weighted): {did_weighted:.4f}")

# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("3. DESCRIPTIVE STATISTICS")
print("="*80)

# Demographics by treatment group
print("\n--- Demographics by Treatment Group (Pre-DACA Period) ---")
pre_daca = df[df['AFTER'] == 0]

demo_vars = ['AGE', 'SEX', 'NCHILD', 'FT']
for var in demo_vars:
    if var in df.columns:
        treat_mean = pre_daca[pre_daca['ELIGIBLE'] == 1][var].mean()
        control_mean = pre_daca[pre_daca['ELIGIBLE'] == 0][var].mean()
        print(f"{var}: Treatment={treat_mean:.3f}, Control={control_mean:.3f}")

# Check education distribution
if 'EDUC_RECODE' in df.columns:
    print("\n--- Education Distribution (Pre-DACA Period) ---")
    edu_treat = pre_daca[pre_daca['ELIGIBLE'] == 1]['EDUC_RECODE'].value_counts(normalize=True).sort_index()
    edu_control = pre_daca[pre_daca['ELIGIBLE'] == 0]['EDUC_RECODE'].value_counts(normalize=True).sort_index()
    edu_compare = pd.DataFrame({'Treatment': edu_treat, 'Control': edu_control})
    print(edu_compare)

# =============================================================================
# 4. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("4. MAIN DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)

# Model 1: Basic DiD (no weights)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE * AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model1.params['ELIGIBLE:AFTER']:.4f}")
print(f"Std Error: {model1.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE:AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE:AFTER', 1]:.4f}]")

# Model 2: Basic DiD with weights
print("\n--- Model 2: Basic DiD (Weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE * AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model2.params['ELIGIBLE:AFTER']:.4f}")
print(f"Std Error: {model2.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE:AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE:AFTER', 1]:.4f}]")

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects (Weighted) ---")
df['YEAR_factor'] = df['YEAR'].astype(str)
model3 = smf.wls('FT ~ ELIGIBLE * AFTER + C(YEAR_factor)', data=df, weights=df['PERWT']).fit()
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model3.params['ELIGIBLE:AFTER']:.4f}")
print(f"Std Error: {model3.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE:AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE:AFTER', 1]:.4f}]")

# Model 4: DiD with covariates
print("\n--- Model 4: DiD with Covariates (Weighted) ---")
# Check available covariates
covariates = []

# Sex (1=Male, 2=Female in IPUMS)
if 'SEX' in df.columns:
    df['female'] = (df['SEX'] == 2).astype(int)
    covariates.append('female')

# Marital status
if 'MARST' in df.columns:
    df['married'] = (df['MARST'].isin([1, 2])).astype(int)
    covariates.append('married')

# Number of children
if 'NCHILD' in df.columns:
    covariates.append('NCHILD')

# Education
if 'EDUC_RECODE' in df.columns:
    df['educ_hs'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
    df['educ_somecoll'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
    df['educ_twoyear'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
    df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
    covariates.extend(['educ_hs', 'educ_somecoll', 'educ_twoyear', 'educ_ba'])

# State fixed effects
if 'STATEFIP' in df.columns:
    df['state'] = df['STATEFIP'].astype(str)

print(f"Covariates included: {covariates}")

# Build formula with covariates
formula_covars = 'FT ~ ELIGIBLE * AFTER + ' + ' + '.join(covariates) + ' + C(YEAR_factor)'
model4 = smf.wls(formula_covars, data=df, weights=df['PERWT']).fit()
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model4.params['ELIGIBLE:AFTER']:.4f}")
print(f"Std Error: {model4.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE:AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE:AFTER', 1]:.4f}]")

# Model 5: DiD with covariates and state fixed effects
print("\n--- Model 5: DiD with Covariates and State FE (Weighted) ---")
formula_full = 'FT ~ ELIGIBLE * AFTER + ' + ' + '.join(covariates) + ' + C(YEAR_factor) + C(state)'
model5 = smf.wls(formula_full, data=df, weights=df['PERWT']).fit()
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model5.params['ELIGIBLE:AFTER']:.4f}")
print(f"Std Error: {model5.bse['ELIGIBLE:AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE:AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE:AFTER', 1]:.4f}]")

# =============================================================================
# 5. ROBUST STANDARD ERRORS
# =============================================================================
print("\n" + "="*80)
print("5. ROBUST STANDARD ERRORS")
print("="*80)

# Get the index of the interaction term
interaction_term = 'ELIGIBLE:AFTER'
param_names = list(model5.params.index)
interaction_idx = param_names.index(interaction_term)

# Model with heteroskedasticity-robust standard errors
print("\n--- Model 5 with Robust (HC1) Standard Errors ---")
model5_robust = model5.get_robustcov_results(cov_type='HC1')
print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model5_robust.params[interaction_idx]:.4f}")
print(f"Robust Std Error: {model5_robust.bse[interaction_idx]:.4f}")
print(f"95% CI: [{model5_robust.conf_int()[interaction_idx, 0]:.4f}, {model5_robust.conf_int()[interaction_idx, 1]:.4f}]")

# Clustered standard errors by state
print("\n--- Model 5 with State-Clustered Standard Errors ---")
try:
    model5_cluster = model5.get_robustcov_results(cov_type='cluster', groups=df['STATEFIP'])
    print(f"\nDiD Coefficient (ELIGIBLE:AFTER): {model5_cluster.params[interaction_idx]:.4f}")
    print(f"Clustered Std Error: {model5_cluster.bse[interaction_idx]:.4f}")
    print(f"95% CI: [{model5_cluster.conf_int()[interaction_idx, 0]:.4f}, {model5_cluster.conf_int()[interaction_idx, 1]:.4f}]")
    cluster_se = model5_cluster.bse[interaction_idx]
    cluster_ci = model5_cluster.conf_int()[interaction_idx]
    cluster_pval = model5_cluster.pvalues[interaction_idx]
except Exception as e:
    print(f"Clustering failed: {e}")
    cluster_se = model5_robust.bse[interaction_idx]
    cluster_ci = model5_robust.conf_int()[interaction_idx]
    cluster_pval = model5_robust.pvalues[interaction_idx]

# =============================================================================
# 6. PARALLEL TRENDS CHECK
# =============================================================================
print("\n" + "="*80)
print("6. PARALLEL TRENDS CHECK")
print("="*80)

# Create year-specific treatment effects
df['treat_2008'] = df['ELIGIBLE'] * (df['YEAR'] == 2008).astype(int)
df['treat_2009'] = df['ELIGIBLE'] * (df['YEAR'] == 2009).astype(int)
df['treat_2010'] = df['ELIGIBLE'] * (df['YEAR'] == 2010).astype(int)
# 2011 is reference year (just before treatment)
df['treat_2013'] = df['ELIGIBLE'] * (df['YEAR'] == 2013).astype(int)
df['treat_2014'] = df['ELIGIBLE'] * (df['YEAR'] == 2014).astype(int)
df['treat_2015'] = df['ELIGIBLE'] * (df['YEAR'] == 2015).astype(int)
df['treat_2016'] = df['ELIGIBLE'] * (df['YEAR'] == 2016).astype(int)

event_formula = 'FT ~ ELIGIBLE + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + C(YEAR_factor) + ' + ' + '.join(covariates)
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit()

print("\n--- Event Study Coefficients ---")
event_vars = ['treat_2008', 'treat_2009', 'treat_2010', 'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']
event_results = []
for var in event_vars:
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low, ci_high = model_event.conf_int().loc[var]
    print(f"{var}: {coef:.4f} (SE: {se:.4f}) [{ci_low:.4f}, {ci_high:.4f}]")
    event_results.append({'year': int(var.split('_')[1]), 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# =============================================================================
# 7. HETEROGENEITY ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("7. HETEROGENEITY ANALYSIS")
print("="*80)

# By sex
print("\n--- Effect by Sex ---")
het_sex_results = []
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex_val]
    model_sex = smf.wls('FT ~ ELIGIBLE * AFTER + C(YEAR_factor)', data=df_sex, weights=df_sex['PERWT']).fit()
    coef = model_sex.params['ELIGIBLE:AFTER']
    se = model_sex.bse['ELIGIBLE:AFTER']
    print(f"{sex_name}: DiD = {coef:.4f} (SE: {se:.4f}), N={len(df_sex)}")
    het_sex_results.append({'group': sex_name, 'coef': coef, 'se': se, 'n': len(df_sex)})

# By education level
print("\n--- Effect by Education Level ---")
het_edu_results = []
if 'EDUC_RECODE' in df.columns:
    for edu in ['Less than High School', 'High School Degree', 'Some College', 'BA+']:
        df_edu = df[df['EDUC_RECODE'] == edu]
        if len(df_edu) > 100:
            model_edu = smf.wls('FT ~ ELIGIBLE * AFTER + C(YEAR_factor)', data=df_edu, weights=df_edu['PERWT']).fit()
            coef = model_edu.params['ELIGIBLE:AFTER']
            se = model_edu.bse['ELIGIBLE:AFTER']
            print(f"{edu}: DiD = {coef:.4f} (SE: {se:.4f}), N={len(df_edu)}")
            het_edu_results.append({'group': edu, 'coef': coef, 'se': se, 'n': len(df_edu)})

# =============================================================================
# 8. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "="*80)
print("8. SUMMARY OF RESULTS")
print("="*80)

print("\n--- Summary Table: DiD Estimates ---")
results_list = [
    ('1. Basic (Unweighted)', model1.params['ELIGIBLE:AFTER'], model1.bse['ELIGIBLE:AFTER']),
    ('2. Basic (Weighted)', model2.params['ELIGIBLE:AFTER'], model2.bse['ELIGIBLE:AFTER']),
    ('3. Year FE', model3.params['ELIGIBLE:AFTER'], model3.bse['ELIGIBLE:AFTER']),
    ('4. + Covariates', model4.params['ELIGIBLE:AFTER'], model4.bse['ELIGIBLE:AFTER']),
    ('5. + State FE', model5.params['ELIGIBLE:AFTER'], model5.bse['ELIGIBLE:AFTER']),
    ('5. Robust SE', model5_robust.params[interaction_idx], model5_robust.bse[interaction_idx]),
    ('5. Clustered SE', model5_cluster.params[interaction_idx], cluster_se)
]

results_table = pd.DataFrame(results_list, columns=['Model', 'DiD Coefficient', 'Std Error'])
results_table['t-stat'] = results_table['DiD Coefficient'] / results_table['Std Error']
print(results_table.to_string(index=False))

# Preferred estimate
print("\n" + "="*80)
print("PREFERRED ESTIMATE")
print("="*80)
preferred_coef = model5_cluster.params[interaction_idx]
preferred_se = cluster_se
preferred_ci = cluster_ci
preferred_pval = cluster_pval

print(f"\nModel: DiD with covariates, state and year fixed effects, weighted by PERWT, clustered SE by state")
print(f"Sample Size: {len(df):,}")
print(f"DiD Coefficient: {preferred_coef:.4f}")
print(f"Standard Error: {preferred_se:.4f}")
print(f"95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"p-value: {preferred_pval:.4f}")

# Interpretation
print("\n--- Interpretation ---")
if preferred_pval < 0.05:
    print(f"The DiD estimate is statistically significant at the 5% level.")
else:
    print(f"The DiD estimate is NOT statistically significant at the 5% level.")

print(f"\nThis suggests that DACA eligibility is associated with a {preferred_coef*100:.1f} percentage point")
print(f"{'increase' if preferred_coef > 0 else 'decrease'} in the probability of full-time employment")
print(f"among the treatment group (ages 26-30) relative to the control group (ages 31-35).")

# Save key results to file
results_dict = {
    'model': 'DiD with covariates, state/year FE, weighted, clustered SE by state',
    'sample_size': int(len(df)),
    'did_coefficient': float(preferred_coef),
    'std_error': float(preferred_se),
    'ci_lower': float(preferred_ci[0]),
    'ci_upper': float(preferred_ci[1]),
    'p_value': float(preferred_pval),
    'treatment_n': int(df['ELIGIBLE'].sum()),
    'control_n': int((df['ELIGIBLE'] == 0).sum()),
    'pre_period_n': int((df['AFTER'] == 0).sum()),
    'post_period_n': int((df['AFTER'] == 1).sum())
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Save event study results
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Save heterogeneity results
het_sex_df = pd.DataFrame(het_sex_results)
het_sex_df.to_csv('heterogeneity_sex.csv', index=False)

if het_edu_results:
    het_edu_df = pd.DataFrame(het_edu_results)
    het_edu_df.to_csv('heterogeneity_education.csv', index=False)

# Save model coefficients table
results_table.to_csv('model_comparison.csv', index=False)

print("\n\nAnalysis complete. Results saved to:")
print("  - results_summary.json")
print("  - event_study_results.csv")
print("  - heterogeneity_sex.csv")
print("  - heterogeneity_education.csv")
print("  - model_comparison.csv")

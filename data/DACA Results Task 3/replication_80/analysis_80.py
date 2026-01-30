"""
DACA Replication Analysis - Study 80
Independent replication of DACA effect on full-time employment

Research Question: What was the causal impact of DACA eligibility on
full-time employment among ethnically Hispanic-Mexican, Mexican-born
people living in the United States?

Treatment: Ages 26-30 at June 15, 2012 (ELIGIBLE=1)
Control: Ages 31-35 at June 15, 2012 (ELIGIBLE=0)
Pre-period: 2008-2011
Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Load and Explore Data
# =============================================================================

print("="*80)
print("DACA REPLICATION ANALYSIS - STUDY 80")
print("="*80)
print()

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_80\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print("STEP 1: DATA EXPLORATION")
print("-"*40)
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Variables: {len(df.columns)}")
print()

# Key variables summary
print("Key Variables Summary:")
print(f"  ELIGIBLE distribution: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"  AFTER distribution: {df['AFTER'].value_counts().to_dict()}")
print(f"  FT distribution: {df['FT'].value_counts().to_dict()}")
print()

# Check for missing values in key variables
key_vars = ['YEAR', 'ELIGIBLE', 'AFTER', 'FT', 'PERWT', 'SEX', 'AGE', 'EDUC_RECODE', 'MARST']
print("Missing values in key variables:")
for var in key_vars:
    if var in df.columns:
        missing = df[var].isna().sum()
        print(f"  {var}: {missing} ({100*missing/len(df):.2f}%)")
print()

# =============================================================================
# STEP 2: Descriptive Statistics
# =============================================================================

print("STEP 2: DESCRIPTIVE STATISTICS")
print("-"*40)

# Sample sizes by group and period
print("\nSample Sizes by Group and Period:")
cross_tab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
cross_tab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
cross_tab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(cross_tab)
print()

# Sample sizes by year
print("\nObservations by Year:")
year_counts = df.groupby('YEAR').size()
print(year_counts.to_string())
print()

# Mean FT by group and period (unweighted)
print("\nMean Full-Time Employment (Unweighted):")
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_means.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
ft_means.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_means.round(4))
print()

# Calculate simple DiD
print("\nSimple Difference-in-Differences (Unweighted):")
ft_control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
ft_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

diff_control = ft_control_post - ft_control_pre
diff_treat = ft_treat_post - ft_treat_pre
did_simple = diff_treat - diff_control

print(f"  Control group change: {diff_control:.4f}")
print(f"  Treatment group change: {diff_treat:.4f}")
print(f"  DiD estimate: {did_simple:.4f}")
print()

# Weighted means
print("\nMean Full-Time Employment (Weighted by PERWT):")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_means_w = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean, include_groups=False).unstack()
ft_means_w.columns = ['Pre (2008-2011)', 'Post (2013-2016)']
ft_means_w.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_means_w.round(4))
print()

# Weighted DiD
ft_control_pre_w = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                               weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
ft_control_post_w = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                                weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])
ft_treat_pre_w = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                             weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
ft_treat_post_w = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                              weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])

diff_control_w = ft_control_post_w - ft_control_pre_w
diff_treat_w = ft_treat_post_w - ft_treat_pre_w
did_weighted = diff_treat_w - diff_control_w

print("\nSimple Difference-in-Differences (Weighted):")
print(f"  Control group change: {diff_control_w:.4f}")
print(f"  Treatment group change: {diff_treat_w:.4f}")
print(f"  DiD estimate: {did_weighted:.4f}")
print()

# =============================================================================
# STEP 3: Demographic Characteristics
# =============================================================================

print("STEP 3: DEMOGRAPHIC CHARACTERISTICS BY GROUP")
print("-"*40)

# Sex distribution
print("\nSex Distribution (1=Male, 2=Female):")
sex_dist = df.groupby('ELIGIBLE')['SEX'].value_counts(normalize=True).unstack()
sex_dist.index = ['Control (31-35)', 'Treatment (26-30)']
print(sex_dist.round(3))

# Education distribution
print("\nEducation Distribution:")
educ_dist = df.groupby('ELIGIBLE')['EDUC_RECODE'].value_counts(normalize=True).unstack()
educ_dist.index = ['Control (31-35)', 'Treatment (26-30)']
print(educ_dist.round(3))

# Marital status
print("\nMarital Status Distribution (1=married spouse present, 6=never married):")
marst_dist = df.groupby('ELIGIBLE')['MARST'].value_counts(normalize=True).unstack()
marst_dist.index = ['Control (31-35)', 'Treatment (26-30)']
print(marst_dist.round(3))

# Age distribution
print("\nAge Summary (at survey time):")
age_stats = df.groupby('ELIGIBLE')['AGE'].describe()
age_stats.index = ['Control (31-35)', 'Treatment (26-30)']
print(age_stats.round(2))

print()

# =============================================================================
# STEP 4: MAIN REGRESSION ANALYSIS
# =============================================================================

print("STEP 4: MAIN REGRESSION ANALYSIS")
print("-"*40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted OLS)
print("\n--- Model 1: Basic DiD (Unweighted OLS) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"N: {int(model1.nobs):,}")

# Model 2: Basic DiD (weighted)
print("\n--- Model 2: Basic DiD (Weighted by PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: DiD with demographic controls (weighted)
print("\n--- Model 3: DiD with Demographic Controls (Weighted) ---")
# Create dummy variables for categorical controls
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Education dummies (reference: Less than High School)
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True)
df = pd.concat([df, educ_dummies], axis=1)

model3_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE'
# Add education dummies
for col in educ_dummies.columns:
    model3_formula += f' + Q("{col}")'

model3 = smf.wls(model3_formula, data=df, weights=df['PERWT']).fit()
print("Coefficients for DiD terms:")
print(f"  ELIGIBLE: {model3.params['ELIGIBLE']:.4f} (SE: {model3.bse['ELIGIBLE']:.4f})")
print(f"  AFTER: {model3.params['AFTER']:.4f} (SE: {model3.bse['AFTER']:.4f})")
print(f"  ELIGIBLE_AFTER: {model3.params['ELIGIBLE_AFTER']:.4f} (SE: {model3.bse['ELIGIBLE_AFTER']:.4f})")
print(f"\nDiD Coefficient: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state fixed effects (weighted)
print("\n--- Model 4: DiD with State Fixed Effects (Weighted) ---")
# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

model4_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE'
for col in educ_dummies.columns:
    model4_formula += f' + Q("{col}")'
for col in state_dummies.columns:
    model4_formula += f' + Q("{col}")'

model4 = smf.wls(model4_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: DiD with robust/clustered standard errors at state level
print("\n--- Model 5: DiD with Clustered Standard Errors (State) ---")
model5 = smf.wls(model3_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                    cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

print()

# =============================================================================
# STEP 5: PARALLEL TRENDS ANALYSIS
# =============================================================================

print("STEP 5: PARALLEL TRENDS ANALYSIS")
print("-"*40)

# Year-by-year means
print("\nFull-Time Employment by Year and Group (Weighted):")
yearly_means = []
for year in sorted(df['YEAR'].unique()):
    for elig in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==elig)]
        if len(subset) > 0:
            w_mean = np.average(subset['FT'], weights=subset['PERWT'])
            n = len(subset)
            yearly_means.append({'Year': year, 'ELIGIBLE': elig, 'FT_mean': w_mean, 'N': n})

yearly_df = pd.DataFrame(yearly_means)
print(yearly_df.pivot(index='Year', columns='ELIGIBLE', values='FT_mean').round(4))

# Event study regression
print("\n--- Event Study Regression ---")
# Create year dummies (reference: 2011, last pre-treatment year)
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
# 2011 is reference
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Create interactions
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_X_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

event_formula = ('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 '
                 '+ ELIG_X_2008 + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2013 + ELIG_X_2014 + ELIG_X_2015 + ELIG_X_2016 '
                 '+ FEMALE + MARRIED + AGE')

event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                        cov_kwds={'groups': df['STATEFIP']})

print("\nEvent Study Coefficients (Eligible x Year interactions):")
print("Reference year: 2011")
event_vars = ['ELIG_X_2008', 'ELIG_X_2009', 'ELIG_X_2010', 'ELIG_X_2013', 'ELIG_X_2014', 'ELIG_X_2015', 'ELIG_X_2016']
for var in event_vars:
    year = var.split('_')[-1]
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci_lo, ci_hi = event_model.conf_int().loc[var]
    pre_post = "PRE" if int(year) < 2012 else "POST"
    sig = "*" if pval < 0.05 else ("+" if pval < 0.1 else "")
    print(f"  {year} ({pre_post}): {coef:7.4f} (SE: {se:.4f}) [{ci_lo:.4f}, {ci_hi:.4f}] p={pval:.3f} {sig}")

print()

# =============================================================================
# STEP 6: ROBUSTNESS CHECKS
# =============================================================================

print("STEP 6: ROBUSTNESS CHECKS")
print("-"*40)

# Robustness 1: By sex
print("\n--- Robustness 1: Heterogeneity by Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex_val]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MARRIED + AGE',
                        data=subset, weights=subset['PERWT']).fit()
    print(f"{sex_name}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_sex.nobs):,}")

# Robustness 2: By education level
print("\n--- Robustness 2: Heterogeneity by Education ---")
for educ in ['Less than High School', 'High School Degree', 'Some College', 'BA+']:
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE',
                            data=subset, weights=subset['PERWT']).fit()
        print(f"{educ}: DiD = {model_educ.params['ELIGIBLE_AFTER']:.4f} (SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_educ.nobs):,}")

# Robustness 3: Linear probability model vs interpretation note
print("\n--- Note on Model Specification ---")
print("All models use Linear Probability Model (OLS/WLS) for interpretability.")
print("Coefficients represent changes in probability of full-time employment.")

# Robustness 4: Excluding 2008 (potential pre-trend test)
print("\n--- Robustness 3: Excluding 2008 ---")
df_no2008 = df[df['YEAR'] != 2008]
model_no2008 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE',
                       data=df_no2008, weights=df_no2008['PERWT']).fit()
print(f"DiD (excl. 2008): {model_no2008.params['ELIGIBLE_AFTER']:.4f} (SE: {model_no2008.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_no2008.nobs):,}")

# Robustness 5: Including state policy controls
print("\n--- Robustness 4: With State Policy Controls ---")
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY']
policy_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE'
for pv in policy_vars:
    policy_formula += f' + {pv}'

model_policy = smf.wls(policy_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD (with policy controls): {model_policy.params['ELIGIBLE_AFTER']:.4f} (SE: {model_policy.bse['ELIGIBLE_AFTER']:.4f})")

print()

# =============================================================================
# STEP 7: SUMMARY OF RESULTS
# =============================================================================

print("="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)
print()

print("PREFERRED SPECIFICATION: Model 5 (DiD with controls and clustered SEs)")
print("-"*60)
print(f"Effect of DACA eligibility on full-time employment:")
print(f"  DiD Coefficient: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Clustered SE (state): {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample Size: {int(model5.nobs):,}")
print()

print("Interpretation:")
if model5.pvalues['ELIGIBLE_AFTER'] < 0.05:
    direction = "increased" if model5.params['ELIGIBLE_AFTER'] > 0 else "decreased"
    print(f"  DACA eligibility {direction} the probability of full-time employment")
    print(f"  by {abs(model5.params['ELIGIBLE_AFTER'])*100:.2f} percentage points (p < 0.05).")
elif model5.pvalues['ELIGIBLE_AFTER'] < 0.10:
    direction = "increased" if model5.params['ELIGIBLE_AFTER'] > 0 else "decreased"
    print(f"  DACA eligibility {direction} the probability of full-time employment")
    print(f"  by {abs(model5.params['ELIGIBLE_AFTER'])*100:.2f} percentage points (marginally significant, p < 0.10).")
else:
    print("  No statistically significant effect of DACA eligibility on full-time employment")
    print(f"  was detected (point estimate: {model5.params['ELIGIBLE_AFTER']*100:.2f} pp, p = {model5.pvalues['ELIGIBLE_AFTER']:.3f}).")

print()
print("Parallel Trends Assessment:")
pre_coefs = [event_model.params[f'ELIG_X_{y}'] for y in [2008, 2009, 2010]]
pre_pvals = [event_model.pvalues[f'ELIG_X_{y}'] for y in [2008, 2009, 2010]]
if all(p > 0.10 for p in pre_pvals):
    print("  Pre-treatment coefficients are not statistically different from zero,")
    print("  supporting the parallel trends assumption.")
else:
    print("  WARNING: Some pre-treatment coefficients are statistically significant,")
    print("  which may indicate a violation of the parallel trends assumption.")

print()
print("="*80)

# Save key results to file
results_summary = {
    'did_coefficient': model5.params['ELIGIBLE_AFTER'],
    'standard_error': model5.bse['ELIGIBLE_AFTER'],
    'ci_lower': model5.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_upper': model5.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model5.pvalues['ELIGIBLE_AFTER'],
    'sample_size': int(model5.nobs)
}

print("\nResults saved for report generation.")
print(f"Key estimate: {results_summary['did_coefficient']:.4f} (SE: {results_summary['standard_error']:.4f})")

"""
DACA Replication Study - Analysis Script
=========================================
This script performs a difference-in-differences analysis to estimate the causal
effect of DACA eligibility on full-time employment among Hispanic-Mexican
Mexican-born individuals in the United States.

Treatment: DACA eligibility (ages 26-30 on June 15, 2012)
Control: Ages 31-35 on June 15, 2012 (would have been eligible but for age)
Outcome: Full-time employment (≥35 hours/week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)
print()

# =============================================================================
# 1. DATA LOADING AND INSPECTION
# =============================================================================
print("1. LOADING DATA")
print("-"*40)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Dataset shape: {df.shape}")
print(f"Number of observations: {len(df)}")
print(f"Number of variables: {df.shape[1]}")
print()

# Check key variables
print("Key variables present:")
key_vars = ['FT', 'ELIGIBLE', 'AFTER', 'YEAR', 'PERWT', 'AGE', 'SEX', 'EDUC_RECODE']
for var in key_vars:
    if var in df.columns:
        print(f"  [OK] {var}")
    else:
        print(f"  [MISSING] {var}")
print()

# =============================================================================
# 2. DATA VALIDATION
# =============================================================================
print("2. DATA VALIDATION")
print("-"*40)

# Check YEAR distribution
print("Year distribution:")
print(df['YEAR'].value_counts().sort_index())
print()

# Check ELIGIBLE distribution
print("ELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}")
print(f"Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}")
print()

# Check AFTER distribution
print("AFTER distribution:")
print(df['AFTER'].value_counts())
print(f"Pre-period (AFTER=0): {(df['AFTER']==0).sum()}")
print(f"Post-period (AFTER=1): {(df['AFTER']==1).sum()}")
print()

# Check FT distribution
print("FT (Full-time) distribution:")
print(df['FT'].value_counts())
print(f"Not full-time (FT=0): {(df['FT']==0).sum()}")
print(f"Full-time (FT=1): {(df['FT']==1).sum()}")
print()

# Check for missing values in key variables
print("Missing values in key variables:")
for var in ['FT', 'ELIGIBLE', 'AFTER', 'PERWT']:
    missing = df[var].isnull().sum()
    print(f"  {var}: {missing}")
print()

# =============================================================================
# 3. SUMMARY STATISTICS
# =============================================================================
print("3. SUMMARY STATISTICS")
print("-"*40)

# Create 2x2 table of means
print("Full-time employment rates by group and period:")
print()

# Calculate means by group
means_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
means_table_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
)
print("Unweighted means:")
print(means_table)
print()

print("Weighted means (using PERWT):")
print(means_table_weighted.unstack())
print()

# Calculate simple DiD
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_ctrl = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_ctrl = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"Treatment group (ELIGIBLE=1):")
print(f"  Pre-period mean: {pre_treat:.4f}")
print(f"  Post-period mean: {post_treat:.4f}")
print(f"  Change: {post_treat - pre_treat:.4f}")
print()
print(f"Control group (ELIGIBLE=0):")
print(f"  Pre-period mean: {pre_ctrl:.4f}")
print(f"  Post-period mean: {post_ctrl:.4f}")
print(f"  Change: {post_ctrl - pre_ctrl:.4f}")
print()
print(f"Simple DiD estimate: {(post_treat - pre_treat) - (post_ctrl - pre_ctrl):.4f}")
print()

# Weighted means
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

pre_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
post_treat_w = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')
pre_ctrl_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')
post_ctrl_w = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')

print("Weighted estimates:")
print(f"Treatment group (ELIGIBLE=1):")
print(f"  Pre-period mean: {pre_treat_w:.4f}")
print(f"  Post-period mean: {post_treat_w:.4f}")
print(f"  Change: {post_treat_w - pre_treat_w:.4f}")
print()
print(f"Control group (ELIGIBLE=0):")
print(f"  Pre-period mean: {pre_ctrl_w:.4f}")
print(f"  Post-period mean: {post_ctrl_w:.4f}")
print(f"  Change: {post_ctrl_w - pre_ctrl_w:.4f}")
print()
print(f"Weighted DiD estimate: {(post_treat_w - pre_treat_w) - (post_ctrl_w - pre_ctrl_w):.4f}")
print()

# =============================================================================
# 4. DEMOGRAPHIC SUMMARY
# =============================================================================
print("4. DEMOGRAPHIC SUMMARY")
print("-"*40)

# Sex distribution (1=Male, 2=Female in IPUMS)
print("Sex distribution:")
sex_dist = df.groupby('ELIGIBLE')['SEX'].value_counts(normalize=True).unstack()
print(sex_dist)
print()

# Age distribution
print("Age distribution by group:")
age_stats = df.groupby('ELIGIBLE')['AGE'].describe()
print(age_stats)
print()

# Education distribution
print("Education distribution (EDUC_RECODE):")
if 'EDUC_RECODE' in df.columns:
    educ_dist = df.groupby('ELIGIBLE')['EDUC_RECODE'].value_counts(normalize=True).unstack()
    print(educ_dist)
print()

# Marital status (1=Married spouse present, 6=Never married)
print("Marital status distribution:")
marst_dist = df.groupby('ELIGIBLE')['MARST'].value_counts(normalize=True).unstack()
print(marst_dist)
print()

# =============================================================================
# 5. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("="*80)
print("5. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("="*80)
print()

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("MODEL 1: Basic DiD (Unweighted OLS)")
print("-"*40)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())
print()

# Model 2: Basic DiD (weighted)
print("MODEL 2: Basic DiD (Weighted OLS)")
print("-"*40)
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())
print()

# Model 3: DiD with robust standard errors
print("MODEL 3: DiD with Robust (HC3) Standard Errors")
print("-"*40)
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC3')
print(model3.summary())
print()

# =============================================================================
# 6. MODELS WITH COVARIATES
# =============================================================================
print("="*80)
print("6. MODELS WITH COVARIATES")
print("="*80)
print()

# Prepare categorical variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Model 4: DiD with demographic controls
print("MODEL 4: DiD with Demographic Controls")
print("-"*40)
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE)'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit()
print(model4.summary())
print()

# Model 5: DiD with year fixed effects
print("MODEL 5: DiD with Year Fixed Effects")
print("-"*40)
formula5 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit()
print(model5.summary())
print()

# Model 6: DiD with state fixed effects
print("MODEL 6: DiD with State Fixed Effects")
print("-"*40)
formula6 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit()
# Print only key coefficients
print("Key coefficients:")
print(f"  ELIGIBLE: {model6.params['ELIGIBLE']:.4f} (SE: {model6.bse['ELIGIBLE']:.4f})")
print(f"  AFTER: {model6.params['AFTER']:.4f} (SE: {model6.bse['AFTER']:.4f})")
print(f"  ELIGIBLE_AFTER (DiD): {model6.params['ELIGIBLE_AFTER']:.4f} (SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  R-squared: {model6.rsquared:.4f}")
print(f"  N: {int(model6.nobs)}")
print()

# Model 7: Full model with demographics, year FE, and state FE
print("MODEL 7: Full Model (Demographics + Year FE + State FE)")
print("-"*40)
formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + C(EDUC_RECODE) + C(YEAR) + C(STATEFIP)'
model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit()
print("Key coefficients:")
print(f"  ELIGIBLE: {model7.params['ELIGIBLE']:.4f} (SE: {model7.bse['ELIGIBLE']:.4f})")
print(f"  ELIGIBLE_AFTER (DiD): {model7.params['ELIGIBLE_AFTER']:.4f} (SE: {model7.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  FEMALE: {model7.params['FEMALE']:.4f} (SE: {model7.bse['FEMALE']:.4f})")
print(f"  MARRIED: {model7.params['MARRIED']:.4f} (SE: {model7.bse['MARRIED']:.4f})")
print(f"  R-squared: {model7.rsquared:.4f}")
print(f"  N: {int(model7.nobs)}")
print()

# =============================================================================
# 7. ROBUSTNESS: CLUSTERED STANDARD ERRORS
# =============================================================================
print("="*80)
print("7. ROBUSTNESS: CLUSTERED STANDARD ERRORS")
print("="*80)
print()

# Model 8: Basic DiD with state-clustered SEs
print("MODEL 8: DiD with State-Clustered Standard Errors")
print("-"*40)
model8 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print(model8.summary())
print()

# =============================================================================
# 8. PRE-TRENDS ANALYSIS
# =============================================================================
print("="*80)
print("8. PRE-TRENDS ANALYSIS")
print("="*80)
print()

# Calculate FT rates by year and eligibility
print("Full-time employment rates by year and eligibility status:")
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_mean': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x)
    })
).unstack()
print(yearly_means)
print()

# Test for pre-trends in pre-period only
print("Testing pre-trends (2008-2011 only):")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_YEAR_TREND'] = df_pre['ELIGIBLE'] * df_pre['YEAR_TREND']

model_pretrend = smf.wls('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_YEAR_TREND',
                          data=df_pre, weights=df_pre['PERWT']).fit()
print(model_pretrend.summary())
print()
print(f"Pre-trend test (ELIGIBLE x YEAR_TREND): coef = {model_pretrend.params['ELIGIBLE_YEAR_TREND']:.4f}, "
      f"p-value = {model_pretrend.pvalues['ELIGIBLE_YEAR_TREND']:.4f}")
print()

# =============================================================================
# 9. HETEROGENEITY ANALYSIS
# =============================================================================
print("="*80)
print("9. HETEROGENEITY ANALYSIS")
print("="*80)
print()

# By sex
print("Heterogeneity by Sex:")
print("-"*40)

# Males
df_male = df[df['SEX'] == 1].copy()
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                      data=df_male, weights=df_male['PERWT']).fit()
print(f"Males (N={len(df_male)}):")
print(f"  DiD estimate: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, "
      f"{model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print()

# Females
df_female = df[df['SEX'] == 2].copy()
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                        data=df_female, weights=df_female['PERWT']).fit()
print(f"Females (N={len(df_female)}):")
print(f"  DiD estimate: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, "
      f"{model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print()

# By education
print("Heterogeneity by Education:")
print("-"*40)
if 'EDUC_RECODE' in df.columns:
    # Filter out NaN values and get unique education levels
    educ_levels = df['EDUC_RECODE'].dropna().unique()
    for educ_level in educ_levels:
        df_educ = df[df['EDUC_RECODE'] == educ_level].copy()
        if len(df_educ) > 100:  # Only if sufficient sample
            model_educ = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                                  data=df_educ, weights=df_educ['PERWT']).fit()
            print(f"Education level {educ_level} (N={len(df_educ)}):")
            print(f"  DiD estimate: {model_educ.params['ELIGIBLE_AFTER']:.4f}")
            print(f"  SE: {model_educ.bse['ELIGIBLE_AFTER']:.4f}")
            print()

# =============================================================================
# 10. STATE POLICY INTERACTIONS
# =============================================================================
print("="*80)
print("10. STATE POLICY INTERACTIONS")
print("="*80)
print()

# Check for state policy variables
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID',
               'EVERIFY', 'SECURECOMMUNITIES']
available_policies = [v for v in policy_vars if v in df.columns]
print(f"Available state policy variables: {available_policies}")
print()

# Model with driver's license policy interaction
if 'DRIVERSLICENSES' in df.columns:
    print("Driver's License Policy Interaction:")
    df['ELIGIBLE_AFTER_DL'] = df['ELIGIBLE_AFTER'] * df['DRIVERSLICENSES']
    formula_dl = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + DRIVERSLICENSES + ELIGIBLE_AFTER_DL'
    model_dl = smf.wls(formula_dl, data=df, weights=df['PERWT']).fit()
    print(f"  ELIGIBLE_AFTER: {model_dl.params['ELIGIBLE_AFTER']:.4f} (SE: {model_dl.bse['ELIGIBLE_AFTER']:.4f})")
    print(f"  ELIGIBLE_AFTER x DRIVERSLICENSES: {model_dl.params['ELIGIBLE_AFTER_DL']:.4f} (SE: {model_dl.bse['ELIGIBLE_AFTER_DL']:.4f})")
    print()

# =============================================================================
# 11. EVENT STUDY SPECIFICATION
# =============================================================================
print("="*80)
print("11. EVENT STUDY SPECIFICATION")
print("="*80)
print()

# Create year-by-eligibility interactions
# Using 2011 as reference year (last pre-treatment year)
years = sorted(df['YEAR'].unique())
ref_year = 2011

print(f"Reference year: {ref_year}")
print()

# Create year dummies and interactions
for year in years:
    if year != ref_year:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Build formula
year_terms = ' + '.join([f'YEAR_{y}' for y in years if y != ref_year])
interact_terms = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in years if y != ref_year])
formula_es = f'FT ~ ELIGIBLE + {year_terms} + {interact_terms}'

model_es = smf.wls(formula_es, data=df, weights=df['PERWT']).fit()

print("Event Study Coefficients (ELIGIBLE x YEAR interactions):")
print("(Reference year: 2011)")
print()
for year in years:
    if year != ref_year:
        coef = model_es.params[f'ELIGIBLE_YEAR_{year}']
        se = model_es.bse[f'ELIGIBLE_YEAR_{year}']
        ci_lo, ci_hi = model_es.conf_int().loc[f'ELIGIBLE_YEAR_{year}']
        sig = "*" if model_es.pvalues[f'ELIGIBLE_YEAR_{year}'] < 0.05 else ""
        print(f"  {year}: {coef:7.4f} (SE: {se:.4f}) [{ci_lo:.4f}, {ci_hi:.4f}] {sig}")
print()

# =============================================================================
# 12. SUMMARY TABLE OF MAIN RESULTS
# =============================================================================
print("="*80)
print("12. SUMMARY TABLE OF MAIN RESULTS")
print("="*80)
print()

# Collect results
results = []

# Model 1: Basic OLS
results.append({
    'Model': '(1) Basic OLS',
    'DiD_Estimate': model1.params['ELIGIBLE_AFTER'],
    'SE': model1.bse['ELIGIBLE_AFTER'],
    'CI_Low': model1.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model1.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model1.pvalues['ELIGIBLE_AFTER'],
    'N': int(model1.nobs),
    'R2': model1.rsquared
})

# Model 2: Weighted OLS
results.append({
    'Model': '(2) Weighted OLS',
    'DiD_Estimate': model2.params['ELIGIBLE_AFTER'],
    'SE': model2.bse['ELIGIBLE_AFTER'],
    'CI_Low': model2.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model2.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model2.pvalues['ELIGIBLE_AFTER'],
    'N': int(model2.nobs),
    'R2': model2.rsquared
})

# Model 3: Robust SE
results.append({
    'Model': '(3) Robust SE',
    'DiD_Estimate': model3.params['ELIGIBLE_AFTER'],
    'SE': model3.bse['ELIGIBLE_AFTER'],
    'CI_Low': model3.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model3.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model3.pvalues['ELIGIBLE_AFTER'],
    'N': int(model3.nobs),
    'R2': model3.rsquared
})

# Model 4: With demographics
results.append({
    'Model': '(4) + Demographics',
    'DiD_Estimate': model4.params['ELIGIBLE_AFTER'],
    'SE': model4.bse['ELIGIBLE_AFTER'],
    'CI_Low': model4.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model4.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model4.pvalues['ELIGIBLE_AFTER'],
    'N': int(model4.nobs),
    'R2': model4.rsquared
})

# Model 5: Year FE
results.append({
    'Model': '(5) + Year FE',
    'DiD_Estimate': model5.params['ELIGIBLE_AFTER'],
    'SE': model5.bse['ELIGIBLE_AFTER'],
    'CI_Low': model5.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model5.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model5.pvalues['ELIGIBLE_AFTER'],
    'N': int(model5.nobs),
    'R2': model5.rsquared
})

# Model 6: State FE
results.append({
    'Model': '(6) + State FE',
    'DiD_Estimate': model6.params['ELIGIBLE_AFTER'],
    'SE': model6.bse['ELIGIBLE_AFTER'],
    'CI_Low': model6.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model6.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model6.pvalues['ELIGIBLE_AFTER'],
    'N': int(model6.nobs),
    'R2': model6.rsquared
})

# Model 7: Full model
results.append({
    'Model': '(7) Full Model',
    'DiD_Estimate': model7.params['ELIGIBLE_AFTER'],
    'SE': model7.bse['ELIGIBLE_AFTER'],
    'CI_Low': model7.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model7.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model7.pvalues['ELIGIBLE_AFTER'],
    'N': int(model7.nobs),
    'R2': model7.rsquared
})

# Model 8: Clustered SE
results.append({
    'Model': '(8) Clustered SE',
    'DiD_Estimate': model8.params['ELIGIBLE_AFTER'],
    'SE': model8.bse['ELIGIBLE_AFTER'],
    'CI_Low': model8.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model8.conf_int().loc['ELIGIBLE_AFTER', 1],
    'p_value': model8.pvalues['ELIGIBLE_AFTER'],
    'N': int(model8.nobs),
    'R2': model8.rsquared
})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# =============================================================================
# 13. PREFERRED ESTIMATE
# =============================================================================
print("="*80)
print("13. PREFERRED ESTIMATE")
print("="*80)
print()

# Using Model 2 (Weighted OLS) as preferred specification
# This accounts for survey weights while keeping the model simple and transparent
preferred_model = model2
preferred_name = "Weighted OLS (Model 2)"

print(f"Preferred specification: {preferred_name}")
print()
print(f"Effect size (DiD estimate): {preferred_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {preferred_model.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence interval: [{preferred_model.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, "
      f"{preferred_model.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {preferred_model.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {int(preferred_model.nobs)}")
print(f"R-squared: {preferred_model.rsquared:.4f}")
print()

# Interpretation
effect = preferred_model.params['ELIGIBLE_AFTER']
if effect > 0:
    direction = "increase"
else:
    direction = "decrease"
    effect = abs(effect)

print(f"Interpretation: DACA eligibility is associated with a {effect:.1f} percentage point "
      f"{direction} in the probability of full-time employment.")

if preferred_model.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")
print()

# =============================================================================
# 14. SAVE RESULTS FOR LATEX
# =============================================================================
print("="*80)
print("14. SAVING RESULTS")
print("="*80)
print()

# Save main results
results_df.to_csv('results_summary.csv', index=False)
print("Saved: results_summary.csv")

# Save yearly means for plotting
yearly_means_export = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_weighted_mean': np.average(x['FT'], weights=x['PERWT']),
        'FT_unweighted_mean': x['FT'].mean(),
        'N': len(x)
    })
).reset_index()
yearly_means_export.to_csv('yearly_means.csv', index=False)
print("Saved: yearly_means.csv")

# Save event study coefficients
es_coefs = []
for year in years:
    if year != ref_year:
        es_coefs.append({
            'Year': year,
            'Coefficient': model_es.params[f'ELIGIBLE_YEAR_{year}'],
            'SE': model_es.bse[f'ELIGIBLE_YEAR_{year}'],
            'CI_Low': model_es.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 0],
            'CI_High': model_es.conf_int().loc[f'ELIGIBLE_YEAR_{year}', 1],
            'p_value': model_es.pvalues[f'ELIGIBLE_YEAR_{year}']
        })
    else:
        es_coefs.append({
            'Year': year,
            'Coefficient': 0,
            'SE': 0,
            'CI_Low': 0,
            'CI_High': 0,
            'p_value': np.nan
        })

es_df = pd.DataFrame(es_coefs)
es_df.to_csv('event_study_coefficients.csv', index=False)
print("Saved: event_study_coefficients.csv")

# Save demographic summary
demo_summary = df.groupby('ELIGIBLE').agg({
    'FT': ['mean', 'std'],
    'AGE': ['mean', 'std'],
    'FEMALE': 'mean',
    'MARRIED': 'mean',
    'PERWT': 'sum'
}).round(4)
demo_summary.to_csv('demographic_summary.csv')
print("Saved: demographic_summary.csv")

# Save heterogeneity results
hetero_results = []
hetero_results.append({
    'Subgroup': 'Males',
    'DiD_Estimate': model_male.params['ELIGIBLE_AFTER'],
    'SE': model_male.bse['ELIGIBLE_AFTER'],
    'N': len(df_male)
})
hetero_results.append({
    'Subgroup': 'Females',
    'DiD_Estimate': model_female.params['ELIGIBLE_AFTER'],
    'SE': model_female.bse['ELIGIBLE_AFTER'],
    'N': len(df_female)
})
pd.DataFrame(hetero_results).to_csv('heterogeneity_results.csv', index=False)
print("Saved: heterogeneity_results.csv")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)

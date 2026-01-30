"""
DACA Replication Analysis Script
================================
This script analyzes the effect of DACA eligibility on full-time employment
using a difference-in-differences design.

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living
in the United States, what was the causal impact of eligibility for DACA on
the probability that the eligible person is employed full-time?

Treatment group: Ages 26-30 at the time of DACA implementation (June 2012)
Control group: Ages 31-35 at the time of DACA implementation
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
print("DACA REPLICATION ANALYSIS")
print("Effect of DACA Eligibility on Full-Time Employment")
print("=" * 80)

# Load data
print("\n1. LOADING DATA")
print("-" * 40)
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df):,}")
print(f"Total variables: {df.shape[1]}")
print(f"Years covered: {sorted(df['YEAR'].unique())}")

# Verify key variables
print("\n2. DATA VERIFICATION")
print("-" * 40)
print(f"\nYear distribution:")
print(df['YEAR'].value_counts().sort_index())

print(f"\nELIGIBLE (treatment indicator) distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"  ELIGIBLE=1 (ages 26-30 in June 2012): {(df['ELIGIBLE']==1).sum():,}")
print(f"  ELIGIBLE=0 (ages 31-35 in June 2012): {(df['ELIGIBLE']==0).sum():,}")

print(f"\nAFTER (post-DACA indicator) distribution:")
print(df['AFTER'].value_counts())
print(f"  AFTER=0 (years 2008-2011): {(df['AFTER']==0).sum():,}")
print(f"  AFTER=1 (years 2013-2016): {(df['AFTER']==1).sum():,}")

print(f"\nFT (full-time employment outcome) distribution:")
print(df['FT'].value_counts())
print(f"  FT=0 (not full-time): {(df['FT']==0).sum():,}")
print(f"  FT=1 (full-time, 35+ hrs/week): {(df['FT']==1).sum():,}")
print(f"  Overall FT rate: {df['FT'].mean():.3f}")

# 3. DESCRIPTIVE STATISTICS
print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Group means by treatment status and period
print("\nMean Full-Time Employment Rate by Group and Period:")
group_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
group_means.columns = ['Mean FT Rate', 'N', 'Std Dev']
print(group_means)

# Weighted means
print("\nWeighted Mean Full-Time Employment Rate by Group and Period:")
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        n = len(subset)
        label_elig = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
        label_after = "Post (2013-2016)" if after == 1 else "Pre (2008-2011)"
        print(f"  {label_elig}, {label_after}: {weighted_mean:.4f} (N={n:,})")

# 4. SIMPLE DIFFERENCE-IN-DIFFERENCES (UNWEIGHTED)
print("\n4. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Calculate group means
mean_treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
mean_treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
mean_ctrl_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
mean_ctrl_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\nUnweighted Means:")
print(f"                          Pre-DACA      Post-DACA     Difference")
print(f"Treatment (26-30):        {mean_treat_pre:.4f}        {mean_treat_post:.4f}        {mean_treat_post-mean_treat_pre:+.4f}")
print(f"Control (31-35):          {mean_ctrl_pre:.4f}        {mean_ctrl_post:.4f}        {mean_ctrl_post-mean_ctrl_pre:+.4f}")
print(f"Difference:               {mean_treat_pre-mean_ctrl_pre:+.4f}        {mean_treat_post-mean_ctrl_post:+.4f}        ")
print(f"\nDifference-in-Differences Estimate: {(mean_treat_post-mean_treat_pre)-(mean_ctrl_post-mean_ctrl_pre):+.4f}")

# Weighted DID
def weighted_mean(data, weight_col='PERWT'):
    return np.average(data['FT'], weights=data[weight_col])

wmean_treat_pre = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)])
wmean_treat_post = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)])
wmean_ctrl_pre = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)])
wmean_ctrl_post = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)])

print(f"\nWeighted Means:")
print(f"                          Pre-DACA      Post-DACA     Difference")
print(f"Treatment (26-30):        {wmean_treat_pre:.4f}        {wmean_treat_post:.4f}        {wmean_treat_post-wmean_treat_pre:+.4f}")
print(f"Control (31-35):          {wmean_ctrl_pre:.4f}        {wmean_ctrl_post:.4f}        {wmean_ctrl_post-wmean_ctrl_pre:+.4f}")
print(f"Difference:               {wmean_treat_pre-wmean_ctrl_pre:+.4f}        {wmean_treat_post-wmean_ctrl_post:+.4f}        ")
print(f"\nWeighted Difference-in-Differences Estimate: {(wmean_treat_post-wmean_treat_pre)-(wmean_ctrl_post-wmean_ctrl_pre):+.4f}")

# 5. REGRESSION-BASED DID
print("\n5. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DID (OLS, unweighted)
print("\nModel 1: Basic DID (OLS, unweighted)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDID Estimate (ELIGIBLE_X_AFTER): {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 2: Basic DID (OLS, weighted)
print("\nModel 2: Basic DID (OLS, weighted by PERWT)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDID Estimate (ELIGIBLE_X_AFTER): {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 3: DID with robust standard errors
print("\nModel 3: Basic DID with Heteroskedasticity-Robust SEs (HC1)")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit(cov_type='HC1')
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 6. DID WITH COVARIATES
print("\n6. DID WITH COVARIATES")
print("-" * 40)

# Check available covariates
print("\nAvailable demographic covariates:")
print(f"  SEX: {df['SEX'].value_counts().to_dict()}")
print(f"  MARST: {df['MARST'].value_counts().to_dict()}")
print(f"  NCHILD range: {df['NCHILD'].min()}-{df['NCHILD'].max()}")

# Create dummy variables for categorical variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)  # 1 = married spouse present, 2 = married spouse absent

# Education recode - check values
print(f"\nEDUC_RECODE values: {df['EDUC_RECODE'].value_counts()}")

# Create education dummies if EDUC_RECODE exists
if 'EDUC_RECODE' in df.columns:
    df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
    df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
    df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
    df['EDUC_BAPLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model 4: DID with demographic covariates
print("\nModel 4: DID with Demographic Covariates (unweighted, robust SEs)")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE'
model4 = smf.ols(formula4, data=df).fit(cov_type='HC1')
print(model4.summary().tables[1])
print(f"\nDID Estimate (ELIGIBLE_X_AFTER): {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model4.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 5: DID with demographic + education covariates
print("\nModel 5: DID with Demographic + Education Covariates (unweighted, robust SEs)")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BAPLUS'
model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')
print(model5.summary().tables[1])
print(f"\nDID Estimate (ELIGIBLE_X_AFTER): {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 6: DID with covariates, weighted
print("\nModel 6: DID with Covariates (weighted by PERWT)")
model6 = smf.wls(formula5, data=df, weights=df['PERWT']).fit()
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model6.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 7. DID WITH YEAR FIXED EFFECTS
print("\n7. DID WITH YEAR FIXED EFFECTS")
print("-" * 40)

# Create year dummies
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# Model 7: DID with year fixed effects
print("\nModel 7: DID with Year Fixed Effects (unweighted, robust SEs)")
year_dummies = ' + '.join([f'YEAR_{y}' for y in sorted(df['YEAR'].unique())[1:]])  # Exclude first year as reference
formula7 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + {year_dummies}'
model7 = smf.ols(formula7, data=df).fit(cov_type='HC1')
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model7.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model7.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model7.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 8: DID with year FE + covariates
print("\nModel 8: DID with Year Fixed Effects + Covariates (unweighted, robust SEs)")
formula8 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BAPLUS + {year_dummies}'
model8 = smf.ols(formula8, data=df).fit(cov_type='HC1')
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model8.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model8.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model8.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 8. DID WITH STATE FIXED EFFECTS
print("\n8. DID WITH STATE FIXED EFFECTS")
print("-" * 40)

# Create state dummies
states = sorted(df['STATEFIP'].unique())
for state in states:
    df[f'STATE_{state}'] = (df['STATEFIP'] == state).astype(int)

# Model 9: DID with state FE
print("\nModel 9: DID with State Fixed Effects (unweighted, robust SEs)")
state_dummies = ' + '.join([f'STATE_{s}' for s in states[1:]])  # Exclude first state as reference
formula9 = f'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + {state_dummies}'
model9 = smf.ols(formula9, data=df).fit(cov_type='HC1')
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model9.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model9.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model9.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model9.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model9.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 10: Full model - year FE + state FE + covariates
print("\nModel 10: Full Model - Year FE + State FE + Covariates (unweighted, robust SEs)")
formula10 = f'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BAPLUS + {year_dummies} + {state_dummies}'
model10 = smf.ols(formula10, data=df).fit(cov_type='HC1')
print(f"DID Estimate (ELIGIBLE_X_AFTER): {model10.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model10.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model10.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model10.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model10.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 9. PARALLEL TRENDS CHECK
print("\n9. PARALLEL TRENDS CHECK")
print("-" * 40)

# Calculate annual means by treatment status
print("\nAnnual FT Rates by Treatment Status:")
annual_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
annual_means.columns = ['Control (31-35)', 'Treatment (26-30)']
print(annual_means)
print(f"\nDifference (Treatment - Control):")
annual_means['Difference'] = annual_means['Treatment (26-30)'] - annual_means['Control (31-35)']
print(annual_means['Difference'])

# Test for parallel trends in pre-period
print("\nPre-Period Trend Test:")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_TREND'] = pre_data['YEAR'] - 2008  # Normalize years

# Test: interaction of treatment with year trend in pre-period
pre_model = smf.ols('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE:YEAR_TREND', data=pre_data).fit(cov_type='HC1')
print(f"Interaction (ELIGIBLE:YEAR_TREND) coefficient: {pre_model.params['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"Standard Error: {pre_model.bse['ELIGIBLE:YEAR_TREND']:.4f}")
print(f"p-value: {pre_model.pvalues['ELIGIBLE:YEAR_TREND']:.4f}")
if pre_model.pvalues['ELIGIBLE:YEAR_TREND'] > 0.05:
    print("=> Parallel trends assumption appears satisfied (p > 0.05)")
else:
    print("=> WARNING: Parallel trends assumption may be violated (p < 0.05)")

# 10. SUMMARY OF RESULTS
print("\n" + "=" * 80)
print("10. SUMMARY OF RESULTS")
print("=" * 80)

results_summary = []
models_list = [
    ("Model 1: Basic DID (unweighted)", model1, 'standard'),
    ("Model 2: Basic DID (weighted)", model2, 'standard'),
    ("Model 3: Basic DID (robust SEs)", model3, 'robust'),
    ("Model 4: DID + Demographics (robust SEs)", model4, 'robust'),
    ("Model 5: DID + Demographics + Education (robust SEs)", model5, 'robust'),
    ("Model 7: DID + Year FEs (robust SEs)", model7, 'robust'),
    ("Model 8: DID + Year FEs + Covariates (robust SEs)", model8, 'robust'),
    ("Model 9: DID + State FEs (robust SEs)", model9, 'robust'),
    ("Model 10: Full Model (robust SEs)", model10, 'robust'),
]

print("\nModel Summary Table:")
print("-" * 100)
print(f"{'Model':<50} {'Coefficient':>12} {'SE':>10} {'95% CI':>22} {'p-value':>10}")
print("-" * 100)

for name, model, se_type in models_list:
    coef = model.params['ELIGIBLE_X_AFTER']
    se = model.bse['ELIGIBLE_X_AFTER']
    ci_low = model.conf_int().loc['ELIGIBLE_X_AFTER', 0]
    ci_high = model.conf_int().loc['ELIGIBLE_X_AFTER', 1]
    pval = model.pvalues['ELIGIBLE_X_AFTER']
    print(f"{name:<50} {coef:>12.4f} {se:>10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {pval:>10.4f}")
    results_summary.append({
        'Model': name,
        'Coefficient': coef,
        'SE': se,
        'CI_Low': ci_low,
        'CI_High': ci_high,
        'p-value': pval
    })

print("-" * 100)

# PREFERRED SPECIFICATION
print("\n" + "=" * 80)
print("PREFERRED SPECIFICATION: Model 5")
print("DID with Demographic and Education Covariates, Robust Standard Errors")
print("=" * 80)
print(f"\nEffect Size: {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"Sample Size: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")

# Save results to CSV
results_df = pd.DataFrame(results_summary)
results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# 11. BALANCE TABLE
print("\n11. BALANCE TABLE (PRE-PERIOD)")
print("-" * 40)

# Compare treatment and control in pre-period
pre_data = df[df['AFTER'] == 0]

balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC_HS', 'EDUC_SOMECOLL',
                'EDUC_TWOYEAR', 'EDUC_BAPLUS']

print("\nPre-Period Covariate Balance:")
print(f"{'Variable':<20} {'Control Mean':>12} {'Treatment Mean':>15} {'Difference':>12} {'p-value':>10}")
print("-" * 70)

for var in balance_vars:
    if var in pre_data.columns:
        ctrl_mean = pre_data[pre_data['ELIGIBLE']==0][var].mean()
        treat_mean = pre_data[pre_data['ELIGIBLE']==1][var].mean()
        diff = treat_mean - ctrl_mean
        # t-test
        ctrl_vals = pre_data[pre_data['ELIGIBLE']==0][var].dropna()
        treat_vals = pre_data[pre_data['ELIGIBLE']==1][var].dropna()
        t_stat, p_val = stats.ttest_ind(ctrl_vals, treat_vals)
        print(f"{var:<20} {ctrl_mean:>12.4f} {treat_mean:>15.4f} {diff:>12.4f} {p_val:>10.4f}")

# 12. HETEROGENEITY BY SEX
print("\n12. HETEROGENEITY ANALYSIS BY SEX")
print("-" * 40)

# Male subsample
print("\nMales only:")
df_male = df[df['SEX'] == 1]
model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_male).fit(cov_type='HC1')
print(f"DID Estimate: {model_male.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_X_AFTER']:.4f})")
print(f"95% CI: [{model_male.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model_male.pvalues['ELIGIBLE_X_AFTER']:.4f}, N={int(model_male.nobs):,}")

# Female subsample
print("\nFemales only:")
df_female = df[df['SEX'] == 2]
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_female).fit(cov_type='HC1')
print(f"DID Estimate: {model_female.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_X_AFTER']:.4f})")
print(f"95% CI: [{model_female.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model_female.pvalues['ELIGIBLE_X_AFTER']:.4f}, N={int(model_female.nobs):,}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

# Save annual means for plotting
annual_means.to_csv('annual_means.csv')
print("\nAnnual means saved to annual_means.csv")

# Output full regression results for Model 5 (preferred)
print("\nFull regression output for Model 5 (preferred specification):")
print(model5.summary())

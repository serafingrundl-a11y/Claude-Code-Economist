"""
DACA Replication Analysis
Effect of DACA Eligibility on Full-Time Employment
Difference-in-Differences Approach

This script conducts the main analysis for the DACA replication study.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Load Data
# =============================================================================
print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)

print(f"\nDataset dimensions: {df.shape[0]} observations, {df.shape[1]} variables")

# =============================================================================
# Data Preparation
# =============================================================================

# Create numeric versions of categorical variables for regression
df['FEMALE'] = (df['SEX'] == 'Female').astype(int)

# Marital status - create married indicator
df['MARRIED'] = df['MARST'].isin(['Married, spouse present', 'Married, spouse absent']).astype(int)

# Education dummies
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=False)
df = pd.concat([df, educ_dummies], axis=1)

# Create reference category: Less than High School
# But note: only 9 obs with less than HS, so use High School as reference
df['EDUC_SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
df['EDUC_LT_HS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)

# Race dummies
df['WHITE'] = (df['RACE_RECODE'] == 'White').astype(int)

# Metropolitan area indicator
df['METRO_IND'] = df['METRO'].isin(['In metropolitan area: In central/principal city',
                                    'In metropolitan area: Not in central/principal city',
                                    'In metropolitan area: Central/principal city status indeterminable (mixed)']).astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Year indicators
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# State indicators (for fixed effects)
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

# =============================================================================
# Summary Statistics
# =============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\n--- Sample Sizes ---")
print(f"Total observations: {len(df)}")
print(f"\nBy ELIGIBLE status:")
print(f"  Treatment (ELIGIBLE=1, ages 26-30 in June 2012): {(df['ELIGIBLE']==1).sum()}")
print(f"  Control (ELIGIBLE=0, ages 31-35 in June 2012): {(df['ELIGIBLE']==0).sum()}")

print(f"\nBy Time Period:")
print(f"  Pre-DACA (2008-2011): {(df['AFTER']==0).sum()}")
print(f"  Post-DACA (2013-2016): {(df['AFTER']==1).sum()}")

print(f"\nBy ELIGIBLE x AFTER:")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
crosstab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(crosstab)

# Outcome variable
print("\n--- Full-Time Employment (FT) ---")
print(f"Overall FT rate: {df['FT'].mean()*100:.2f}%")

ft_by_group = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()
print("\nFT Rate by Treatment Group and Time Period:")
print(f"  Control, Pre-DACA:  {ft_by_group[(0,0)]*100:.2f}%")
print(f"  Control, Post-DACA: {ft_by_group[(0,1)]*100:.2f}%")
print(f"  Treatment, Pre-DACA:  {ft_by_group[(1,0)]*100:.2f}%")
print(f"  Treatment, Post-DACA: {ft_by_group[(1,1)]*100:.2f}%")

# Simple DiD calculation
control_diff = ft_by_group[(0,1)] - ft_by_group[(0,0)]
treat_diff = ft_by_group[(1,1)] - ft_by_group[(1,0)]
simple_did = treat_diff - control_diff

print(f"\n--- Simple DiD Calculation ---")
print(f"Control group change: {control_diff*100:.3f} pp")
print(f"Treatment group change: {treat_diff*100:.3f} pp")
print(f"DiD estimate: {simple_did*100:.3f} pp")

# =============================================================================
# Covariate Balance Table
# =============================================================================
print("\n" + "="*80)
print("COVARIATE BALANCE (Pre-DACA Period)")
print("="*80)

# Pre-DACA observations only for balance check
pre_df = df[df['AFTER'] == 0].copy()

balance_vars = ['FEMALE', 'AGE_IN_JUNE_2012', 'MARRIED', 'NCHILD', 'HAS_CHILDREN',
                'EDUC_SOME_COLLEGE', 'EDUC_TWO_YEAR', 'EDUC_BA_PLUS',
                'WHITE', 'METRO_IND', 'AGE_AT_IMMIGRATION', 'YRSUSA1']

print("\n{:<25} {:>12} {:>12} {:>12} {:>10}".format(
    'Variable', 'Control', 'Treatment', 'Diff', 'p-value'))
print("-"*75)

for var in balance_vars:
    if var in pre_df.columns:
        control_mean = pre_df[pre_df['ELIGIBLE']==0][var].mean()
        treat_mean = pre_df[pre_df['ELIGIBLE']==1][var].mean()
        diff = treat_mean - control_mean

        # t-test
        control_vals = pre_df[pre_df['ELIGIBLE']==0][var].dropna()
        treat_vals = pre_df[pre_df['ELIGIBLE']==1][var].dropna()
        if len(control_vals) > 1 and len(treat_vals) > 1:
            t_stat, p_val = stats.ttest_ind(control_vals, treat_vals)
        else:
            p_val = np.nan

        print("{:<25} {:>12.3f} {:>12.3f} {:>12.3f} {:>10.3f}".format(
            var, control_mean, treat_mean, diff, p_val))

# =============================================================================
# MAIN REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
formula2 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012'
model2 = smf.ols(formula2, data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic + education controls
print("\n--- Model 3: DiD with Demographics + Education ---")
formula3 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012 + EDUC_SOME_COLLEGE + EDUC_TWO_YEAR + EDUC_BA_PLUS'
model3 = smf.ols(formula3, data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: Full model with state and year fixed effects
print("\n--- Model 4: Full Model with State and Year FE ---")
# Create state fixed effects using C() notation
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012 + EDUC_SOME_COLLEGE + EDUC_TWO_YEAR + EDUC_BA_PLUS + C(STATEFIP) + C(YEAR)'
model4 = smf.ols(formula4, data=df).fit(cov_type='HC1')

# Extract key coefficients
key_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER', 'FEMALE', 'MARRIED',
            'NCHILD', 'AGE_IN_JUNE_2012', 'EDUC_SOME_COLLEGE', 'EDUC_TWO_YEAR', 'EDUC_BA_PLUS']
print("\nKey Coefficients from Model 4:")
print("{:<25} {:>12} {:>12} {:>12}".format('Variable', 'Coef', 'Std Err', 't'))
print("-"*60)
for var in key_vars:
    if var in model4.params.index:
        coef = model4.params[var]
        se = model4.bse[var]
        t = model4.tvalues[var]
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.3f}".format(var, coef, se, t))

print(f"\nN = {int(model4.nobs)}, R-squared = {model4.rsquared:.4f}")

# Model 5: With state-level policy controls
print("\n--- Model 5: Full Model with State Policies ---")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012 + EDUC_SOME_COLLEGE + EDUC_TWO_YEAR + EDUC_BA_PLUS + UNEMP + LFPR + DRIVERSLICENSES + SECURECOMMUNITIES + C(STATEFIP) + C(YEAR)'
model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')

print("\nKey Coefficients from Model 5:")
print("{:<25} {:>12} {:>12} {:>12}".format('Variable', 'Coef', 'Std Err', 't'))
print("-"*60)
key_vars5 = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER', 'FEMALE', 'MARRIED',
             'NCHILD', 'AGE_IN_JUNE_2012', 'EDUC_SOME_COLLEGE', 'EDUC_TWO_YEAR', 'EDUC_BA_PLUS',
             'UNEMP', 'LFPR', 'DRIVERSLICENSES', 'SECURECOMMUNITIES']
for var in key_vars5:
    if var in model5.params.index:
        coef = model5.params[var]
        se = model5.bse[var]
        t = model5.tvalues[var]
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.3f}".format(var, coef, se, t))

print(f"\nN = {int(model5.nobs)}, R-squared = {model5.rsquared:.4f}")

# =============================================================================
# PREFERRED SPECIFICATION: Model with clustered standard errors
# =============================================================================
print("\n" + "="*80)
print("PREFERRED SPECIFICATION")
print("="*80)

# Cluster at state level for robust inference
formula_preferred = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012 + EDUC_SOME_COLLEGE + EDUC_TWO_YEAR + EDUC_BA_PLUS + C(STATEFIP) + C(YEAR)'
model_preferred = smf.ols(formula_preferred, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print("\nPreferred Model: DiD with State/Year FE, Clustered SE at State Level")
print("="*80)
print("\nKey Coefficients:")
print("{:<25} {:>12} {:>12} {:>12} {:>12}".format('Variable', 'Coef', 'Std Err', 't', 'p-value'))
print("-"*75)
for var in key_vars:
    if var in model_preferred.params.index:
        coef = model_preferred.params[var]
        se = model_preferred.bse[var]
        t = model_preferred.tvalues[var]
        p = model_preferred.pvalues[var]
        print("{:<25} {:>12.4f} {:>12.4f} {:>12.3f} {:>12.4f}".format(var, coef, se, t, p))

did_coef = model_preferred.params['ELIGIBLE_x_AFTER']
did_se = model_preferred.bse['ELIGIBLE_x_AFTER']
did_ci = model_preferred.conf_int().loc['ELIGIBLE_x_AFTER']

print(f"\n{'='*80}")
print("MAIN RESULT - DACA EFFECT ON FULL-TIME EMPLOYMENT")
print(f"{'='*80}")
print(f"\nDiD Estimate (ELIGIBLE x AFTER): {did_coef:.4f}")
print(f"Robust Standard Error (clustered at state): {did_se:.4f}")
print(f"95% Confidence Interval: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"t-statistic: {model_preferred.tvalues['ELIGIBLE_x_AFTER']:.3f}")
print(f"p-value: {model_preferred.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"\nInterpretation: DACA eligibility is associated with a {did_coef*100:.2f} percentage point")
print(f"{'increase' if did_coef > 0 else 'decrease'} in the probability of full-time employment.")

print(f"\nSample Size: {int(model_preferred.nobs)}")
print(f"R-squared: {model_preferred.rsquared:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# 1. Linear probability model without weights
print("\n--- Robustness 1: Without State Fixed Effects ---")
formula_rob1 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + MARRIED + NCHILD + AGE_IN_JUNE_2012 + EDUC_SOME_COLLEGE + EDUC_TWO_YEAR + EDUC_BA_PLUS + C(YEAR)'
model_rob1 = smf.ols(formula_rob1, data=df).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_rob1.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_rob1.bse['ELIGIBLE_x_AFTER']:.4f})")

# 2. With survey weights
print("\n--- Robustness 2: With Survey Weights (PERWT) ---")
import statsmodels.api as sm
X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_x_AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'AGE_IN_JUNE_2012',
        'EDUC_SOME_COLLEGE', 'EDUC_TWO_YEAR', 'EDUC_BA_PLUS']].copy()
# Add year dummies
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    X[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
X = sm.add_constant(X)
y = df['FT']
w = df['PERWT']
model_rob2 = sm.WLS(y, X, weights=w).fit(cov_type='HC1')
print(f"DiD Coefficient: {model_rob2.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_rob2.bse['ELIGIBLE_x_AFTER']:.4f})")

# 3. Placebo test: Pre-treatment trend
print("\n--- Robustness 3: Pre-Trend Analysis ---")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_TREND'] = pre_data['YEAR'] - 2008
pre_data['ELIGIBLE_x_YEAR'] = pre_data['ELIGIBLE'] * pre_data['YEAR_TREND']
formula_pre = 'FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_x_YEAR + FEMALE + MARRIED + NCHILD'
model_pre = smf.ols(formula_pre, data=pre_data).fit(cov_type='HC1')
print(f"Pre-trend (ELIGIBLE x YEAR): {model_pre.params['ELIGIBLE_x_YEAR']:.4f} (SE: {model_pre.bse['ELIGIBLE_x_YEAR']:.4f})")
print(f"p-value: {model_pre.pvalues['ELIGIBLE_x_YEAR']:.4f}")

# 4. Event study analysis
print("\n--- Robustness 4: Event Study Coefficients ---")
# Create year-specific treatment effects
for year in df['YEAR'].unique():
    df[f'ELIGIBLE_x_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

event_vars = ['ELIGIBLE_x_2008', 'ELIGIBLE_x_2009', 'ELIGIBLE_x_2010', 'ELIGIBLE_x_2011',
              'ELIGIBLE_x_2013', 'ELIGIBLE_x_2014', 'ELIGIBLE_x_2015', 'ELIGIBLE_x_2016']
# Use 2011 as reference year (year before treatment)
event_vars_model = [v for v in event_vars if v != 'ELIGIBLE_x_2011']

formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(event_vars_model) + ' + FEMALE + MARRIED + NCHILD + C(YEAR)'
model_event = smf.ols(formula_event, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for var in event_vars_model:
    year = var.split('_')[-1]
    coef = model_event.params[var]
    se = model_event.bse[var]
    print(f"  {year}: {coef:.4f} ({se:.4f})")

# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SUBGROUP ANALYSIS (Exploratory)")
print("="*80)

# By gender
print("\n--- By Gender ---")
for gender, label in [(1, 'Female'), (0, 'Male')]:
    sub_df = df[df['FEMALE'] == gender].copy()
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + MARRIED + NCHILD + C(YEAR)',
                        data=sub_df).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_x_AFTER']:.4f}), N = {int(model_sub.nobs)}")

# By marital status
print("\n--- By Marital Status ---")
for married, label in [(1, 'Married'), (0, 'Not Married')]:
    sub_df = df[df['MARRIED'] == married].copy()
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + NCHILD + C(YEAR)',
                        data=sub_df).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['ELIGIBLE_x_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_x_AFTER']:.4f}), N = {int(model_sub.nobs)}")

# =============================================================================
# PARALLEL TRENDS CHECK BY YEAR
# =============================================================================
print("\n" + "="*80)
print("PARALLEL TRENDS - FT RATE BY YEAR AND GROUP")
print("="*80)

yearly_ft = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nFull-Time Employment Rate by Year:")
print(yearly_ft.round(4))

# =============================================================================
# SAVE RESULTS
# =============================================================================

# Save key results to file
results_dict = {
    'did_estimate': did_coef,
    'did_se': did_se,
    'did_ci_lower': did_ci[0],
    'did_ci_upper': did_ci[1],
    'did_pvalue': model_preferred.pvalues['ELIGIBLE_x_AFTER'],
    'n_obs': int(model_preferred.nobs),
    'r_squared': model_preferred.rsquared
}

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nSummary of Preferred Estimate:")
print(f"  Effect Size: {did_coef:.4f} ({did_coef*100:.2f} percentage points)")
print(f"  Standard Error: {did_se:.4f}")
print(f"  95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]")
print(f"  Sample Size: {int(model_preferred.nobs)}")

# Export results for LaTeX
with open('analysis_results.txt', 'w') as f:
    f.write("DACA REPLICATION ANALYSIS - KEY RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Preferred Estimate (DiD with State/Year FE, Clustered SE)\n")
    f.write(f"Effect Size: {did_coef:.4f}\n")
    f.write(f"Standard Error: {did_se:.4f}\n")
    f.write(f"95% CI: [{did_ci[0]:.4f}, {did_ci[1]:.4f}]\n")
    f.write(f"Sample Size: {int(model_preferred.nobs)}\n")
    f.write(f"R-squared: {model_preferred.rsquared:.4f}\n")

print("\nResults saved to analysis_results.txt")

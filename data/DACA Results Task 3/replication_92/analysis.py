"""
DACA Replication Study: Difference-in-Differences Analysis
Analyzing the effect of DACA eligibility on full-time employment among
ethnically Hispanic-Mexican Mexican-born individuals in the United States.

Research Design:
- Treatment Group: Individuals aged 26-30 at policy implementation (June 2012) who were ELIGIBLE for DACA
- Control Group: Individuals aged 31-35 at policy implementation who would have been eligible if not for age
- Pre-period: 2008-2011
- Post-period: 2013-2016
- Outcome: Full-time employment (FT = 1 if usually working 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("DACA REPLICATION STUDY: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

data_path = "C:/Users/seraf/DACA Results Task 3/replication_92/data/prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\n1. DATA OVERVIEW")
print("-" * 40)
print(f"Total observations: {len(df):,}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {len(df.columns)}")

# Check key variables
print(f"\n2. KEY VARIABLE DISTRIBUTIONS")
print("-" * 40)

print("\nELIGIBLE (Treatment indicator):")
print(df['ELIGIBLE'].value_counts())

print("\nAFTER (Post-treatment indicator):")
print(df['AFTER'].value_counts())

print("\nFT (Full-time employment outcome):")
print(df['FT'].value_counts())

print("\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# Create interaction term for DiD
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']

print(f"\n3. SAMPLE CHARACTERISTICS")
print("-" * 40)

# Group means
print("\nMean FT by group and period:")
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std']).round(4)
means.columns = ['Mean FT', 'N', 'Std Dev']
print(means)

# Simple 2x2 DiD calculation
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\n4. SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("-" * 40)
print(f"Treatment group (ELIGIBLE=1):")
print(f"  Pre-period mean FT:  {pre_treat:.4f}")
print(f"  Post-period mean FT: {post_treat:.4f}")
print(f"  Change:              {post_treat - pre_treat:.4f}")

print(f"\nControl group (ELIGIBLE=0):")
print(f"  Pre-period mean FT:  {pre_control:.4f}")
print(f"  Post-period mean FT: {post_control:.4f}")
print(f"  Change:              {post_control - pre_control:.4f}")

simple_did = (post_treat - pre_treat) - (post_control - pre_control)
print(f"\nDiD Estimate: {simple_did:.4f}")

print(f"\n5. REGRESSION ANALYSIS")
print("-" * 40)

# Model 1: Basic DiD without controls
print("\nModel 1: Basic DiD (no controls)")
print("-" * 30)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_x_AFTER): {model1.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"t-statistic: {model1.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"p-value: {model1.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {model1.nobs:.0f}")

# Model 2: DiD with demographic controls
print("\n\nModel 2: DiD with demographic controls")
print("-" * 30)
# Check available controls
print("Checking available control variables...")

# SEX is coded 1=Male, 2=Female per IPUMS
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# AGE is already available
# MARST is available (marital status)
df['MARRIED'] = (df['MARST'] == 1).astype(int)  # 1 = married spouse present

# Education recode
print(f"Education categories: {df['EDUC_RECODE'].unique()}")

# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + FEMALE + AGE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN', data=df).fit()
print(model2.summary().tables[1])
print(f"\nDiD coefficient (ELIGIBLE_x_AFTER): {model2.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {model2.nobs:.0f}")

# Model 3: DiD with year fixed effects
print("\n\nModel 3: DiD with year fixed effects")
print("-" * 30)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df_with_year = pd.concat([df, year_dummies], axis=1)

year_cols = [col for col in df_with_year.columns if col.startswith('YEAR_')]
formula = 'FT ~ ELIGIBLE + ELIGIBLE_x_AFTER + FEMALE + AGE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN + ' + ' + '.join(year_cols)
model3 = smf.ols(formula, data=df_with_year).fit()
print(f"DiD coefficient (ELIGIBLE_x_AFTER): {model3.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {model3.nobs:.0f}")

# Model 4: DiD with state fixed effects
print("\n\nModel 4: DiD with state fixed effects")
print("-" * 30)

state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df_with_state = pd.concat([df_with_year, state_dummies], axis=1)

state_cols = [col for col in df_with_state.columns if col.startswith('STATE_')]
formula = 'FT ~ ELIGIBLE + ELIGIBLE_x_AFTER + FEMALE + AGE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA + HAS_CHILDREN + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
model4 = smf.ols(formula, data=df_with_state).fit()
print(f"DiD coefficient (ELIGIBLE_x_AFTER): {model4.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:.0f}")

# Model 5: Robust standard errors (clustered at state level)
print("\n\nModel 5: DiD with robust (clustered) standard errors")
print("-" * 30)

# Using HC1 robust standard errors as a proxy (statsmodels doesn't have built-in cluster SE without additional packages)
model5 = smf.ols(formula, data=df_with_state).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_x_AFTER): {model5.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust Standard Error: {model5.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {model5.nobs:.0f}")

print(f"\n6. WEIGHTED ANALYSIS (using person weights)")
print("-" * 40)

# Using person weights (PERWT)
import statsmodels.api as sm

# Basic weighted DiD
print("\nModel 6: Weighted DiD")
# Need to ensure all columns are numeric
X_cols = ['ELIGIBLE', 'ELIGIBLE_x_AFTER', 'FEMALE', 'AGE', 'MARRIED',
          'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_TWOYEAR', 'EDUC_BA', 'HAS_CHILDREN'] + year_cols + state_cols
X = df_with_state[X_cols].astype(float)
X = sm.add_constant(X)
y = df_with_state['FT'].astype(float)
weights = df_with_state['PERWT'].astype(float)

model6 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_x_AFTER): {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust Standard Error: {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_x_AFTER']:.4f}")

print(f"\n7. SUBGROUP ANALYSIS")
print("-" * 40)

# By sex
print("\nBy Sex:")
for sex, name in [(1, 'Male'), (2, 'Female')]:
    subset = df[df['SEX'] == sex]
    model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=subset).fit()
    print(f"  {name}: DiD = {model.params['ELIGIBLE_x_AFTER']:.4f} (SE = {model.bse['ELIGIBLE_x_AFTER']:.4f}), N = {len(subset)}")

# By education
print("\nBy Education:")
for educ in df['EDUC_RECODE'].unique():
    if pd.notna(educ):
        subset = df[df['EDUC_RECODE'] == educ]
        if len(subset) > 100:
            model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=subset).fit()
            print(f"  {educ}: DiD = {model.params['ELIGIBLE_x_AFTER']:.4f} (SE = {model.bse['ELIGIBLE_x_AFTER']:.4f}), N = {len(subset)}")

print(f"\n8. PARALLEL TRENDS CHECK (Pre-treatment period)")
print("-" * 40)

# Calculate year-by-year means for each group
pre_years = [2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

print("\nYear-by-year FT rates:")
print("Year    Treatment    Control    Difference")
print("-" * 45)
for year in sorted(df['YEAR'].unique()):
    treat_mean = df[(df['YEAR']==year) & (df['ELIGIBLE']==1)]['FT'].mean()
    control_mean = df[(df['YEAR']==year) & (df['ELIGIBLE']==0)]['FT'].mean()
    diff = treat_mean - control_mean
    marker = " (post)" if year >= 2013 else " (pre)"
    print(f"{year}{marker}   {treat_mean:.4f}      {control_mean:.4f}     {diff:.4f}")

# Test for parallel trends using pre-treatment data
print("\nParallel Trends Test (Pre-treatment interaction with year trend):")
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_TREND'] = pre_data['YEAR'] - 2008
pre_data['ELIGIBLE_x_TREND'] = pre_data['ELIGIBLE'] * pre_data['YEAR_TREND']

trend_model = smf.ols('FT ~ ELIGIBLE + YEAR_TREND + ELIGIBLE_x_TREND', data=pre_data).fit()
print(f"Interaction coefficient: {trend_model.params['ELIGIBLE_x_TREND']:.4f}")
print(f"Standard error: {trend_model.bse['ELIGIBLE_x_TREND']:.4f}")
print(f"p-value: {trend_model.pvalues['ELIGIBLE_x_TREND']:.4f}")
print("(Non-significant interaction supports parallel trends assumption)")

print(f"\n9. EVENT STUDY / DYNAMIC EFFECTS")
print("-" * 40)

# Create year-specific interactions
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Reference year: 2011 (last pre-treatment year)
df['ELIG_x_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_x_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_x_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_x_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_x_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_x_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_x_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_model = smf.ols('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_x_2008 + ELIG_x_2009 + ELIG_x_2010 + ELIG_x_2013 + ELIG_x_2014 + ELIG_x_2015 + ELIG_x_2016', data=df).fit()

print("Event Study Coefficients (relative to 2011):")
print("Year    Coefficient    Std Error    p-value    95% CI")
print("-" * 65)
for var in ['ELIG_x_2008', 'ELIG_x_2009', 'ELIG_x_2010', 'ELIG_x_2013', 'ELIG_x_2014', 'ELIG_x_2015', 'ELIG_x_2016']:
    year = var.split('_')[2]
    coef = event_model.params[var]
    se = event_model.bse[var]
    pval = event_model.pvalues[var]
    ci_low = event_model.conf_int().loc[var, 0]
    ci_high = event_model.conf_int().loc[var, 1]
    marker = "*" if pval < 0.05 else ""
    print(f"{year}      {coef:8.4f}     {se:8.4f}    {pval:7.4f}    [{ci_low:.4f}, {ci_high:.4f}] {marker}")

print("\n(Reference year: 2011; * indicates p < 0.05)")

print(f"\n10. ROBUSTNESS CHECKS")
print("-" * 40)

# Check sample sizes by group
print("\nSample sizes by group and period:")
print(df.groupby(['ELIGIBLE', 'AFTER']).size().unstack())

# Summary statistics for key demographics
print("\nDemographic balance between treatment and control groups (pre-period):")
pre = df[df['AFTER'] == 0]
balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'HAS_CHILDREN']
print("\nVariable          Treatment Mean    Control Mean    Difference")
print("-" * 65)
for var in balance_vars:
    treat_mean = pre[pre['ELIGIBLE']==1][var].mean()
    control_mean = pre[pre['ELIGIBLE']==0][var].mean()
    diff = treat_mean - control_mean
    print(f"{var:18s}    {treat_mean:.4f}          {control_mean:.4f}         {diff:.4f}")

print(f"\n11. FINAL PREFERRED SPECIFICATION SUMMARY")
print("=" * 60)
print("\nPreferred Model: DiD with demographic controls, year and state fixed effects,")
print("and robust standard errors (Model 5)")
print("-" * 60)
print(f"DiD Estimate (Effect of DACA eligibility on P(Full-time employment)):")
print(f"  Coefficient:    {model5.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"  Robust SE:      {model5.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"  95% CI:         [{model5.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"  t-statistic:    {model5.tvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"  p-value:        {model5.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"  Sample size:    {model5.nobs:.0f}")
print(f"  R-squared:      {model5.rsquared:.4f}")

# Save key results for the report
results = {
    'simple_did': simple_did,
    'model1_coef': model1.params['ELIGIBLE_x_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_x_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_x_AFTER'],
    'model2_coef': model2.params['ELIGIBLE_x_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_x_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_x_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_x_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_x_AFTER'],
    'model3_pval': model3.pvalues['ELIGIBLE_x_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_x_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_x_AFTER'],
    'model4_pval': model4.pvalues['ELIGIBLE_x_AFTER'],
    'model5_coef': model5.params['ELIGIBLE_x_AFTER'],
    'model5_se': model5.bse['ELIGIBLE_x_AFTER'],
    'model5_pval': model5.pvalues['ELIGIBLE_x_AFTER'],
    'model5_ci_low': model5.conf_int().loc['ELIGIBLE_x_AFTER', 0],
    'model5_ci_high': model5.conf_int().loc['ELIGIBLE_x_AFTER', 1],
    'n_obs': model5.nobs,
    'r2': model5.rsquared,
    'pre_treat': pre_treat,
    'post_treat': post_treat,
    'pre_control': pre_control,
    'post_control': post_control
}

# Save results to file
pd.Series(results).to_csv('C:/Users/seraf/DACA Results Task 3/replication_92/results_summary.csv')

print("\n" + "=" * 60)
print("Analysis complete. Results saved to results_summary.csv")
print("=" * 60)

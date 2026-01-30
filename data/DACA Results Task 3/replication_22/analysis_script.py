"""
DACA Replication Analysis Script
================================
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the United States.

Treatment: DACA eligibility (ages 26-30 at June 15, 2012)
Control: Ages 31-35 at June 15, 2012 (would have been eligible if not for age)
Outcome: Full-time employment (35+ hours/week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 70)
print("DACA REPLICATION ANALYSIS")
print("=" * 70)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\n1. DATA OVERVIEW")
print("-" * 50)
print(f"Total observations: {len(df):,}")
print(f"Variables: {df.shape[1]}")
print(f"Years: {sorted(df['YEAR'].unique())}")

# Key variable distributions
print(f"\n2. KEY VARIABLE DISTRIBUTIONS")
print("-" * 50)
print(f"ELIGIBLE (1=treated, 0=control):")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"\nAFTER (1=post-DACA, 0=pre-DACA):")
print(df['AFTER'].value_counts().sort_index())
print(f"\nFT (1=full-time, 0=not full-time):")
print(df['FT'].value_counts().sort_index())

# Cross-tabulation
print(f"\n3. SAMPLE SIZES BY GROUP AND PERIOD")
print("-" * 50)
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre-DACA', 'Post-DACA', 'Total']
print(crosstab)

# Weighted sample sizes
print(f"\n4. WEIGHTED POPULATION BY GROUP AND PERIOD")
print("-" * 50)
weighted = df.groupby(['ELIGIBLE', 'AFTER'])['PERWT'].sum().unstack()
weighted.index = ['Control (31-35)', 'Treated (26-30)']
weighted.columns = ['Pre-DACA', 'Post-DACA']
print(weighted.round(0).astype(int))

# Raw full-time employment rates
print(f"\n5. FULL-TIME EMPLOYMENT RATES (UNWEIGHTED)")
print("-" * 50)
ft_rates_unweighted = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates_unweighted.index = ['Control (31-35)', 'Treated (26-30)']
ft_rates_unweighted.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates_unweighted.round(4))

# Weighted full-time employment rates
print(f"\n6. FULL-TIME EMPLOYMENT RATES (WEIGHTED)")
print("-" * 50)

def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()
ft_rates_weighted.index = ['Control (31-35)', 'Treated (26-30)']
ft_rates_weighted.columns = ['Pre-DACA', 'Post-DACA']
print(ft_rates_weighted.round(4))

# Simple Difference-in-Differences calculation
print(f"\n7. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 50)
print("Unweighted:")
# Unweighted
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
treat_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treat_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

diff_control = control_post - control_pre
diff_treat = treat_post - treat_pre
did_unweighted = diff_treat - diff_control

print(f"  Control change (Post - Pre): {diff_control:.4f}")
print(f"  Treated change (Post - Pre): {diff_treat:.4f}")
print(f"  DiD Estimate: {did_unweighted:.4f}")

print("\nWeighted:")
# Weighted
control_pre_w = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                           weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
control_post_w = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                            weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])
treat_pre_w = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                         weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
treat_post_w = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                          weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])

diff_control_w = control_post_w - control_pre_w
diff_treat_w = treat_post_w - treat_pre_w
did_weighted = diff_treat_w - diff_control_w

print(f"  Control change (Post - Pre): {diff_control_w:.4f}")
print(f"  Treated change (Post - Pre): {diff_treat_w:.4f}")
print(f"  DiD Estimate: {did_weighted:.4f}")

# Regression models
print(f"\n8. REGRESSION ANALYSIS")
print("=" * 70)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\nMODEL 1: Basic DiD (Unweighted, No Covariates)")
print("-" * 50)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Estimate: {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"N: {int(model1.nobs)}")

# Model 2: Basic DiD with robust standard errors
print("\nMODEL 2: Basic DiD (Unweighted, Robust SE)")
print("-" * 50)
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"DiD Estimate: {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 3: Weighted DiD
print("\nMODEL 3: Basic DiD (Weighted)")
print("-" * 50)
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Estimate: {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 4: Weighted DiD with robust SE
print("\nMODEL 4: Basic DiD (Weighted, Robust SE)")
print("-" * 50)
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# Create dummy variables for categorical covariates
print("\nPreparing covariates...")

# SEX: 1 = Male, 2 = Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Marital status dummies (reference: Never married/single = 6)
df['MARRIED'] = (df['MARST'] == 1).astype(int)
df['MARRIED_ABSENT'] = (df['MARST'] == 2).astype(int)
df['SEPARATED'] = (df['MARST'] == 3).astype(int)
df['DIVORCED'] = (df['MARST'] == 4).astype(int)
df['WIDOWED'] = (df['MARST'] == 5).astype(int)

# Education dummies (reference: High School)
df['EDUC_LESS_HS'] = (df['EDUC_RECODE'] == 'Less than High School').astype(int)
df['EDUC_SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Number of children
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Year dummies (reference: 2008)
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# State fixed effects preparation
df['STATE_FE'] = pd.Categorical(df['STATEFIP'])

# Model 5: DiD with demographic controls
print("\nMODEL 5: DiD with Demographic Controls (Weighted, Robust SE)")
print("-" * 50)
covariates = ['FEMALE', 'MARRIED', 'MARRIED_ABSENT', 'SEPARATED', 'DIVORCED',
              'EDUC_LESS_HS', 'EDUC_SOME_COLLEGE', 'EDUC_TWO_YEAR', 'EDUC_BA_PLUS',
              'HAS_CHILDREN', 'NCHILD']
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + ' + ' + '.join(covariates)
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 6: DiD with demographic controls and year fixed effects
print("\nMODEL 6: DiD with Demographics + Year FE (Weighted, Robust SE)")
print("-" * 50)
year_vars = ['YEAR_2009', 'YEAR_2010', 'YEAR_2011', 'YEAR_2013', 'YEAR_2014', 'YEAR_2015', 'YEAR_2016']
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + ' + ' + '.join(covariates + year_vars)
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 7: Full model with state fixed effects
print("\nMODEL 7: DiD with Demographics + Year FE + State FE (Weighted, Robust SE)")
print("-" * 50)
formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + ' + ' + '.join(covariates + year_vars) + ' + C(STATEFIP)'
model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model7.rsquared:.4f}")

# Model 8: Add state policy variables
print("\nMODEL 8: Full Model + State Policy Variables (Weighted, Robust SE)")
print("-" * 50)
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY',
               'SECURECOMMUNITIES', 'LFPR', 'UNEMP']
formula8 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + ' + ' + '.join(covariates + year_vars + policy_vars) + ' + C(STATEFIP)'
model8 = smf.wls(formula8, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model8.rsquared:.4f}")

# PREFERRED MODEL
print("\n" + "=" * 70)
print("PREFERRED MODEL SUMMARY (Model 7)")
print("=" * 70)
print("""
Model: Weighted Least Squares with Difference-in-Differences
Dependent Variable: Full-time employment (FT)
Treatment: DACA eligibility (ages 26-30)
Control: Ages 31-35 at June 15, 2012
Weighting: Person weights (PERWT)
Standard Errors: Heteroskedasticity-robust (HC1)
""")
print(f"DiD Effect Estimate: {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"t-statistic: {model7.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {int(model7.nobs)}")
print(f"R-squared: {model7.rsquared:.4f}")

# Interpretation
effect_pct = model7.params['ELIGIBLE_AFTER'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.1f} percentage point")
print("change in the probability of full-time employment.")

# Parallel trends check: Year-by-year analysis
print("\n" + "=" * 70)
print("9. PARALLEL TRENDS ANALYSIS")
print("=" * 70)

# Create year-specific treatment effects
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_by_year.columns = ['Control (31-35)', 'Treated (26-30)']

print("\nWeighted FT Employment Rates by Year:")
print(ft_by_year.round(4))

print("\nDifference (Treated - Control) by Year:")
ft_by_year['Difference'] = ft_by_year['Treated (26-30)'] - ft_by_year['Control (31-35)']
print(ft_by_year['Difference'].round(4))

# Event study style regression
print("\nEvent Study Regression (Year-specific treatment effects):")
print("-" * 50)
# Create year-treatment interactions (reference: 2011, last pre-treatment year)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

event_study_vars = [f'ELIGIBLE_YEAR_{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]
formula_es = 'FT ~ ELIGIBLE + ' + ' + '.join(year_vars + event_study_vars + covariates) + ' + C(STATEFIP)'
model_es = smf.wls(formula_es, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Year-specific effects relative to 2011 (reference year):")
for var in event_study_vars:
    year = var.split('_')[-1]
    coef = model_es.params[var]
    se = model_es.bse[var]
    pval = model_es.pvalues[var]
    sig = '*' if pval < 0.05 else ''
    print(f"  {year}: {coef:7.4f} (SE: {se:.4f}){sig}")

# Robustness: By gender
print("\n" + "=" * 70)
print("10. HETEROGENEITY ANALYSIS")
print("=" * 70)

print("\nBy Gender:")
print("-" * 50)
for gender, label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == gender]
    model_sub = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + ' + ' + '.join(covariates[1:] + year_vars) + ' + C(STATEFIP)',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_sub.nobs)}")

print("\nBy Education Level:")
print("-" * 50)
for educ in ['High School Degree', 'Some College', 'BA+']:
    df_sub = df[df['EDUC_RECODE'] == educ]
    if len(df_sub) > 100:
        try:
            model_sub = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + HAS_CHILDREN + ' + ' + '.join(year_vars),
                                data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
            print(f"{educ}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_sub.nobs)}")
        except:
            print(f"{educ}: Could not estimate")

print("\nBy Marital Status:")
print("-" * 50)
for married_status, label in [(1, 'Married'), (0, 'Not Married')]:
    df_sub = df[df['MARRIED'] == married_status]
    model_sub = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + HAS_CHILDREN + ' + ' + '.join(year_vars) + ' + C(STATEFIP)',
                        data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{label}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_sub.nobs)}")

# Summary statistics for report
print("\n" + "=" * 70)
print("11. SUMMARY STATISTICS FOR REPORT")
print("=" * 70)

summary_stats = pd.DataFrame()
for (elig, after), group in df.groupby(['ELIGIBLE', 'AFTER']):
    label = f"{'Treated' if elig else 'Control'}_{'Post' if after else 'Pre'}"

    stats_dict = {
        'FT Rate': np.average(group['FT'], weights=group['PERWT']),
        'Age (mean)': np.average(group['AGE'], weights=group['PERWT']),
        'Female (%)': np.average(group['FEMALE'], weights=group['PERWT']),
        'Married (%)': np.average(group['MARRIED'], weights=group['PERWT']),
        'Has Children (%)': np.average(group['HAS_CHILDREN'], weights=group['PERWT']),
        'N (unweighted)': len(group),
        'N (weighted)': group['PERWT'].sum()
    }
    summary_stats[label] = pd.Series(stats_dict)

print(summary_stats.T.round(4))

# Covariate balance
print("\n" + "=" * 70)
print("12. COVARIATE BALANCE (Pre-Period)")
print("=" * 70)
df_pre = df[df['AFTER'] == 0]

balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'NCHILD',
                'EDUC_BA_PLUS', 'EDUC_SOME_COLLEGE']
print(f"\n{'Variable':<20} {'Control Mean':>12} {'Treated Mean':>12} {'Difference':>12}")
print("-" * 58)
for var in balance_vars:
    ctrl_mean = np.average(df_pre[df_pre['ELIGIBLE']==0][var], weights=df_pre[df_pre['ELIGIBLE']==0]['PERWT'])
    treat_mean = np.average(df_pre[df_pre['ELIGIBLE']==1][var], weights=df_pre[df_pre['ELIGIBLE']==1]['PERWT'])
    diff = treat_mean - ctrl_mean
    print(f"{var:<20} {ctrl_mean:>12.4f} {treat_mean:>12.4f} {diff:>12.4f}")

# Final summary table for all models
print("\n" + "=" * 70)
print("13. REGRESSION RESULTS SUMMARY TABLE")
print("=" * 70)

models = [
    ("Model 1: Basic DiD (OLS)", model1, False),
    ("Model 2: Basic DiD (Robust SE)", model2, False),
    ("Model 3: Weighted DiD", model3, False),
    ("Model 4: Weighted DiD (Robust SE)", model4, False),
    ("Model 5: + Demographics", model5, False),
    ("Model 6: + Year FE", model6, False),
    ("Model 7: + State FE (PREFERRED)", model7, True),
    ("Model 8: + State Policies", model8, False),
]

print(f"\n{'Model':<40} {'DiD Estimate':>12} {'Robust SE':>12} {'p-value':>10} {'N':>8}")
print("-" * 84)
for name, model, preferred in models:
    est = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    n = int(model.nobs)
    marker = " *" if preferred else ""
    print(f"{name:<40} {est:>12.4f} {se:>12.4f} {pval:>10.4f} {n:>8}{marker}")

print("\n* indicates preferred specification")
print(f"\nStatistical significance: * p<0.05, ** p<0.01, *** p<0.001")

# Save key results to file
results = {
    'preferred_estimate': model7.params['ELIGIBLE_AFTER'],
    'preferred_se': model7.bse['ELIGIBLE_AFTER'],
    'preferred_ci_lower': model7.conf_int().loc['ELIGIBLE_AFTER', 0],
    'preferred_ci_upper': model7.conf_int().loc['ELIGIBLE_AFTER', 1],
    'preferred_pvalue': model7.pvalues['ELIGIBLE_AFTER'],
    'sample_size': int(model7.nobs),
    'r_squared': model7.rsquared
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("Analysis complete. Results saved to analysis_results.json")
print("=" * 70)

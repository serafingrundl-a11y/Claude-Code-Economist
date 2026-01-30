"""
DACA Replication Study - Analysis Code
Research Question: Effect of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born individuals in the US.

Approach: Difference-in-Differences (DiD)
- Treatment group: DACA-eligible individuals aged 26-30 in June 2012 (ELIGIBLE=1)
- Control group: Individuals aged 31-35 in June 2012 (ELIGIBLE=0)
- Pre-period: 2008-2011 (AFTER=0)
- Post-period: 2013-2016 (AFTER=1)
- Outcome: Full-time employment (FT=1 if working 35+ hours/week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================
# LOAD DATA
# =============================
print("="*60)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*60)

df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"\nTotal observations: {len(df)}")
print(f"Years covered: {sorted(df['YEAR'].unique())}")

# =============================
# SUMMARY STATISTICS
# =============================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

# Key variables
print("\n--- Sample Sizes by Treatment and Period ---")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Control (Age 31-35)', 'Treatment (Age 26-30)', 'Total']
crosstab.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
print(crosstab)

# Full-time employment rates by group
print("\n--- Full-Time Employment Rates by Group ---")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.columns = ['FT_Rate', 'StdDev', 'N']
ft_rates.index = pd.MultiIndex.from_tuples([
    ('Control (Age 31-35)', 'Pre-DACA'),
    ('Control (Age 31-35)', 'Post-DACA'),
    ('Treatment (Age 26-30)', 'Pre-DACA'),
    ('Treatment (Age 26-30)', 'Post-DACA')
])
print(ft_rates)

# Demographic characteristics by treatment group
print("\n--- Demographic Characteristics by Treatment Group ---")
demographics = ['SEX', 'AGE', 'MARST', 'NCHILD', 'FAMSIZE', 'YRSUSA1']

for var in demographics:
    if var in df.columns:
        print(f"\n{var} by ELIGIBLE status:")
        print(df.groupby('ELIGIBLE')[var].describe()[['mean', 'std', 'min', 'max']])

# Education distribution
print("\n--- Education Distribution (EDUC_RECODE) ---")
print(pd.crosstab(df['ELIGIBLE'], df['EDUC_RECODE'], normalize='index'))

# =============================
# SIMPLE DIFFERENCE-IN-DIFFERENCES
# =============================
print("\n" + "="*60)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES (UNWEIGHTED)")
print("="*60)

treat_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treat_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\nTreatment Group (Eligible, Age 26-30):")
print(f"  Pre-DACA (2008-2011):  {treat_before:.4f}")
print(f"  Post-DACA (2013-2016): {treat_after:.4f}")
print(f"  Change:                {treat_after - treat_before:+.4f}")

print(f"\nControl Group (Age 31-35):")
print(f"  Pre-DACA (2008-2011):  {control_before:.4f}")
print(f"  Post-DACA (2013-2016): {control_after:.4f}")
print(f"  Change:                {control_after - control_before:+.4f}")

did_simple = (treat_after - treat_before) - (control_after - control_before)
print(f"\nDifference-in-Differences Estimate: {did_simple:+.4f}")
print(f"  (Percentage points: {did_simple*100:+.2f} pp)")

# =============================
# REGRESSION-BASED DiD (OLS)
# =============================
print("\n" + "="*60)
print("REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("="*60)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"N: {int(model1.nobs)}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
# Recode SEX to binary (1=Male, 0=Female)
df['MALE'] = (df['SEX'] == 1).astype(int)

# Create married indicator (MARST: 1=married spouse present, 2=married spouse absent)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Use education recodes as dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 2).astype(int)  # HS degree
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 3).astype(int)  # Some college
df['EDUC_ASSOC'] = (df['EDUC_RECODE'] == 4).astype(int)  # Associate degree
df['EDUC_BA'] = (df['EDUC_RECODE'] == 5).astype(int)  # BA+

model2_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + MALE + MARRIED + NCHILD + YRSUSA1 + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA'
model2 = smf.ols(model2_formula, data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"N: {int(model2.nobs)}")

# Model 3: DiD with demographic controls and state fixed effects
print("\n--- Model 3: DiD with Demographics and State Fixed Effects ---")
# Create state dummies using C() notation
model3_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + MALE + MARRIED + NCHILD + YRSUSA1 + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA + C(STATEFIP)'
model3 = smf.ols(model3_formula, data=df).fit(cov_type='HC1')
# Print only main coefficients
main_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'NCHILD', 'YRSUSA1', 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_ASSOC', 'EDUC_BA']
for var in main_vars:
    if var in model3.params.index:
        print(f"{var:20s} coef={model3.params[var]:8.4f}  se={model3.bse[var]:8.4f}  p={model3.pvalues[var]:.4f}")
print(f"\nDiD Coefficient: {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"N: {int(model3.nobs)}, R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with demographics, state FE, and year FE
print("\n--- Model 4: DiD with Demographics, State FE, and Year FE ---")
model4_formula = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + MALE + MARRIED + NCHILD + YRSUSA1 + EDUC_HS + EDUC_SOMECOLL + EDUC_ASSOC + EDUC_BA + C(STATEFIP) + C(YEAR)'
model4 = smf.ols(model4_formula, data=df).fit(cov_type='HC1')
main_vars_m4 = ['Intercept', 'ELIGIBLE', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'NCHILD', 'YRSUSA1', 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_ASSOC', 'EDUC_BA']
for var in main_vars_m4:
    if var in model4.params.index:
        print(f"{var:20s} coef={model4.params[var]:8.4f}  se={model4.bse[var]:8.4f}  p={model4.pvalues[var]:.4f}")
print(f"\nDiD Coefficient: {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model4.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"N: {int(model4.nobs)}, R-squared: {model4.rsquared:.4f}")

# =============================
# WEIGHTED REGRESSION ANALYSIS
# =============================
print("\n" + "="*60)
print("WEIGHTED REGRESSION ANALYSIS (Using PERWT)")
print("="*60)

# Model 5: Basic DiD with person weights
print("\n--- Model 5: Weighted Basic DiD ---")
model5 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Model 6: Weighted DiD with full controls, state FE, year FE
print("\n--- Model 6: Weighted DiD with Full Controls (PREFERRED MODEL) ---")
model6 = smf.wls(model4_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
for var in main_vars_m4:
    if var in model6.params.index:
        print(f"{var:20s} coef={model6.params[var]:8.4f}  se={model6.bse[var]:8.4f}  p={model6.pvalues[var]:.4f}")
print(f"\nPREFERRED DiD ESTIMATE:")
print(f"DiD Coefficient: {model6.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error:  {model6.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"N: {int(model6.nobs)}, R-squared: {model6.rsquared:.4f}")

# =============================
# ROBUSTNESS CHECKS
# =============================
print("\n" + "="*60)
print("ROBUSTNESS CHECKS")
print("="*60)

# By sex
print("\n--- Subgroup Analysis by Sex ---")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
    print(f"{sex_label}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}), N = {int(model_sub.nobs)}")

# By education level
print("\n--- Subgroup Analysis by Education ---")
for educ_val, educ_label in [(1, 'Less than HS'), (2, 'HS Degree'), (3, 'Some College'), (4, 'Associate'), (5, 'BA+')]:
    df_sub = df[df['EDUC_RECODE'] == educ_val]
    if len(df_sub) > 100:
        model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_sub, weights=df_sub['PERWT']).fit(cov_type='HC1')
        print(f"{educ_label:15s}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE: {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}), N = {int(model_sub.nobs)}")

# =============================
# EVENT STUDY (Year-by-Year Effects)
# =============================
print("\n" + "="*60)
print("EVENT STUDY: YEAR-BY-YEAR TREATMENT EFFECTS")
print("="*60)

# Create year dummies and interactions with ELIGIBLE
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_X_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

event_formula = ('FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + '
                 'ELIGIBLE_X_2009 + ELIGIBLE_X_2010 + ELIGIBLE_X_2011 + ELIGIBLE_X_2013 + ELIGIBLE_X_2014 + ELIGIBLE_X_2015 + ELIGIBLE_X_2016')

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("\nYear-specific treatment effects (relative to 2008):")
print(f"{'Year':<10}{'Coefficient':<15}{'Std Error':<15}{'95% CI':<25}{'P-value':<10}")
print("-"*75)
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_X_{yr}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci_low = model_event.conf_int().loc[var, 0]
    ci_high = model_event.conf_int().loc[var, 1]
    pval = model_event.pvalues[var]
    print(f"{yr:<10}{coef:+.4f}{'':>8}{se:.4f}{'':>8}[{ci_low:+.4f}, {ci_high:+.4f}]{'':>5}{pval:.4f}")

# =============================
# PARALLEL TRENDS TEST
# =============================
print("\n" + "="*60)
print("PARALLEL TRENDS TEST (Pre-Treatment Period Only)")
print("="*60)

df_pre = df[df['AFTER'] == 0].copy()
df_pre['TREND'] = df_pre['YEAR'] - 2008
df_pre['ELIGIBLE_X_TREND'] = df_pre['ELIGIBLE'] * df_pre['TREND']

model_trend = smf.wls('FT ~ ELIGIBLE + TREND + ELIGIBLE_X_TREND', data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print("\nDifferential trend test (ELIGIBLE x TREND):")
print(f"Coefficient: {model_trend.params['ELIGIBLE_X_TREND']:.4f}")
print(f"Standard Error: {model_trend.bse['ELIGIBLE_X_TREND']:.4f}")
print(f"P-value: {model_trend.pvalues['ELIGIBLE_X_TREND']:.4f}")
if model_trend.pvalues['ELIGIBLE_X_TREND'] > 0.05:
    print("=> Cannot reject parallel trends assumption (p > 0.05)")
else:
    print("=> Evidence against parallel trends assumption (p < 0.05)")

# =============================
# SUMMARY TABLE
# =============================
print("\n" + "="*60)
print("SUMMARY OF ALL MODELS")
print("="*60)

print(f"\n{'Model':<50}{'DiD Est':<12}{'SE':<10}{'95% CI':<25}{'N':<10}")
print("-"*107)

models = [
    ("1. Basic DiD (unweighted)", model1),
    ("2. DiD + Demographics (unweighted)", model2),
    ("3. DiD + Demographics + State FE (unweighted)", model3),
    ("4. DiD + Demographics + State FE + Year FE (unwtd)", model4),
    ("5. Basic DiD (weighted)", model5),
    ("6. Full model (weighted) - PREFERRED", model6),
]

for name, model in models:
    coef = model.params['ELIGIBLE_X_AFTER']
    se = model.bse['ELIGIBLE_X_AFTER']
    ci = model.conf_int().loc['ELIGIBLE_X_AFTER']
    n = int(model.nobs)
    print(f"{name:<50}{coef:+.4f}{'':>5}{se:.4f}{'':>3}[{ci[0]:+.4f}, {ci[1]:+.4f}]{'':>3}{n}")

# =============================
# SAVE RESULTS FOR REPORT
# =============================
print("\n" + "="*60)
print("EXPORTING RESULTS")
print("="*60)

# Save main results to CSV
results_dict = {
    'Model': [],
    'DiD_Estimate': [],
    'SE': [],
    'CI_Lower': [],
    'CI_Upper': [],
    'P_Value': [],
    'N': [],
    'R_squared': []
}

model_names = [
    "Basic DiD (unweighted)",
    "DiD + Demographics (unweighted)",
    "DiD + Demographics + State FE (unweighted)",
    "DiD + Demographics + State + Year FE (unweighted)",
    "Basic DiD (weighted)",
    "Full model (weighted) - PREFERRED"
]

for name, model in zip(model_names, [model1, model2, model3, model4, model5, model6]):
    results_dict['Model'].append(name)
    results_dict['DiD_Estimate'].append(model.params['ELIGIBLE_X_AFTER'])
    results_dict['SE'].append(model.bse['ELIGIBLE_X_AFTER'])
    results_dict['CI_Lower'].append(model.conf_int().loc['ELIGIBLE_X_AFTER', 0])
    results_dict['CI_Upper'].append(model.conf_int().loc['ELIGIBLE_X_AFTER', 1])
    results_dict['P_Value'].append(model.pvalues['ELIGIBLE_X_AFTER'])
    results_dict['N'].append(int(model.nobs))
    results_dict['R_squared'].append(model.rsquared)

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("Regression results saved to regression_results.csv")

# Save event study results
event_results = {'Year': [], 'Coefficient': [], 'SE': [], 'CI_Lower': [], 'CI_Upper': [], 'P_Value': []}
for yr in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    var = f'ELIGIBLE_X_{yr}'
    event_results['Year'].append(yr)
    event_results['Coefficient'].append(model_event.params[var])
    event_results['SE'].append(model_event.bse[var])
    event_results['CI_Lower'].append(model_event.conf_int().loc[var, 0])
    event_results['CI_Upper'].append(model_event.conf_int().loc[var, 1])
    event_results['P_Value'].append(model_event.pvalues[var])

event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save summary statistics
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'MALE': 'mean',
    'AGE': 'mean',
    'MARRIED': 'mean',
    'NCHILD': 'mean',
    'YRSUSA1': 'mean'
}).round(4)
summary_stats.to_csv('summary_statistics.csv')
print("Summary statistics saved to summary_statistics.csv")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"\nPREFERRED ESTIMATE (Weighted DiD with full controls):")
print(f"  Effect of DACA eligibility on full-time employment: {model6.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"  Standard Error: {model6.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"  95% Confidence Interval: [{model6.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"  Sample Size: {int(model6.nobs)}")
print(f"\nInterpretation: DACA eligibility is associated with a {model6.params['ELIGIBLE_X_AFTER']*100:.2f} percentage point")
print(f"{'increase' if model6.params['ELIGIBLE_X_AFTER'] > 0 else 'decrease'} in the probability of full-time employment.")

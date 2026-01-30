"""
DACA Replication Study - Analysis Script
Replication 87

Research Question: What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born people in the US?

Study Design: Difference-in-Differences
- Treatment: DACA-eligible ages 26-30 at June 15, 2012
- Control: DACA-ineligible ages 31-35 (age restriction only)
- Pre-period: 2008-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
print("="*80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*80)

data_path = r"C:\Users\seraf\DACA Results Task 3\replication_87\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df):,} observations")
print(f"Years in data: {sorted(df['YEAR'].unique())}")
print(f"Columns: {len(df.columns)}")

# ============================================================================
# DATA VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("DATA VERIFICATION")
print("="*80)

# Key variables check
print("\n--- Key Variables ---")
print(f"FT (Full-time employment): values = {sorted(df['FT'].unique())}")
print(f"ELIGIBLE (Treatment group): values = {sorted(df['ELIGIBLE'].unique())}")
print(f"AFTER (Post-treatment): values = {sorted(df['AFTER'].unique())}")

# Verify ELIGIBLE coding
print("\n--- ELIGIBLE by AGE_IN_JUNE_2012 ---")
age_elig = df.groupby(['AGE_IN_JUNE_2012', 'ELIGIBLE']).size().unstack(fill_value=0)
print(age_elig)

# Create interaction term
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# ============================================================================
# DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Sample sizes by group and period
print("\n--- Sample Sizes ---")
sample_table = df.groupby(['ELIGIBLE', 'AFTER']).agg(
    N=('FT', 'count'),
    FT_mean=('FT', 'mean'),
    FT_std=('FT', 'std')
).round(4)
print(sample_table)

# Calculate unweighted group means
print("\n--- Unweighted Group Means (FT) ---")
group_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
group_means.index = ['Control (31-35)', 'Treatment (26-30)']
group_means.columns = ['Pre (2008-11)', 'Post (2013-16)']
print(group_means.round(4))

# Calculate raw DiD
raw_did = (group_means.loc['Treatment (26-30)', 'Post (2013-16)'] -
           group_means.loc['Treatment (26-30)', 'Pre (2008-11)']) - \
          (group_means.loc['Control (31-35)', 'Post (2013-16)'] -
           group_means.loc['Control (31-35)', 'Pre (2008-11)'])
print(f"\nRaw DiD estimate (unweighted): {raw_did:.4f}")

# Weighted group means
print("\n--- Weighted Group Means (FT, using PERWT) ---")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

weighted_group_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean).unstack()
weighted_group_means.index = ['Control (31-35)', 'Treatment (26-30)']
weighted_group_means.columns = ['Pre (2008-11)', 'Post (2013-16)']
print(weighted_group_means.round(4))

# Weighted DiD
weighted_did = (weighted_group_means.loc['Treatment (26-30)', 'Post (2013-16)'] -
                weighted_group_means.loc['Treatment (26-30)', 'Pre (2008-11)']) - \
               (weighted_group_means.loc['Control (31-35)', 'Post (2013-16)'] -
                weighted_group_means.loc['Control (31-35)', 'Pre (2008-11)'])
print(f"\nWeighted DiD estimate: {weighted_did:.4f}")

# Year-by-year means
print("\n--- Full-Time Employment Rate by Year and Group ---")
yearly = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly.columns = ['Control (31-35)', 'Treatment (26-30)']
print(yearly.round(4))

# Demographic characteristics
print("\n--- Baseline Demographics (Pre-period, 2008-2011) ---")
pre_df = df[df['AFTER'] == 0]

# SEX: 1=Male, 2=Female per IPUMS
df['FEMALE'] = (df['SEX'] == 2).astype(int)
pre_df_with_female = pre_df.copy()
pre_df_with_female['FEMALE'] = (pre_df_with_female['SEX'] == 2).astype(int)

demo_vars = ['AGE', 'FEMALE', 'FAMSIZE', 'NCHILD']
for var in demo_vars:
    if var in pre_df_with_female.columns:
        by_group = pre_df_with_female.groupby('ELIGIBLE')[var].agg(['mean', 'std'])
        by_group.index = ['Control', 'Treatment']
        print(f"\n{var}:")
        print(by_group.round(3))

# Education distribution
print("\n--- Education Distribution (Pre-period) ---")
if 'EDUC_RECODE' in pre_df.columns:
    educ_dist = pd.crosstab(pre_df['EDUC_RECODE'], pre_df['ELIGIBLE'], normalize='columns')
    educ_dist.columns = ['Control', 'Treatment']
    print(educ_dist.round(3))

# Employment status
print("\n--- Employment Status Distribution (Pre-period) ---")
emp_dist = pd.crosstab(pre_df['EMPSTAT'], pre_df['ELIGIBLE'], normalize='columns')
emp_dist.columns = ['Control', 'Treatment']
print(emp_dist.round(3))

# ============================================================================
# MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# ============================================================================
print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Create female variable for full dataset
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Model 1: Basic DiD (unweighted)
print("\n--- Model 1: Basic DiD (Unweighted) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Coefficient: {model1.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 2: Basic DiD with survey weights (WLS)
print("\n--- Model 2: Basic DiD (Weighted with PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Coefficient: {model2.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 3: DiD with Year Fixed Effects (unweighted)
print("\n--- Model 3: DiD with Year Fixed Effects (Unweighted) ---")
df['YEAR_factor'] = df['YEAR'].astype(str)
model3 = smf.ols('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_x_AFTER', data=df).fit()
print(f"DiD Coefficient: {model3.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 4: DiD with Year FE (weighted)
print("\n--- Model 4: DiD with Year Fixed Effects (Weighted) ---")
model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR_factor) + ELIGIBLE_x_AFTER', data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model4.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 5: DiD with covariates (unweighted)
print("\n--- Model 5: DiD with Covariates (Unweighted) ---")
# Add controls: age, sex, marital status, education, number of children, state
# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_2YR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Marital status: 1=Married, spouse present
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Model with individual-level covariates
model5_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA'
model5 = smf.ols(model5_formula, data=df).fit()
print(f"DiD Coefficient: {model5.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 6: DiD with covariates (weighted)
print("\n--- Model 6: DiD with Covariates (Weighted) - PREFERRED SPECIFICATION ---")
model6 = smf.wls(model5_formula, data=df, weights=df['PERWT']).fit()
print(model6.summary().tables[1])
print(f"\nDiD Coefficient: {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {int(model6.nobs):,}")

# Model 7: DiD with covariates and state fixed effects (weighted)
print("\n--- Model 7: DiD with Covariates and State FE (Weighted) ---")
model7_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA + C(STATEFIP)'
model7 = smf.wls(model7_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model7.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model7.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model7.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# Model 8: DiD with covariates, year FE, and state FE (weighted)
print("\n--- Model 8: DiD with Covariates, Year FE, and State FE (Weighted) ---")
model8_formula = 'FT ~ ELIGIBLE + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA + C(YEAR_factor) + C(STATEFIP)'
model8 = smf.wls(model8_formula, data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient: {model8.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model8.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model8.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model8.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model8.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# ============================================================================
# ROBUST STANDARD ERRORS
# ============================================================================
print("\n" + "="*80)
print("HETEROSKEDASTICITY-ROBUST STANDARD ERRORS")
print("="*80)

# Model 6 with robust SE (HC1)
print("\n--- Model 6 with Robust (HC1) Standard Errors ---")
model6_robust = smf.wls(model5_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient: {model6_robust.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Robust SE: {model6_robust.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% CI: [{model6_robust.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6_robust.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model6_robust.pvalues['ELIGIBLE_x_AFTER']:.4f}")

# ============================================================================
# PARALLEL TRENDS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PARALLEL TRENDS ANALYSIS")
print("="*80)

# Event study specification
print("\n--- Event Study: Year-by-Year Treatment Effects ---")
# Create year indicators
for yr in df['YEAR'].unique():
    df[f'YEAR_{yr}'] = (df['YEAR'] == yr).astype(int)
    df[f'ELIG_x_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

# Omit 2011 as reference year
event_study_terms = ' + '.join([f'ELIG_x_{yr}' for yr in sorted(df['YEAR'].unique()) if yr != 2011])
event_formula = f'FT ~ ELIGIBLE + C(YEAR_factor) + {event_study_terms}'
event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit()

print("Year-specific treatment effects (reference year: 2011):")
event_results = []
for yr in sorted(df['YEAR'].unique()):
    if yr != 2011:
        coef = event_model.params[f'ELIG_x_{yr}']
        se = event_model.bse[f'ELIG_x_{yr}']
        pval = event_model.pvalues[f'ELIG_x_{yr}']
        event_results.append({'Year': yr, 'Coefficient': coef, 'SE': se, 'p-value': pval})
        period = "Pre" if yr < 2012 else "Post"
        print(f"  {yr} ({period}): {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

event_df = pd.DataFrame(event_results)

# Test for pre-trends
print("\n--- Pre-trend Test (Joint F-test for 2008, 2009, 2010) ---")
pre_years = [2008, 2009, 2010]
pre_hypotheses = ' = '.join([f'ELIG_x_{yr}' for yr in pre_years]) + ' = 0'
try:
    f_test = event_model.f_test(pre_hypotheses)
    print(f"F-statistic: {f_test.fvalue[0][0]:.4f}")
    print(f"P-value: {f_test.pvalue:.4f}")
except:
    print("Joint test not available, testing individually:")
    for yr in pre_years:
        print(f"  {yr}: coef={event_model.params[f'ELIG_x_{yr}']:.4f}, p={event_model.pvalues[f'ELIG_x_{yr}']:.4f}")

# ============================================================================
# SUBGROUP ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SUBGROUP ANALYSIS")
print("="*80)

# By sex
print("\n--- By Sex ---")
for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    sub_model = smf.wls(model5_formula, data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"  {sex_label}: DiD = {sub_model.params['ELIGIBLE_x_AFTER']:.4f} (SE: {sub_model.bse['ELIGIBLE_x_AFTER']:.4f}, p={sub_model.pvalues['ELIGIBLE_x_AFTER']:.4f}), N={len(df_sub):,}")

# By education
print("\n--- By Education Level ---")
for educ_level in ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']:
    df_sub = df[df['EDUC_RECODE'] == educ_level]
    if len(df_sub) > 100:
        # Simpler formula without education controls for education subgroups
        simple_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD'
        sub_model = smf.wls(simple_formula, data=df_sub, weights=df_sub['PERWT']).fit()
        print(f"  {educ_level}: DiD = {sub_model.params['ELIGIBLE_x_AFTER']:.4f} (SE: {sub_model.bse['ELIGIBLE_x_AFTER']:.4f}, p={sub_model.pvalues['ELIGIBLE_x_AFTER']:.4f}), N={len(df_sub):,}")

# By marital status
print("\n--- By Marital Status ---")
for mar_val, mar_label in [(1, 'Married'), (0, 'Not Married')]:
    df_sub = df[df['MARRIED'] == mar_val]
    simple_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA'
    sub_model = smf.wls(simple_formula, data=df_sub, weights=df_sub['PERWT']).fit()
    print(f"  {mar_label}: DiD = {sub_model.params['ELIGIBLE_x_AFTER']:.4f} (SE: {sub_model.bse['ELIGIBLE_x_AFTER']:.4f}, p={sub_model.pvalues['ELIGIBLE_x_AFTER']:.4f}), N={len(df_sub):,}")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Narrower age bandwidth (27-29 vs 32-34)
print("\n--- Narrower Age Band (27-29 vs 32-34) ---")
df_narrow = df[(df['AGE_IN_JUNE_2012'] >= 27) & (df['AGE_IN_JUNE_2012'] <= 29) |
               (df['AGE_IN_JUNE_2012'] >= 32) & (df['AGE_IN_JUNE_2012'] <= 34)]
df_narrow['ELIGIBLE_narrow'] = ((df_narrow['AGE_IN_JUNE_2012'] >= 27) &
                                 (df_narrow['AGE_IN_JUNE_2012'] <= 29)).astype(int)
df_narrow['ELIG_x_AFTER_narrow'] = df_narrow['ELIGIBLE_narrow'] * df_narrow['AFTER']

narrow_formula = 'FT ~ ELIGIBLE_narrow + AFTER + ELIG_x_AFTER_narrow + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA'
narrow_model = smf.wls(narrow_formula, data=df_narrow, weights=df_narrow['PERWT']).fit()
print(f"DiD Coefficient: {narrow_model.params['ELIG_x_AFTER_narrow']:.4f}")
print(f"SE: {narrow_model.bse['ELIG_x_AFTER_narrow']:.4f}")
print(f"95% CI: [{narrow_model.conf_int().loc['ELIG_x_AFTER_narrow', 0]:.4f}, {narrow_model.conf_int().loc['ELIG_x_AFTER_narrow', 1]:.4f}]")
print(f"N: {len(df_narrow):,}")

# Placebo test: use 2010 as fake treatment year
print("\n--- Placebo Test (Fake Treatment in 2010) ---")
df_placebo = df[df['YEAR'].isin([2008, 2009, 2010, 2011])]
df_placebo['AFTER_PLACEBO'] = (df_placebo['YEAR'] >= 2010).astype(int)
df_placebo['ELIG_x_AFTER_PLACEBO'] = df_placebo['ELIGIBLE'] * df_placebo['AFTER_PLACEBO']

placebo_formula = 'FT ~ ELIGIBLE + AFTER_PLACEBO + ELIG_x_AFTER_PLACEBO + AGE + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA'
placebo_model = smf.wls(placebo_formula, data=df_placebo, weights=df_placebo['PERWT']).fit()
print(f"Placebo DiD Coefficient: {placebo_model.params['ELIG_x_AFTER_PLACEBO']:.4f}")
print(f"SE: {placebo_model.bse['ELIG_x_AFTER_PLACEBO']:.4f}")
print(f"P-value: {placebo_model.pvalues['ELIG_x_AFTER_PLACEBO']:.4f}")
print("(A non-significant result supports the parallel trends assumption)")

# Controlling for state-level policies
print("\n--- Model with State-Level Policy Controls ---")
policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY', 'SECURECOMMUNITIES']
available_policies = [p for p in policy_vars if p in df.columns and df[p].notna().any()]
if available_policies:
    policy_formula = model5_formula + ' + ' + ' + '.join(available_policies)
    policy_model = smf.wls(policy_formula, data=df, weights=df['PERWT']).fit()
    print(f"DiD Coefficient: {policy_model.params['ELIGIBLE_x_AFTER']:.4f}")
    print(f"SE: {policy_model.bse['ELIGIBLE_x_AFTER']:.4f}")
    print(f"95% CI: [{policy_model.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {policy_model.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")

# ============================================================================
# SUMMARY STATISTICS FOR OUTPUT
# ============================================================================
print("\n" + "="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)

print("\n*** PREFERRED SPECIFICATION (Model 6: Weighted DiD with Covariates) ***")
print(f"Effect Size (DiD Coefficient): {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_x_AFTER']:.4f}")
print(f"Sample Size: {int(model6.nobs):,}")
print(f"R-squared: {model6.rsquared:.4f}")

# Interpretation
effect = model6.params['ELIGIBLE_x_AFTER']
baseline_treatment_pre = weighted_group_means.loc['Treatment (26-30)', 'Pre (2008-11)']
pct_change = (effect / baseline_treatment_pre) * 100

print(f"\nInterpretation:")
print(f"DACA eligibility increased full-time employment probability by {effect:.3f} percentage points")
print(f"(or {effect*100:.1f} percentage points on a 0-100 scale)")
print(f"Relative to baseline treatment group FT rate of {baseline_treatment_pre:.3f}, this is a {pct_change:.1f}% increase")

# ============================================================================
# SAVE RESULTS
# ============================================================================
# Create summary table
results_summary = pd.DataFrame({
    'Model': ['1. Basic DiD (Unweighted)', '2. Basic DiD (Weighted)',
              '3. Year FE (Unweighted)', '4. Year FE (Weighted)',
              '5. Covariates (Unweighted)', '6. Covariates (Weighted)*',
              '7. Covariates + State FE', '8. Full Model'],
    'DiD_Coef': [model1.params['ELIGIBLE_x_AFTER'], model2.params['ELIGIBLE_x_AFTER'],
                 model3.params['ELIGIBLE_x_AFTER'], model4.params['ELIGIBLE_x_AFTER'],
                 model5.params['ELIGIBLE_x_AFTER'], model6.params['ELIGIBLE_x_AFTER'],
                 model7.params['ELIGIBLE_x_AFTER'], model8.params['ELIGIBLE_x_AFTER']],
    'SE': [model1.bse['ELIGIBLE_x_AFTER'], model2.bse['ELIGIBLE_x_AFTER'],
           model3.bse['ELIGIBLE_x_AFTER'], model4.bse['ELIGIBLE_x_AFTER'],
           model5.bse['ELIGIBLE_x_AFTER'], model6.bse['ELIGIBLE_x_AFTER'],
           model7.bse['ELIGIBLE_x_AFTER'], model8.bse['ELIGIBLE_x_AFTER']],
    'P_value': [model1.pvalues['ELIGIBLE_x_AFTER'], model2.pvalues['ELIGIBLE_x_AFTER'],
                model3.pvalues['ELIGIBLE_x_AFTER'], model4.pvalues['ELIGIBLE_x_AFTER'],
                model5.pvalues['ELIGIBLE_x_AFTER'], model6.pvalues['ELIGIBLE_x_AFTER'],
                model7.pvalues['ELIGIBLE_x_AFTER'], model8.pvalues['ELIGIBLE_x_AFTER']]
})
results_summary['CI_lower'] = results_summary['DiD_Coef'] - 1.96 * results_summary['SE']
results_summary['CI_upper'] = results_summary['DiD_Coef'] + 1.96 * results_summary['SE']
print("\n" + "="*80)
print("ALL MODEL COMPARISON")
print("="*80)
print(results_summary.to_string(index=False))

# Save to CSV
results_summary.to_csv(r"C:\Users\seraf\DACA Results Task 3\replication_87\results_summary.csv", index=False)

# ============================================================================
# CREATE FIGURES
# ============================================================================
print("\n" + "="*80)
print("CREATING FIGURES")
print("="*80)

# Figure 1: Parallel Trends
fig1, ax1 = plt.subplots(figsize=(10, 6))
years = sorted(df['YEAR'].unique())
treatment_means = df[df['ELIGIBLE']==1].groupby('YEAR')['FT'].mean()
control_means = df[df['ELIGIBLE']==0].groupby('YEAR')['FT'].mean()

ax1.plot(years, treatment_means[years], 'b-o', label='Treatment (26-30)', linewidth=2, markersize=8)
ax1.plot(years, control_means[years], 'r--s', label='Control (31-35)', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
ax1.fill_between([2012, 2016.5], 0, 1, alpha=0.1, color='green', label='Post-treatment')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xlim([2007.5, 2016.5])
ax1.set_ylim([0.30, 0.60])
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_87\figure1_parallel_trends.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure1_parallel_trends.png")

# Figure 2: Event Study
fig2, ax2 = plt.subplots(figsize=(10, 6))
event_years = [yr for yr in sorted(df['YEAR'].unique()) if yr != 2011]
event_coefs = [event_model.params[f'ELIG_x_{yr}'] for yr in event_years]
event_ses = [event_model.bse[f'ELIG_x_{yr}'] for yr in event_years]
event_ci_low = [c - 1.96*s for c, s in zip(event_coefs, event_ses)]
event_ci_high = [c + 1.96*s for c, s in zip(event_coefs, event_ses)]

ax2.errorbar(event_years, event_coefs, yerr=[np.array(event_coefs)-np.array(event_ci_low),
                                               np.array(event_ci_high)-np.array(event_coefs)],
             fmt='o', color='blue', capsize=5, capthick=2, markersize=8, linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2011.5, color='gray', linestyle=':', linewidth=2, label='Reference Year (2011)')
ax2.fill_between([2011.5, 2016.5], -0.1, 0.15, alpha=0.1, color='green', label='Post-treatment')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Treatment Effect Relative to 2011', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects on Full-Time Employment', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim([2007.5, 2016.5])
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_87\figure2_event_study.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure2_event_study.png")

# Figure 3: Model Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))
models_plot = results_summary['Model'].tolist()
coefs_plot = results_summary['DiD_Coef'].tolist()
ci_low_plot = results_summary['CI_lower'].tolist()
ci_high_plot = results_summary['CI_upper'].tolist()

y_pos = np.arange(len(models_plot))
ax3.barh(y_pos, coefs_plot, xerr=[np.array(coefs_plot)-np.array(ci_low_plot),
                                   np.array(ci_high_plot)-np.array(coefs_plot)],
         align='center', alpha=0.7, color='steelblue', capsize=5)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(models_plot)
ax3.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax3.set_title('Comparison of DiD Estimates Across Specifications', fontsize=14)
ax3.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_87\figure3_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure3_model_comparison.png")

# Figure 4: Subgroup Analysis
fig4, ax4 = plt.subplots(figsize=(10, 6))
subgroups = ['Overall', 'Male', 'Female', 'Less than HS', 'High School', 'Some College+', 'Married', 'Not Married']
subgroup_effects = []
subgroup_ses = []

# Get subgroup results
subgroup_effects.append(model6.params['ELIGIBLE_x_AFTER'])
subgroup_ses.append(model6.bse['ELIGIBLE_x_AFTER'])

for sex_val in [1, 2]:
    df_sub = df[df['SEX'] == sex_val]
    sub_model = smf.wls(model5_formula, data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_effects.append(sub_model.params['ELIGIBLE_x_AFTER'])
    subgroup_ses.append(sub_model.bse['ELIGIBLE_x_AFTER'])

for educ_group, educ_vals in [('Less than HS', ['Less than High School']),
                               ('High School', ['High School Degree']),
                               ('Some College+', ['Some College', 'Two-Year Degree', 'BA+'])]:
    df_sub = df[df['EDUC_RECODE'].isin(educ_vals)]
    simple_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + MARRIED + NCHILD'
    sub_model = smf.wls(simple_formula, data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_effects.append(sub_model.params['ELIGIBLE_x_AFTER'])
    subgroup_ses.append(sub_model.bse['ELIGIBLE_x_AFTER'])

for mar_val in [1, 0]:
    df_sub = df[df['MARRIED'] == mar_val]
    simple_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER + AGE + FEMALE + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_2YR + EDUC_BA'
    sub_model = smf.wls(simple_formula, data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_effects.append(sub_model.params['ELIGIBLE_x_AFTER'])
    subgroup_ses.append(sub_model.bse['ELIGIBLE_x_AFTER'])

y_pos = np.arange(len(subgroups))
ci_low = [e - 1.96*s for e, s in zip(subgroup_effects, subgroup_ses)]
ci_high = [e + 1.96*s for e, s in zip(subgroup_effects, subgroup_ses)]

ax4.barh(y_pos, subgroup_effects, xerr=[np.array(subgroup_effects)-np.array(ci_low),
                                         np.array(ci_high)-np.array(subgroup_effects)],
         align='center', alpha=0.7, color='steelblue', capsize=5)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(subgroups)
ax4.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax4.set_title('Subgroup Analysis: Heterogeneity in Treatment Effects', fontsize=14)
ax4.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_87\figure4_subgroups.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure4_subgroups.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nPreferred estimate (Model 6):")
print(f"  Effect: {model6.params['ELIGIBLE_x_AFTER']:.4f}")
print(f"  SE: {model6.bse['ELIGIBLE_x_AFTER']:.4f}")
print(f"  95% CI: [{model6.conf_int().loc['ELIGIBLE_x_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_x_AFTER', 1]:.4f}]")
print(f"  N: {int(model6.nobs):,}")

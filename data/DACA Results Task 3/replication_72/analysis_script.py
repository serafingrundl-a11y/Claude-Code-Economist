"""
DACA Replication Study - Difference-in-Differences Analysis
============================================================
Research Question: What was the causal impact of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican Mexican-born individuals in the United States?

Treatment Group: Individuals aged 26-30 at the time of DACA implementation (June 2012)
Comparison Group: Individuals aged 31-35 at the time of DACA implementation (June 2012)
Outcome: Full-time employment (FT = 1 if usually working 35+ hours per week)
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
pd.set_option('display.width', 200)

print("="*80)
print("DACA ELIGIBILITY AND FULL-TIME EMPLOYMENT: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("="*80)
print()

# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
print("="*80)
print("SECTION 1: DATA LOADING AND PREPARATION")
print("="*80)
print()

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"Total observations: {len(df):,}")
print(f"Years in dataset: {sorted(df['YEAR'].unique())}")
print(f"Note: 2012 is excluded as treatment timing is ambiguous")
print()

# Key variables
print("Key Variables:")
print(f"  - FT (Full-time employment): 1 if working 35+ hours/week, 0 otherwise")
print(f"  - ELIGIBLE: 1 for treatment group (age 26-30 in June 2012), 0 for comparison (age 31-35)")
print(f"  - AFTER: 1 for post-DACA years (2013-2016), 0 for pre-DACA years (2008-2011)")
print()

# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================
print("="*80)
print("SECTION 2: DESCRIPTIVE STATISTICS")
print("="*80)
print()

# Sample sizes by group and period
print("Sample Sizes by Treatment Group and Period:")
print("-" * 50)
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
crosstab.index = ['Comparison (31-35)', 'Treated (26-30)', 'Total']
crosstab.columns = ['Pre-DACA', 'Post-DACA', 'Total']
print(crosstab)
print()

# FT rates by group and period
print("Full-Time Employment Rates by Group and Period:")
print("-" * 50)
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates = ft_rates.reset_index()
ft_rates.columns = ['ELIGIBLE', 'AFTER', 'FT_Rate', 'Std_Dev', 'N']
print(ft_rates.to_string(index=False))
print()

# Summary statistics for covariates
print("Summary Statistics for Continuous Variables:")
print("-" * 50)
continuous_vars = ['AGE', 'UHRSWORK', 'FAMSIZE', 'NCHILD', 'YRSUSA1']
for var in continuous_vars:
    if var in df.columns:
        print(f"{var}: Mean = {df[var].mean():.2f}, SD = {df[var].std():.2f}, N = {df[var].notna().sum()}")
print()

# Categorical variable frequencies
print("Frequencies for Key Categorical Variables:")
print("-" * 50)
print(f"\nSEX (1=Male, 2=Female):")
print(df['SEX'].value_counts().sort_index())
print(f"\nMARST (Marital Status):")
print(df['MARST'].value_counts().sort_index())
print(f"\nEDUC_RECODE:")
if 'EDUC_RECODE' in df.columns:
    print(df['EDUC_RECODE'].value_counts())
print()

# =============================================================================
# 3. BASIC DIFFERENCE-IN-DIFFERENCES CALCULATION
# =============================================================================
print("="*80)
print("SECTION 3: BASIC DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("="*80)
print()

# Calculate means for each group-period cell
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
means.index = ['Comparison (31-35)', 'Treated (26-30)']
means.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
means['Change'] = means['Post-DACA (2013-2016)'] - means['Pre-DACA (2008-2011)']

print("Full-Time Employment Rates:")
print("-" * 60)
print(means.round(4))
print()

# DiD estimate
did_estimate = means.loc['Treated (26-30)', 'Change'] - means.loc['Comparison (31-35)', 'Change']
print(f"Difference-in-Differences Estimate: {did_estimate:.4f}")
print(f"Interpretation: DACA eligibility increased full-time employment by {did_estimate*100:.2f} percentage points")
print()

# =============================================================================
# 4. REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES
# =============================================================================
print("="*80)
print("SECTION 4: REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("="*80)
print()

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("Model 1: Basic DiD Regression (No Controls)")
print("-" * 60)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()
print(model1.summary().tables[1])
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 2: DiD with demographic controls
print("Model 2: DiD with Demographic Controls")
print("-" * 60)

# Prepare control variables
# SEX: 1=Male, 2=Female -> create Female dummy
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Marital status: Create married dummy (MARST 1 or 2 = married)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_AA'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
# Reference: Less than High School

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA', data=df).fit()
print(model2.summary().tables[1])
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 3: DiD with demographic + economic controls
print("Model 3: DiD with Demographic and Economic Controls")
print("-" * 60)

# Years in USA (handle missing)
df['YRSUSA1_clean'] = df['YRSUSA1'].replace([0], np.nan)
df['YRSUSA1_clean'] = df['YRSUSA1_clean'].fillna(df['YRSUSA1_clean'].median())

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA + FAMSIZE + YRSUSA1_clean', data=df).fit()
print(model3.summary().tables[1])
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 4: DiD with state fixed effects
print("Model 4: DiD with State Fixed Effects")
print("-" * 60)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

# Build formula with state fixed effects
state_cols = [col for col in state_dummies.columns]
controls = 'FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA'
state_fe = ' + '.join(state_cols)
formula4 = f'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + {controls} + {state_fe}'

model4 = smf.ols(formula4, data=df_with_states).fit()
print(f"Number of observations: {model4.nobs:.0f}")
print(f"R-squared: {model4.rsquared:.4f}")
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 5: DiD with year fixed effects
print("Model 5: DiD with Year Fixed Effects")
print("-" * 60)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)

year_cols = [col for col in year_dummies.columns]
year_fe = ' + '.join(year_cols)
formula5 = f'FT ~ ELIGIBLE + {controls} + ELIGIBLE_X_AFTER + {year_fe}'

model5 = smf.ols(formula5, data=df_with_years).fit()
print(f"Number of observations: {model5.nobs:.0f}")
print(f"R-squared: {model5.rsquared:.4f}")
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# Model 6: Full model with state and year FE
print("Model 6: DiD with State and Year Fixed Effects (Full Model)")
print("-" * 60)

df_full = pd.concat([df, state_dummies, year_dummies], axis=1)
formula6 = f'FT ~ ELIGIBLE + {controls} + ELIGIBLE_X_AFTER + {state_fe} + {year_fe}'

model6 = smf.ols(formula6, data=df_full).fit()
print(f"Number of observations: {model6.nobs:.0f}")
print(f"R-squared: {model6.rsquared:.4f}")
print()
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model6.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# =============================================================================
# 5. ROBUST STANDARD ERRORS
# =============================================================================
print("="*80)
print("SECTION 5: HETEROSKEDASTICITY-ROBUST STANDARD ERRORS")
print("="*80)
print()

# Re-estimate preferred model with robust standard errors
print("Model with Robust (HC1) Standard Errors:")
print("-" * 60)
model_robust = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + EDUC_HS + EDUC_SOMECOLL + EDUC_AA + EDUC_BA', data=df).fit(cov_type='HC1')
print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model_robust.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model_robust.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model_robust.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print()

# =============================================================================
# 6. SUBGROUP ANALYSIS
# =============================================================================
print("="*80)
print("SECTION 6: SUBGROUP ANALYSIS")
print("="*80)
print()

# By sex
print("DiD Estimates by Sex:")
print("-" * 50)
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_sub).fit()
    print(f"{sex_name}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}), N = {len(df_sub)}")
print()

# By education
print("DiD Estimates by Education Level:")
print("-" * 50)
for educ in df['EDUC_RECODE'].unique():
    if pd.notna(educ):
        df_sub = df[df['EDUC_RECODE'] == educ]
        if len(df_sub) > 100:  # Only if sufficient sample size
            model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_sub).fit()
            print(f"{educ}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}), N = {len(df_sub)}")
print()

# By marital status
print("DiD Estimates by Marital Status:")
print("-" * 50)
for married_val, married_name in [(1, 'Married'), (0, 'Not Married')]:
    df_sub = df[df['MARRIED'] == married_val]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_sub).fit()
    print(f"{married_name}: DiD = {model_sub.params['ELIGIBLE_X_AFTER']:.4f} (SE = {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}), N = {len(df_sub)}")
print()

# =============================================================================
# 7. PARALLEL TRENDS CHECK (PRE-TREATMENT)
# =============================================================================
print("="*80)
print("SECTION 7: PARALLEL TRENDS ANALYSIS")
print("="*80)
print()

# FT rates by year and treatment group
print("Full-Time Employment Rates by Year and Group:")
print("-" * 60)
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_rates.columns = ['Comparison (31-35)', 'Treated (26-30)']
yearly_rates['Difference'] = yearly_rates['Treated (26-30)'] - yearly_rates['Comparison (31-35)']
print(yearly_rates.round(4))
print()

# Pre-trend test: regress difference on year for pre-period
print("Pre-Treatment Trend Test:")
print("-" * 50)
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_CENTERED'] = pre_data['YEAR'] - 2008
model_pretrend = smf.ols('FT ~ ELIGIBLE + YEAR_CENTERED + ELIGIBLE*YEAR_CENTERED', data=pre_data).fit()
print(f"Interaction (ELIGIBLE x YEAR): {model_pretrend.params['ELIGIBLE:YEAR_CENTERED']:.4f}")
print(f"Standard Error: {model_pretrend.bse['ELIGIBLE:YEAR_CENTERED']:.4f}")
print(f"p-value: {model_pretrend.pvalues['ELIGIBLE:YEAR_CENTERED']:.4f}")
print()
if model_pretrend.pvalues['ELIGIBLE:YEAR_CENTERED'] > 0.05:
    print("Result: No significant differential pre-trend (supports parallel trends assumption)")
else:
    print("Result: Significant differential pre-trend detected (potential violation of parallel trends)")
print()

# =============================================================================
# 8. EVENT STUDY ANALYSIS
# =============================================================================
print("="*80)
print("SECTION 8: EVENT STUDY ANALYSIS")
print("="*80)
print()

# Create year dummies interacted with treatment
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions with ELIGIBLE (omit 2011 as reference)
df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

model_event = smf.ols('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016', data=df).fit()

print("Event Study Coefficients (Reference Year: 2011):")
print("-" * 60)
event_vars = ['ELIG_2008', 'ELIG_2009', 'ELIG_2010', 'ELIG_2013', 'ELIG_2014', 'ELIG_2015', 'ELIG_2016']
event_results = pd.DataFrame({
    'Year': [2008, 2009, 2010, 2013, 2014, 2015, 2016],
    'Coefficient': [model_event.params[v] for v in event_vars],
    'Std_Error': [model_event.bse[v] for v in event_vars],
    'p_value': [model_event.pvalues[v] for v in event_vars]
})
event_results['CI_Lower'] = event_results['Coefficient'] - 1.96 * event_results['Std_Error']
event_results['CI_Upper'] = event_results['Coefficient'] + 1.96 * event_results['Std_Error']
print(event_results.to_string(index=False))
print()

# =============================================================================
# 9. WEIGHTED ANALYSIS
# =============================================================================
print("="*80)
print("SECTION 9: WEIGHTED ANALYSIS (USING PERWT)")
print("="*80)
print()

# Check if PERWT exists
if 'PERWT' in df.columns:
    # Weighted regression using WLS
    import statsmodels.api as sm

    # Prepare data
    y = df['FT']
    X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER', 'FEMALE', 'MARRIED', 'NCHILD', 'EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_AA', 'EDUC_BA']]
    X = sm.add_constant(X)

    # WLS with PERWT
    model_weighted = sm.WLS(y, X, weights=df['PERWT']).fit()

    print("Weighted DiD Model (using PERWT):")
    print("-" * 60)
    print(f"DiD Estimate (ELIGIBLE_X_AFTER): {model_weighted.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"Standard Error: {model_weighted.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"95% CI: [{model_weighted.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_weighted.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
    print()
else:
    print("PERWT variable not found in dataset")
print()

# =============================================================================
# 10. SUMMARY OF RESULTS
# =============================================================================
print("="*80)
print("SECTION 10: SUMMARY OF RESULTS")
print("="*80)
print()

print("Summary Table of DiD Estimates:")
print("-" * 80)
print(f"{'Model':<45} {'Estimate':>10} {'Std Error':>10} {'p-value':>10}")
print("-" * 80)
print(f"{'(1) Basic DiD (no controls)':<45} {model1.params['ELIGIBLE_X_AFTER']:>10.4f} {model1.bse['ELIGIBLE_X_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(2) + Demographic controls':<45} {model2.params['ELIGIBLE_X_AFTER']:>10.4f} {model2.bse['ELIGIBLE_X_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(3) + Economic controls':<45} {model3.params['ELIGIBLE_X_AFTER']:>10.4f} {model3.bse['ELIGIBLE_X_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(4) + State fixed effects':<45} {model4.params['ELIGIBLE_X_AFTER']:>10.4f} {model4.bse['ELIGIBLE_X_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(5) + Year fixed effects':<45} {model5.params['ELIGIBLE_X_AFTER']:>10.4f} {model5.bse['ELIGIBLE_X_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(6) + State and year fixed effects':<45} {model6.params['ELIGIBLE_X_AFTER']:>10.4f} {model6.bse['ELIGIBLE_X_AFTER']:>10.4f} {model6.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(7) With robust standard errors':<45} {model_robust.params['ELIGIBLE_X_AFTER']:>10.4f} {model_robust.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_robust.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
if 'PERWT' in df.columns:
    print(f"{'(8) Weighted (PERWT)':<45} {model_weighted.params['ELIGIBLE_X_AFTER']:>10.4f} {model_weighted.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_weighted.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print("-" * 80)
print()

# Preferred estimate
print("="*80)
print("PREFERRED ESTIMATE")
print("="*80)
print()
print("Based on the analysis, the preferred specification is Model 2 (DiD with demographic controls)")
print("This model balances parsimony with appropriate control for confounders.")
print()
print(f"Effect Size: {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"p-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"Sample Size: {len(df):,}")
print()
print("Interpretation:")
print(f"DACA eligibility is associated with a {model2.params['ELIGIBLE_X_AFTER']*100:.2f} percentage point")
print(f"increase in the probability of full-time employment (p < 0.001).")
print()

# =============================================================================
# 11. EXPORT RESULTS FOR LATEX
# =============================================================================
print("="*80)
print("SECTION 11: EXPORTING RESULTS")
print("="*80)
print()

# Create results dictionary for export
results = {
    'basic_did': did_estimate,
    'model1_coef': model1.params['ELIGIBLE_X_AFTER'],
    'model1_se': model1.bse['ELIGIBLE_X_AFTER'],
    'model1_pval': model1.pvalues['ELIGIBLE_X_AFTER'],
    'model2_coef': model2.params['ELIGIBLE_X_AFTER'],
    'model2_se': model2.bse['ELIGIBLE_X_AFTER'],
    'model2_pval': model2.pvalues['ELIGIBLE_X_AFTER'],
    'model3_coef': model3.params['ELIGIBLE_X_AFTER'],
    'model3_se': model3.bse['ELIGIBLE_X_AFTER'],
    'model4_coef': model4.params['ELIGIBLE_X_AFTER'],
    'model4_se': model4.bse['ELIGIBLE_X_AFTER'],
    'model5_coef': model5.params['ELIGIBLE_X_AFTER'],
    'model5_se': model5.bse['ELIGIBLE_X_AFTER'],
    'model6_coef': model6.params['ELIGIBLE_X_AFTER'],
    'model6_se': model6.bse['ELIGIBLE_X_AFTER'],
    'robust_coef': model_robust.params['ELIGIBLE_X_AFTER'],
    'robust_se': model_robust.bse['ELIGIBLE_X_AFTER'],
    'n_obs': len(df),
    'n_treated': len(df[df['ELIGIBLE']==1]),
    'n_comparison': len(df[df['ELIGIBLE']==0]),
    'ft_rate_overall': df['FT'].mean()
}

# Save to CSV
results_df = pd.DataFrame([results])
results_df.to_csv('results_summary.csv', index=False)
print("Results saved to results_summary.csv")

# Save full model summary
with open('model_summary.txt', 'w') as f:
    f.write("PREFERRED MODEL (Model 2) - Full Summary\n")
    f.write("="*80 + "\n")
    f.write(str(model2.summary()))

print("Full model summary saved to model_summary.txt")
print()
print("Analysis complete!")

"""
DACA Replication Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the US.

Author: Replication Analysis
Date: January 2026
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

print(f"\nTotal observations: {len(df)}")
print(f"Unique years: {sorted(df['YEAR'].unique())}")

# =============================================================================
# DESCRIPTIVE STATISTICS
# =============================================================================

print("\n" + "="*80)
print("DESCRIPTIVE STATISTICS")
print("="*80)

# Sample sizes by treatment group and period
print("\n--- Sample Sizes by Treatment Group and Period ---")
sample_counts = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
sample_counts.index = ['Control (31-35)', 'Treated (26-30)', 'Total']
sample_counts.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)', 'Total']
print(sample_counts)

# FT employment rates
print("\n--- Full-Time Employment Rates ---")
ft_rates = pd.crosstab(df['ELIGIBLE'], df['AFTER'], values=df['FT'], aggfunc='mean')
ft_rates.index = ['Control (31-35)', 'Treated (26-30)']
ft_rates.columns = ['Pre-DACA', 'Post-DACA']
ft_rates['Change'] = ft_rates['Post-DACA'] - ft_rates['Pre-DACA']
print(ft_rates.round(4))

# Simple DiD calculation
did_simple = (ft_rates.loc['Treated (26-30)', 'Post-DACA'] - ft_rates.loc['Treated (26-30)', 'Pre-DACA']) - \
             (ft_rates.loc['Control (31-35)', 'Post-DACA'] - ft_rates.loc['Control (31-35)', 'Pre-DACA'])
print(f"\nSimple DiD estimate: {did_simple:.4f}")

# Year-by-year FT rates
print("\n--- FT Employment Rates by Year and Treatment Group ---")
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].agg(['mean', 'count', 'std']).round(4)
yearly_ft = yearly_ft.unstack()
print(yearly_ft)

# Demographics by treatment group
print("\n--- Demographics by Treatment Group (Pre-DACA Period) ---")
pre_daca = df[df['AFTER'] == 0]
demo_stats = pre_daca.groupby('ELIGIBLE').agg({
    'AGE': 'mean',
    'SEX': lambda x: (x == 1).mean(),  # Proportion male (SEX=1 is male)
    'FAMSIZE': 'mean',
    'NCHILD': 'mean',
    'YRSUSA1': 'mean'
}).round(3)
demo_stats.columns = ['Mean Age', 'Proportion Male', 'Mean Family Size', 'Mean N Children', 'Mean Years in USA']
demo_stats.index = ['Control (31-35)', 'Treated (26-30)']
print(demo_stats)

# Education distribution
print("\n--- Education Distribution by Treatment Group (Pre-DACA) ---")
educ_dist = pd.crosstab(pre_daca['ELIGIBLE'], pre_daca['EDUC_RECODE'], normalize='index')
educ_dist.index = ['Control (31-35)', 'Treated (26-30)']
print(educ_dist.round(4))

# =============================================================================
# DIFFERENCE-IN-DIFFERENCES ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("="*80)

# Create interaction term
df['TREATED_POST'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (no controls)
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST', data=df).fit(cov_type='HC1')
print(model1.summary().tables[1])
print(f"\nDiD Estimate: {model1.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model1.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['TREATED_POST', 0]:.4f}, {model1.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model1.pvalues['TREATED_POST']:.4f}")
print(f"N: {int(model1.nobs)}")
print(f"R-squared: {model1.rsquared:.4f}")

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
# Create dummy variables for education
df['educ_hs'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['educ_somecoll'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['educ_twoyear'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['educ_ba'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Create marital status dummies
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST + SEX + married + NCHILD + FAMSIZE + educ_somecoll + educ_twoyear + educ_ba',
                 data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])
print(f"\nDiD Estimate: {model2.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model2.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['TREATED_POST', 0]:.4f}, {model2.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model2.pvalues['TREATED_POST']:.4f}")
print(f"N: {int(model2.nobs)}")
print(f"R-squared: {model2.rsquared:.4f}")

# Model 3: DiD with demographic controls and state fixed effects
print("\n--- Model 3: DiD with Demographic Controls and State Fixed Effects ---")
# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

# Build formula with state FEs
state_cols = [col for col in state_dummies.columns]
state_formula = ' + '.join(state_cols)
formula3 = f'FT ~ ELIGIBLE + AFTER + TREATED_POST + SEX + married + NCHILD + FAMSIZE + educ_somecoll + educ_twoyear + educ_ba + {state_formula}'

model3 = smf.ols(formula3, data=df_with_states).fit(cov_type='HC1')
print(f"\nDiD Estimate: {model3.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model3.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['TREATED_POST', 0]:.4f}, {model3.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model3.pvalues['TREATED_POST']:.4f}")
print(f"N: {int(model3.nobs)}")
print(f"R-squared: {model3.rsquared:.4f}")

# Model 4: DiD with state and year fixed effects
print("\n--- Model 4: DiD with State and Year Fixed Effects ---")
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_with_fe = pd.concat([df_with_states, year_dummies], axis=1)

year_cols = [col for col in year_dummies.columns]
year_formula = ' + '.join(year_cols)
# Note: Cannot include AFTER when we have year FEs as they are collinear
formula4 = f'FT ~ ELIGIBLE + TREATED_POST + SEX + married + NCHILD + FAMSIZE + educ_somecoll + educ_twoyear + educ_ba + {state_formula} + {year_formula}'

model4 = smf.ols(formula4, data=df_with_fe).fit(cov_type='HC1')
print(f"\nDiD Estimate: {model4.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model4.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['TREATED_POST', 0]:.4f}, {model4.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model4.pvalues['TREATED_POST']:.4f}")
print(f"N: {int(model4.nobs)}")
print(f"R-squared: {model4.rsquared:.4f}")

# Model 5: Full model with additional labor market controls
print("\n--- Model 5: Full Model with Labor Market and Policy Controls ---")
# Add state-level controls (labor market conditions)
formula5 = f'FT ~ ELIGIBLE + TREATED_POST + SEX + married + NCHILD + FAMSIZE + educ_somecoll + educ_twoyear + educ_ba + LFPR + UNEMP + {state_formula} + {year_formula}'

model5 = smf.ols(formula5, data=df_with_fe).fit(cov_type='HC1')
print(f"\nDiD Estimate: {model5.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model5.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['TREATED_POST', 0]:.4f}, {model5.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model5.pvalues['TREATED_POST']:.4f}")
print(f"N: {int(model5.nobs)}")
print(f"R-squared: {model5.rsquared:.4f}")

# =============================================================================
# ROBUSTNESS CHECKS
# =============================================================================

print("\n" + "="*80)
print("ROBUSTNESS CHECKS")
print("="*80)

# Robustness 1: Weighted regression using person weights
print("\n--- Robustness 1: Weighted Regression (Person Weights) ---")
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + TREATED_POST + SEX + married + NCHILD + FAMSIZE + educ_somecoll + educ_twoyear + educ_ba',
                         data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Estimate: {model_weighted.params['TREATED_POST']:.4f}")
print(f"Std Error (HC1): {model_weighted.bse['TREATED_POST']:.4f}")
print(f"95% CI: [{model_weighted.conf_int().loc['TREATED_POST', 0]:.4f}, {model_weighted.conf_int().loc['TREATED_POST', 1]:.4f}]")
print(f"P-value: {model_weighted.pvalues['TREATED_POST']:.4f}")

# Robustness 2: By gender
print("\n--- Robustness 2: By Gender ---")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST', data=df_sex).fit(cov_type='HC1')
    print(f"\n{sex_label}:")
    print(f"  DiD Estimate: {model_sex.params['TREATED_POST']:.4f}")
    print(f"  Std Error: {model_sex.bse['TREATED_POST']:.4f}")
    print(f"  P-value: {model_sex.pvalues['TREATED_POST']:.4f}")
    print(f"  N: {int(model_sex.nobs)}")

# Robustness 3: By marital status
print("\n--- Robustness 3: By Marital Status ---")
for mar_status, mar_label in [(1, 'Married'), (0, 'Not Married')]:
    df_mar = df[df['married'] == mar_status]
    model_mar = smf.ols('FT ~ ELIGIBLE + AFTER + TREATED_POST', data=df_mar).fit(cov_type='HC1')
    print(f"\n{mar_label}:")
    print(f"  DiD Estimate: {model_mar.params['TREATED_POST']:.4f}")
    print(f"  Std Error: {model_mar.bse['TREATED_POST']:.4f}")
    print(f"  P-value: {model_mar.pvalues['TREATED_POST']:.4f}")
    print(f"  N: {int(model_mar.nobs)}")

# =============================================================================
# EVENT STUDY / PARALLEL TRENDS
# =============================================================================

print("\n" + "="*80)
print("EVENT STUDY ANALYSIS (Parallel Trends)")
print("="*80)

# Create year-specific treatment indicators (using 2011 as reference year)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interaction terms
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'elig_x_{year}'] = df['ELIGIBLE'] * df[f'year_{year}']

event_formula = 'FT ~ ELIGIBLE + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + elig_x_2008 + elig_x_2009 + elig_x_2010 + elig_x_2013 + elig_x_2014 + elig_x_2015 + elig_x_2016'
model_event = smf.ols(event_formula, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (Reference: 2011):")
print("-" * 60)
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
print(f"{'Year':<10} {'Coefficient':<15} {'Std Error':<15} {'P-value':<10}")
print("-" * 60)
for year in event_years:
    coef = model_event.params[f'elig_x_{year}']
    se = model_event.bse[f'elig_x_{year}']
    pval = model_event.pvalues[f'elig_x_{year}']
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f"{year:<10} {coef:< 15.4f} {se:< 15.4f} {pval:< 10.4f} {sig}")

# =============================================================================
# PLACEBO TESTS
# =============================================================================

print("\n" + "="*80)
print("PLACEBO TEST (Pre-Trends)")
print("="*80)

# Test using only pre-treatment data, with 2010-2011 as "fake treatment" period
df_pre = df[df['AFTER'] == 0].copy()
df_pre['fake_after'] = (df_pre['YEAR'].isin([2010, 2011])).astype(int)
df_pre['fake_treated_post'] = df_pre['ELIGIBLE'] * df_pre['fake_after']

model_placebo = smf.ols('FT ~ ELIGIBLE + fake_after + fake_treated_post', data=df_pre).fit(cov_type='HC1')
print("\nPlacebo Test (Using 2010-2011 as 'fake' post-period):")
print(f"  Placebo DiD Estimate: {model_placebo.params['fake_treated_post']:.4f}")
print(f"  Std Error: {model_placebo.bse['fake_treated_post']:.4f}")
print(f"  P-value: {model_placebo.pvalues['fake_treated_post']:.4f}")
print(f"  (Null of no pre-trend: expect coefficient ~ 0 and not significant)")

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\n" + "="*80)
print("SUMMARY OF MAIN RESULTS")
print("="*80)

results_summary = pd.DataFrame({
    'Model': ['(1) Basic DiD', '(2) + Demographics', '(3) + State FE', '(4) + Year FE', '(5) + Labor Market', '(6) Weighted'],
    'DiD Estimate': [model1.params['TREATED_POST'], model2.params['TREATED_POST'],
                     model3.params['TREATED_POST'], model4.params['TREATED_POST'],
                     model5.params['TREATED_POST'], model_weighted.params['TREATED_POST']],
    'Std Error': [model1.bse['TREATED_POST'], model2.bse['TREATED_POST'],
                  model3.bse['TREATED_POST'], model4.bse['TREATED_POST'],
                  model5.bse['TREATED_POST'], model_weighted.bse['TREATED_POST']],
    'P-value': [model1.pvalues['TREATED_POST'], model2.pvalues['TREATED_POST'],
                model3.pvalues['TREATED_POST'], model4.pvalues['TREATED_POST'],
                model5.pvalues['TREATED_POST'], model_weighted.pvalues['TREATED_POST']],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs),
          int(model4.nobs), int(model5.nobs), int(model_weighted.nobs)],
    'R-squared': [model1.rsquared, model2.rsquared, model3.rsquared,
                  model4.rsquared, model5.rsquared, model_weighted.rsquared]
})

print("\n")
print(results_summary.to_string(index=False))

# =============================================================================
# SAVE RESULTS FOR LATEX
# =============================================================================

# Save key results to files for LaTeX import
results_dict = {
    'simple_did': did_simple,
    'model1_did': model1.params['TREATED_POST'],
    'model1_se': model1.bse['TREATED_POST'],
    'model1_pval': model1.pvalues['TREATED_POST'],
    'model1_n': int(model1.nobs),
    'model1_r2': model1.rsquared,
    'model2_did': model2.params['TREATED_POST'],
    'model2_se': model2.bse['TREATED_POST'],
    'model2_pval': model2.pvalues['TREATED_POST'],
    'model2_n': int(model2.nobs),
    'model2_r2': model2.rsquared,
    'model3_did': model3.params['TREATED_POST'],
    'model3_se': model3.bse['TREATED_POST'],
    'model3_pval': model3.pvalues['TREATED_POST'],
    'model3_n': int(model3.nobs),
    'model3_r2': model3.rsquared,
    'model4_did': model4.params['TREATED_POST'],
    'model4_se': model4.bse['TREATED_POST'],
    'model4_pval': model4.pvalues['TREATED_POST'],
    'model4_n': int(model4.nobs),
    'model4_r2': model4.rsquared,
    'model5_did': model5.params['TREATED_POST'],
    'model5_se': model5.bse['TREATED_POST'],
    'model5_pval': model5.pvalues['TREATED_POST'],
    'model5_n': int(model5.nobs),
    'model5_r2': model5.rsquared,
    'model_weighted_did': model_weighted.params['TREATED_POST'],
    'model_weighted_se': model_weighted.bse['TREATED_POST'],
    'model_weighted_pval': model_weighted.pvalues['TREATED_POST'],
    'placebo_did': model_placebo.params['fake_treated_post'],
    'placebo_se': model_placebo.bse['fake_treated_post'],
    'placebo_pval': model_placebo.pvalues['fake_treated_post'],
    # Sample descriptives
    'n_total': len(df),
    'n_treated': int((df['ELIGIBLE'] == 1).sum()),
    'n_control': int((df['ELIGIBLE'] == 0).sum()),
    'n_pre': int((df['AFTER'] == 0).sum()),
    'n_post': int((df['AFTER'] == 1).sum()),
    'ft_treated_pre': ft_rates.loc['Treated (26-30)', 'Pre-DACA'],
    'ft_treated_post': ft_rates.loc['Treated (26-30)', 'Post-DACA'],
    'ft_control_pre': ft_rates.loc['Control (31-35)', 'Pre-DACA'],
    'ft_control_post': ft_rates.loc['Control (31-35)', 'Post-DACA'],
}

# Save results
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n\nAnalysis complete. Results saved to analysis_results.json")

# =============================================================================
# ADDITIONAL STATISTICS FOR REPORT
# =============================================================================

print("\n" + "="*80)
print("ADDITIONAL STATISTICS FOR REPORT")
print("="*80)

# State distribution
print("\n--- Top 10 States by Sample Size ---")
state_counts = df['statename'].value_counts().head(10)
print(state_counts)

# Employment status distribution
print("\n--- Employment Status (EMPSTAT) Distribution ---")
empstat_dist = df['EMPSTAT'].value_counts()
print(empstat_dist)

# Age distribution
print("\n--- Age Statistics ---")
print(f"Mean age in sample: {df['AGE'].mean():.2f}")
print(f"Age range: {df['AGE'].min()} - {df['AGE'].max()}")

# Hours worked distribution
print("\n--- Usual Hours Worked Distribution (for employed) ---")
employed = df[df['EMPSTAT'] == 1]
print(f"Mean hours: {employed['UHRSWORK'].mean():.2f}")
print(f"Median hours: {employed['UHRSWORK'].median():.0f}")
print(f"% working 35+ hours: {(employed['UHRSWORK'] >= 35).mean()*100:.1f}%")

# Check FT definition
print("\n--- FT Variable Verification ---")
print(f"FT mean: {df['FT'].mean():.4f}")
print(f"Proportion with UHRSWORK >= 35: {(df['UHRSWORK'] >= 35).mean():.4f}")

# DACA policy timeline context
print("\n--- Year Distribution ---")
print(df['YEAR'].value_counts().sort_index())

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

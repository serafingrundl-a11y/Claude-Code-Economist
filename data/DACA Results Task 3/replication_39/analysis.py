"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among Mexican-born individuals
Treatment: DACA eligibility (ages 26-30 in June 2012 = treated; ages 31-35 in June 2012 = control)
Outcome: Full-time employment (FT = 1 if usually works 35+ hours per week)
Method: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("="*80)
print("DACA REPLICATION ANALYSIS")
print("="*80)
print("\n1. LOADING DATA")
print("-"*40)

df = pd.read_csv('data/prepared_data_labelled_version.csv')
print(f"Total observations: {len(df)}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Check key variables
print("\n2. DATA EXPLORATION")
print("-"*40)

print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())

print(f"\nFT distribution:")
print(df['FT'].value_counts())

print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# Cross-tabulation
print("\n3. TREATMENT AND CONTROL GROUPS")
print("-"*40)
print("\nObservations by ELIGIBLE x AFTER:")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
print(crosstab)

# Calculate mean FT by group
print("\n4. MEAN FULL-TIME EMPLOYMENT BY GROUP")
print("-"*40)
means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std'])
print(means)

# Calculate unweighted DiD
print("\n5. SIMPLE (UNWEIGHTED) DIFFERENCE-IN-DIFFERENCES")
print("-"*40)

# Group means
mean_treated_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
mean_treated_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
mean_control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
mean_control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()

print(f"Treated (ELIGIBLE=1), After: {mean_treated_after:.4f}")
print(f"Treated (ELIGIBLE=1), Before: {mean_treated_before:.4f}")
print(f"Control (ELIGIBLE=0), After: {mean_control_after:.4f}")
print(f"Control (ELIGIBLE=0), Before: {mean_control_before:.4f}")

did_estimate = (mean_treated_after - mean_treated_before) - (mean_control_after - mean_control_before)
print(f"\nDiD estimate (unweighted): {did_estimate:.4f}")

# Weighted means
print("\n6. WEIGHTED DIFFERENCE-IN-DIFFERENCES")
print("-"*40)

def weighted_mean(df, val_col, weight_col):
    return np.average(df[val_col], weights=df[weight_col])

w_mean_treated_after = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')
w_mean_treated_before = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
w_mean_control_after = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')
w_mean_control_before = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')

print(f"Treated (ELIGIBLE=1), After (weighted): {w_mean_treated_after:.4f}")
print(f"Treated (ELIGIBLE=1), Before (weighted): {w_mean_treated_before:.4f}")
print(f"Control (ELIGIBLE=0), After (weighted): {w_mean_control_after:.4f}")
print(f"Control (ELIGIBLE=0), Before (weighted): {w_mean_control_before:.4f}")

w_did_estimate = (w_mean_treated_after - w_mean_treated_before) - (w_mean_control_after - w_mean_control_before)
print(f"\nDiD estimate (weighted): {w_did_estimate:.4f}")

# Main Regression: OLS DiD
print("\n7. MAIN REGRESSION: OLS DIFFERENCE-IN-DIFFERENCES")
print("-"*40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print("\nModel 1: Basic DiD (Unweighted)")
print(model1.summary())

# Model 2: Basic DiD (weighted)
print("\n" + "="*80)
print("Model 2: Basic DiD (Weighted)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with robust standard errors (HC1)
print("\n" + "="*80)
print("Model 3: Basic DiD (Weighted, Robust SE)")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary())

# Descriptive statistics
print("\n8. DESCRIPTIVE STATISTICS")
print("-"*40)

print("\nSample sizes:")
print(f"  Treatment group (ELIGIBLE=1): {len(df[df['ELIGIBLE']==1])}")
print(f"  Control group (ELIGIBLE=0): {len(df[df['ELIGIBLE']==0])}")
print(f"  Pre-DACA period (AFTER=0): {len(df[df['AFTER']==0])}")
print(f"  Post-DACA period (AFTER=1): {len(df[df['AFTER']==1])}")

print("\nAge in June 2012:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

print("\nSex distribution by group (1=Male, 2=Female):")
sex_by_group = df.groupby(['ELIGIBLE', 'AFTER'])['SEX'].value_counts(normalize=True).unstack()
print(sex_by_group)

# Education distribution
print("\nEducation distribution by ELIGIBLE status:")
educ_by_elig = pd.crosstab(df['ELIGIBLE'], df['EDUC_RECODE'], normalize='index')
print(educ_by_elig)

# Model 4: DiD with covariates
print("\n9. REGRESSION WITH COVARIATES")
print("-"*40)

# Recode SEX to binary (male=1, female=0)
df['MALE'] = (df['SEX'] == 'Male').astype(int)

# Create education dummies
educ_dummies = pd.get_dummies(df['EDUC_RECODE'], prefix='EDUC', drop_first=True)
df = pd.concat([df, educ_dummies], axis=1)

# Model 4: DiD with demographics
print("\nModel 4: DiD with Demographic Controls (Weighted, Robust SE)")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + C(MARST) + NCHILD'
model4 = smf.wls(formula4, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with demographics and education
print("\n" + "="*80)
print("Model 5: DiD with Demographic and Education Controls (Weighted, Robust SE)")
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model5 = smf.wls(formula5, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model5.summary())

# Model 6: DiD with year fixed effects
print("\n" + "="*80)
print("Model 6: DiD with Year Fixed Effects (Weighted, Robust SE)")
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR)'
model6 = smf.wls(formula6, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model6.summary())

# Model 7: Full model with year FE and covariates
print("\n" + "="*80)
print("Model 7: Full Model with Year FE and Covariates (Weighted, Robust SE)")
formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model7 = smf.wls(formula7, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model7.summary())

# State-level variation
print("\n10. STATE-LEVEL ANALYSIS")
print("-"*40)

# Model 8: DiD with state fixed effects
print("\nModel 8: DiD with State Fixed Effects (Weighted, Robust SE)")
formula8 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + C(STATEFIP)'
model8 = smf.wls(formula8, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model8.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model8.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model8.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model8.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 9: Full specification
print("\n" + "="*80)
print("Model 9: Full Specification (Year FE, State FE, Covariates, Weighted, Robust SE)")
formula9 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + C(YEAR) + C(STATEFIP) + MALE + C(MARST) + NCHILD + C(EDUC_RECODE)'
model9 = smf.wls(formula9, data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD coefficient (ELIGIBLE_AFTER): {model9.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model9.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model9.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model9.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model9.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model9.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")

# Summary table of main results
print("\n" + "="*80)
print("11. SUMMARY TABLE OF DiD COEFFICIENTS")
print("="*80)

results_summary = pd.DataFrame({
    'Model': [
        'Model 1: Basic DiD (Unweighted)',
        'Model 2: Basic DiD (Weighted)',
        'Model 3: Basic DiD (Weighted, Robust SE)',
        'Model 4: + Demographics',
        'Model 5: + Demographics + Education',
        'Model 6: Year FE',
        'Model 7: Year FE + Controls',
        'Model 8: State FE',
        'Model 9: Full Specification'
    ],
    'DiD Coefficient': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model6.params['ELIGIBLE_AFTER'],
        model7.params['ELIGIBLE_AFTER'],
        model8.params['ELIGIBLE_AFTER'],
        model9.params['ELIGIBLE_AFTER']
    ],
    'Std Error': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model6.bse['ELIGIBLE_AFTER'],
        model7.bse['ELIGIBLE_AFTER'],
        model8.bse['ELIGIBLE_AFTER'],
        model9.bse['ELIGIBLE_AFTER']
    ],
    't-stat': [
        model1.tvalues['ELIGIBLE_AFTER'],
        model2.tvalues['ELIGIBLE_AFTER'],
        model3.tvalues['ELIGIBLE_AFTER'],
        model4.tvalues['ELIGIBLE_AFTER'],
        model5.tvalues['ELIGIBLE_AFTER'],
        model6.tvalues['ELIGIBLE_AFTER'],
        model7.tvalues['ELIGIBLE_AFTER'],
        model8.tvalues['ELIGIBLE_AFTER'],
        model9.tvalues['ELIGIBLE_AFTER']
    ],
    'p-value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER'],
        model6.pvalues['ELIGIBLE_AFTER'],
        model7.pvalues['ELIGIBLE_AFTER'],
        model8.pvalues['ELIGIBLE_AFTER'],
        model9.pvalues['ELIGIBLE_AFTER']
    ]
})
print(results_summary.to_string(index=False))

# Preferred estimate
print("\n" + "="*80)
print("12. PREFERRED ESTIMATE")
print("="*80)
print("\nI select Model 7 (Year FE + Covariates) as the preferred specification because:")
print("  - It includes year fixed effects to control for time trends")
print("  - It includes key demographic covariates (sex, marital status, children, education)")
print("  - It uses sample weights (PERWT) for population representativeness")
print("  - It uses robust standard errors for valid inference")
print("")
print(f"Preferred DiD Estimate: {model7.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error (Robust): {model7.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"Sample Size: {int(model7.nobs)}")

# Parallel trends check
print("\n" + "="*80)
print("13. PARALLEL TRENDS CHECK (PRE-TREATMENT)")
print("="*80)

# Look at pre-period trends
df_pre = df[df['AFTER']==0].copy()
df_pre['ELIGIBLE_YEAR'] = df_pre['ELIGIBLE'] * df_pre['YEAR']

print("\nFT by Year and Eligible (Pre-period):")
pre_trends = df_pre.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
print(pre_trends)

# Event study style analysis
print("\n" + "="*80)
print("14. EVENT STUDY ANALYSIS")
print("="*80)

# Create year dummies interacted with ELIGIBLE
years = sorted(df['YEAR'].unique())
ref_year = 2011  # reference year (last pre-treatment year)

for year in years:
    if year != ref_year:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression
event_vars = [f'ELIGIBLE_YEAR_{year}' for year in years if year != ref_year]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join([f'YEAR_{year}' for year in years if year != ref_year]) + ' + ' + ' + '.join(event_vars)
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print(f"\nEvent Study Coefficients (Reference Year = {ref_year}):")
for year in years:
    if year != ref_year:
        coef = model_event.params[f'ELIGIBLE_YEAR_{year}']
        se = model_event.bse[f'ELIGIBLE_YEAR_{year}']
        pval = model_event.pvalues[f'ELIGIBLE_YEAR_{year}']
        print(f"  Year {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")

# Heterogeneity by sex
print("\n" + "="*80)
print("15. HETEROGENEITY ANALYSIS BY SEX")
print("="*80)

df_male = df[df['SEX']=='Male']
df_female = df[df['SEX']=='Female']

formula_base = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER'

model_male = smf.wls(formula_base, data=df_male, weights=df_male['PERWT']).fit(cov_type='HC1')
model_female = smf.wls(formula_base, data=df_female, weights=df_female['PERWT']).fit(cov_type='HC1')

print(f"\nMales only:")
print(f"  DiD Coefficient: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  N: {int(model_male.nobs)}")

print(f"\nFemales only:")
print(f"  DiD Coefficient: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  N: {int(model_female.nobs)}")

# Heterogeneity by education
print("\n" + "="*80)
print("16. HETEROGENEITY ANALYSIS BY EDUCATION")
print("="*80)

df_lesshs = df[df['EDUC_RECODE']=='Less than High School']
df_hs = df[df['EDUC_RECODE']=='High School Degree']
df_somecol = df[df['EDUC_RECODE'].isin(['Some College', 'Two-Year Degree'])]
df_ba = df[df['EDUC_RECODE']=='BA+']

for name, subset in [('Less than HS', df_lesshs), ('High School', df_hs),
                      ('Some College/2-Year', df_somecol), ('BA+', df_ba)]:
    if len(subset) > 50:  # Only if sufficient sample
        model_sub = smf.wls(formula_base, data=subset, weights=subset['PERWT']).fit(cov_type='HC1')
        print(f"\n{name}:")
        print(f"  DiD Coefficient: {model_sub.params['ELIGIBLE_AFTER']:.4f}")
        print(f"  Standard Error: {model_sub.bse['ELIGIBLE_AFTER']:.4f}")
        print(f"  N: {int(model_sub.nobs)}")
    else:
        print(f"\n{name}: Insufficient sample size (N={len(subset)})")

# Save key results for the report
print("\n" + "="*80)
print("17. SAVING KEY STATISTICS FOR REPORT")
print("="*80)

# Create summary statistics table
summary_stats = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum',
    'AGE': 'mean',
    'MALE': 'mean',
    'NCHILD': 'mean'
}).round(4)
print("\nSummary Statistics by Group:")
print(summary_stats)

# Calculate weighted means for each group
print("\nWeighted FT means by group:")
for elig in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==elig) & (df['AFTER']==after)]
        w_mean = np.average(subset['FT'], weights=subset['PERWT'])
        w_sum = subset['PERWT'].sum()
        label = f"ELIGIBLE={elig}, AFTER={after}"
        print(f"  {label}: {w_mean:.4f} (weighted N = {w_sum:,.0f})")

# Final summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"""
Research Question: Effect of DACA eligibility on full-time employment

Sample: Mexican-born individuals in the US
  - Treatment group: Ages 26-30 in June 2012 (ELIGIBLE=1)
  - Control group: Ages 31-35 in June 2012 (ELIGIBLE=0)

Data: ACS 2008-2011 (pre-DACA) and 2013-2016 (post-DACA)
  - 2012 excluded (treatment timing ambiguity)
  - Total observations: {len(df)}

Preferred Estimate (Model 7: Year FE + Covariates):
  - DiD Effect: {model7.params['ELIGIBLE_AFTER']:.4f}
  - Standard Error (Robust): {model7.bse['ELIGIBLE_AFTER']:.4f}
  - 95% CI: [{model7.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]
  - p-value: {model7.pvalues['ELIGIBLE_AFTER']:.4f}

Interpretation:
  DACA eligibility is associated with a {model7.params['ELIGIBLE_AFTER']*100:.2f} percentage point
  {'increase' if model7.params['ELIGIBLE_AFTER'] > 0 else 'decrease'} in the probability of full-time employment.
  This effect is {'statistically significant' if model7.pvalues['ELIGIBLE_AFTER'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

# Export results for LaTeX
results_for_latex = {
    'n_total': len(df),
    'n_treated': len(df[df['ELIGIBLE']==1]),
    'n_control': len(df[df['ELIGIBLE']==0]),
    'n_pre': len(df[df['AFTER']==0]),
    'n_post': len(df[df['AFTER']==1]),
    'did_basic': model3.params['ELIGIBLE_AFTER'],
    'se_basic': model3.bse['ELIGIBLE_AFTER'],
    'did_preferred': model7.params['ELIGIBLE_AFTER'],
    'se_preferred': model7.bse['ELIGIBLE_AFTER'],
    'ci_low': model7.conf_int().loc['ELIGIBLE_AFTER', 0],
    'ci_high': model7.conf_int().loc['ELIGIBLE_AFTER', 1],
    'pval': model7.pvalues['ELIGIBLE_AFTER'],
    'mean_ft_treated_pre': w_mean_treated_before,
    'mean_ft_treated_post': w_mean_treated_after,
    'mean_ft_control_pre': w_mean_control_before,
    'mean_ft_control_post': w_mean_control_after,
}

# Save to CSV for easy import
import json
with open('results_summary.json', 'w') as f:
    json.dump(results_for_latex, f, indent=2)

print("\nResults saved to results_summary.json")
print("\nAnalysis complete!")

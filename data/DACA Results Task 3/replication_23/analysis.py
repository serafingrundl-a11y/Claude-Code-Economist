"""
DACA Replication Study - Analysis Script
=========================================
Replication 23

Research Question: What was the causal impact of DACA eligibility on
full-time employment among Hispanic-Mexican Mexican-born individuals in the US?

Design: Difference-in-Differences
- Treatment: DACA-eligible individuals aged 26-30 at June 15, 2012
- Control: Individuals aged 31-35 at June 15, 2012
- Pre-period: 2008-2011
- Post-period: 2013-2016
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("\n" + "=" * 80)
print("1. LOADING DATA")
print("=" * 80)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")

# ==============================================================================
# 2. VERIFY KEY VARIABLES
# ==============================================================================
print("\n" + "=" * 80)
print("2. VERIFYING KEY VARIABLES")
print("=" * 80)

print("\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

print("\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())
print(f"  Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum()}")
print(f"  Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum()}")

print("\nAFTER distribution:")
print(df['AFTER'].value_counts())
print(f"  Pre-DACA (AFTER=0): {(df['AFTER']==0).sum()}")
print(f"  Post-DACA (AFTER=1): {(df['AFTER']==1).sum()}")

print("\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())
print(f"  Not full-time (FT=0): {(df['FT']==0).sum()}")
print(f"  Full-time (FT=1): {(df['FT']==1).sum()}")

print("\nAGE_IN_JUNE_2012 by ELIGIBLE:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# ==============================================================================
# 3. DESCRIPTIVE STATISTICS BY GROUP AND PERIOD
# ==============================================================================
print("\n" + "=" * 80)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 80)

# Create group labels for clarity
df['Group'] = df['ELIGIBLE'].map({1: 'Treatment (26-30)', 0: 'Control (31-35)'})
df['Period'] = df['AFTER'].map({1: 'Post-DACA (2013-16)', 0: 'Pre-DACA (2008-11)'})

# Full-time employment rates by group and period (unweighted)
print("\n--- Unweighted Full-Time Employment Rates ---")
ft_by_group_period = df.groupby(['Group', 'Period'])['FT'].agg(['mean', 'count', 'std'])
ft_by_group_period['se'] = ft_by_group_period['std'] / np.sqrt(ft_by_group_period['count'])
print(ft_by_group_period)

# Create 2x2 table
print("\n--- 2x2 DiD Table (Unweighted) ---")
table_unweighted = df.pivot_table(values='FT', index='ELIGIBLE', columns='AFTER', aggfunc='mean')
table_unweighted.index = ['Control (31-35)', 'Treatment (26-30)']
table_unweighted.columns = ['Pre-DACA', 'Post-DACA']
table_unweighted['Difference'] = table_unweighted['Post-DACA'] - table_unweighted['Pre-DACA']
print(table_unweighted)

# DiD estimate
did_unweighted = ((table_unweighted.loc['Treatment (26-30)', 'Post-DACA'] -
                   table_unweighted.loc['Treatment (26-30)', 'Pre-DACA']) -
                  (table_unweighted.loc['Control (31-35)', 'Post-DACA'] -
                   table_unweighted.loc['Control (31-35)', 'Pre-DACA']))
print(f"\nDifference-in-Differences (unweighted): {did_unweighted:.4f}")

# Weighted analysis
print("\n--- Weighted Full-Time Employment Rates ---")
def weighted_mean(x, w):
    return np.average(x, weights=w)

ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda g: weighted_mean(g['FT'], g['PERWT'])
).unstack()
ft_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
ft_weighted.columns = ['Pre-DACA', 'Post-DACA']
ft_weighted['Difference'] = ft_weighted['Post-DACA'] - ft_weighted['Pre-DACA']
print(ft_weighted)

did_weighted = ((ft_weighted.loc['Treatment (26-30)', 'Post-DACA'] -
                 ft_weighted.loc['Treatment (26-30)', 'Pre-DACA']) -
                (ft_weighted.loc['Control (31-35)', 'Post-DACA'] -
                 ft_weighted.loc['Control (31-35)', 'Pre-DACA']))
print(f"\nDifference-in-Differences (weighted): {did_weighted:.4f}")

# ==============================================================================
# 4. SAMPLE SIZE BY GROUP/PERIOD
# ==============================================================================
print("\n" + "=" * 80)
print("4. SAMPLE SIZES")
print("=" * 80)

sample_sizes = df.groupby(['Group', 'Period']).size().unstack()
print(sample_sizes)
print(f"\nTotal sample size: {len(df)}")

# ==============================================================================
# 5. DEMOGRAPHIC CHARACTERISTICS BY GROUP
# ==============================================================================
print("\n" + "=" * 80)
print("5. DEMOGRAPHIC CHARACTERISTICS")
print("=" * 80)

# Sex distribution (IPUMS: 1=Male, 2=Female)
print("\nSex distribution by group:")
sex_dist = pd.crosstab(df['Group'], df['SEX'], normalize='index') * 100
sex_dist.columns = ['Male', 'Female']
print(sex_dist.round(1))

# Education distribution
print("\nEducation distribution by group:")
educ_dist = pd.crosstab(df['Group'], df['EDUC_RECODE'], normalize='index') * 100
print(educ_dist.round(1))

# Marital status distribution (IPUMS codes)
print("\nMarital status by group:")
marst_dist = pd.crosstab(df['Group'], df['MARST'], normalize='index') * 100
print(marst_dist.round(1))

# ==============================================================================
# 6. TRENDS BY YEAR
# ==============================================================================
print("\n" + "=" * 80)
print("6. FULL-TIME EMPLOYMENT TRENDS BY YEAR")
print("=" * 80)

# Unweighted by year
ft_by_year_group = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
ft_by_year_group.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nUnweighted FT rates by year:")
print(ft_by_year_group.round(4))

# Calculate year-over-year differences
ft_by_year_group['Difference'] = ft_by_year_group['Treatment (26-30)'] - ft_by_year_group['Control (31-35)']
print("\nDifference (Treatment - Control) by year:")
print(ft_by_year_group['Difference'].round(4))

# ==============================================================================
# 7. DIFFERENCE-IN-DIFFERENCES REGRESSION
# ==============================================================================
print("\n" + "=" * 80)
print("7. DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (Unweighted)
print("\n--- Model 1: Basic DiD (Unweighted OLS) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# Model 2: Basic DiD (Weighted)
print("\n--- Model 2: Basic DiD (Weighted OLS) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects (Weighted) ---")
df['YEAR_cat'] = pd.Categorical(df['YEAR'])
model3 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model3.summary())

# Model 4: DiD with covariates
print("\n--- Model 4: DiD with Demographic Covariates (Weighted) ---")
# Create binary sex variable (1=Female, 0=Male based on IPUMS coding)
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Create education dummies from EDUC_RECODE
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Marital status: create married indicator
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

model4 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit()
print(model4.summary())

# Model 5: DiD with covariates and state fixed effects
print("\n--- Model 5: DiD with Covariates and State Fixed Effects (Weighted) ---")
df['STATE_cat'] = pd.Categorical(df['STATEFIP'])
model5 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit()
# Print only key coefficients
print("\nKey coefficients from Model 5:")
print(f"ELIGIBLE_AFTER (DiD): {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"\nFEMALE: {model5.params['FEMALE']:.4f} (SE: {model5.bse['FEMALE']:.4f})")
print(f"MARRIED: {model5.params['MARRIED']:.4f} (SE: {model5.bse['MARRIED']:.4f})")
print(f"\nR-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs)}")

# ==============================================================================
# 8. ROBUST STANDARD ERRORS (CLUSTERED BY STATE)
# ==============================================================================
print("\n" + "=" * 80)
print("8. CLUSTERED STANDARD ERRORS (BY STATE)")
print("=" * 80)

# Model with clustered standard errors
print("\n--- Model 6: DiD with Covariates, State FE, and Clustered SE ---")
model6 = smf.wls('FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_AFTER + FEMALE + MARRIED + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BA',
                 data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print("\nKey coefficients from Model 6 (Clustered SE by State):")
print(f"ELIGIBLE_AFTER (DiD): {model6.params['ELIGIBLE_AFTER']:.4f} (Clustered SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# ==============================================================================
# 9. SUMMARY OF RESULTS
# ==============================================================================
print("\n" + "=" * 80)
print("9. SUMMARY OF RESULTS")
print("=" * 80)

results_summary = pd.DataFrame({
    'Model': ['Basic DiD (unweighted)', 'Basic DiD (weighted)',
              'Year FE (weighted)', 'Covariates (weighted)',
              'State + Year FE (weighted)', 'Clustered SE'],
    'DiD Estimate': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                     model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                     model5.params['ELIGIBLE_AFTER'], model6.params['ELIGIBLE_AFTER']],
    'SE': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
           model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
           model5.bse['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER']],
    'p-value': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
                model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
                model5.pvalues['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER']]
})

results_summary['CI_lower'] = results_summary['DiD Estimate'] - 1.96 * results_summary['SE']
results_summary['CI_upper'] = results_summary['DiD Estimate'] + 1.96 * results_summary['SE']

print("\nAll Model Results:")
print(results_summary.to_string(index=False))

# ==============================================================================
# 10. PREFERRED ESTIMATE
# ==============================================================================
print("\n" + "=" * 80)
print("10. PREFERRED ESTIMATE")
print("=" * 80)

print("\nPreferred specification: Model 5 (Weighted DiD with Year FE, State FE, and Covariates)")
print("\nJustification:")
print("- Weights (PERWT) account for ACS survey design and make estimates representative")
print("- Year fixed effects control for common time trends affecting all groups")
print("- State fixed effects control for time-invariant state-level confounders")
print("- Covariates (sex, education, marital status) improve precision and control for composition")
print("- This is a standard specification in the DACA literature")

print(f"\n>>> PREFERRED ESTIMATE <<<")
print(f"Effect of DACA eligibility on full-time employment: {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {int(model5.nobs)}")

# Interpretation
effect_pct = model5.params['ELIGIBLE_AFTER'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.2f} percentage point")
print(f"{'increase' if effect_pct > 0 else 'decrease'} in the probability of full-time employment.")

if model5.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("This effect is statistically significant at the 5% level.")
else:
    print("This effect is NOT statistically significant at the 5% level.")

# ==============================================================================
# 11. SAVE RESULTS FOR REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("11. SAVING RESULTS")
print("=" * 80)

# Save key results to CSV for use in LaTeX report
results_summary.to_csv('model_results_summary.csv', index=False)
print("Model summary saved to model_results_summary.csv")

# Save detailed model output
with open('detailed_model_output.txt', 'w') as f:
    f.write("DACA REPLICATION STUDY - DETAILED MODEL OUTPUT\n")
    f.write("=" * 80 + "\n\n")
    f.write("MODEL 1: Basic DiD (Unweighted)\n")
    f.write(model1.summary().as_text())
    f.write("\n\n" + "=" * 80 + "\n\n")
    f.write("MODEL 2: Basic DiD (Weighted)\n")
    f.write(model2.summary().as_text())
    f.write("\n\n" + "=" * 80 + "\n\n")
    f.write("MODEL 4: DiD with Covariates (Weighted)\n")
    f.write(model4.summary().as_text())
    f.write("\n\n" + "=" * 80 + "\n\n")
    f.write("MODEL 5: Full specification (Preferred)\n")
    f.write(model5.summary().as_text())

print("Detailed output saved to detailed_model_output.txt")

# Save 2x2 tables
table_unweighted.to_csv('did_table_unweighted.csv')
ft_weighted.to_csv('did_table_weighted.csv')
print("DiD tables saved")

# Save trends data
ft_by_year_group.to_csv('ft_trends_by_year.csv')
print("Trends data saved")

# Save sample sizes
sample_sizes.to_csv('sample_sizes.csv')
print("Sample sizes saved")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

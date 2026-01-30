"""
DACA Replication Study - Difference-in-Differences Analysis
Analyzing the effect of DACA eligibility on full-time employment
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# Load the numeric version of the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 40)
print(f"Total observations: {len(df):,}")
print(f"Number of variables: {len(df.columns)}")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")

# ============================================================================
# 2. SAMPLE CHARACTERISTICS
# ============================================================================

print("\n2. SAMPLE CHARACTERISTICS")
print("-" * 40)

# Check ELIGIBLE and AFTER variables
print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts())

# Full-time employment
print(f"\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())

# Age at June 2012
print(f"\nAGE_IN_JUNE_2012 summary for ELIGIBLE=1 (treated):")
print(df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].describe())

print(f"\nAGE_IN_JUNE_2012 summary for ELIGIBLE=0 (comparison):")
print(df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].describe())

# ============================================================================
# 3. DESCRIPTIVE STATISTICS BY GROUP
# ============================================================================

print("\n3. DESCRIPTIVE STATISTICS")
print("-" * 40)

# Create cross-tabulation
cross_tab = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'PERWT': 'sum'
}).round(4)
print("\nFull-time employment rate by group:")
print(cross_tab)

# Calculate weighted means
print("\n\nWeighted Full-Time Employment Rates:")
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        weighted_mean = np.average(subset['FT'], weights=subset['PERWT'])
        n = len(subset)
        group_name = f"ELIGIBLE={eligible}, AFTER={after}"
        print(f"  {group_name}: {weighted_mean:.4f} (n={n:,})")

# ============================================================================
# 4. DIFFERENCE-IN-DIFFERENCES CALCULATION (SIMPLE)
# ============================================================================

print("\n4. SIMPLE DIFFERENCE-IN-DIFFERENCES")
print("-" * 40)

# Unweighted DiD
mean_11 = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()  # Treated, After
mean_10 = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()  # Treated, Before
mean_01 = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()  # Control, After
mean_00 = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()  # Control, Before

did_unweighted = (mean_11 - mean_10) - (mean_01 - mean_00)

print("\nUnweighted Means:")
print(f"  Treated (26-30), After (2013-16):  {mean_11:.4f}")
print(f"  Treated (26-30), Before (2008-11): {mean_10:.4f}")
print(f"  Control (31-35), After (2013-16):  {mean_01:.4f}")
print(f"  Control (31-35), Before (2008-11): {mean_00:.4f}")
print(f"\nDifference for Treated: {mean_11 - mean_10:.4f}")
print(f"Difference for Control: {mean_01 - mean_00:.4f}")
print(f"\nDifference-in-Differences (unweighted): {did_unweighted:.4f}")

# Weighted DiD
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

wmean_11 = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')
wmean_10 = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
wmean_01 = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')
wmean_00 = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')

did_weighted = (wmean_11 - wmean_10) - (wmean_01 - wmean_00)

print("\nWeighted Means:")
print(f"  Treated (26-30), After (2013-16):  {wmean_11:.4f}")
print(f"  Treated (26-30), Before (2008-11): {wmean_10:.4f}")
print(f"  Control (31-35), After (2013-16):  {wmean_01:.4f}")
print(f"  Control (31-35), Before (2008-11): {wmean_00:.4f}")
print(f"\nDifference for Treated: {wmean_11 - wmean_10:.4f}")
print(f"Difference for Control: {wmean_01 - wmean_00:.4f}")
print(f"\nDifference-in-Differences (weighted): {did_weighted:.4f}")

# ============================================================================
# 5. REGRESSION ANALYSIS - BASIC DiD MODEL
# ============================================================================

print("\n5. REGRESSION ANALYSIS - BASIC DiD MODEL")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (unweighted)
print("\nModel 1: Basic DiD (OLS, unweighted)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Basic DiD (weighted)
print("\nModel 2: Basic DiD (WLS, weighted by PERWT)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# ============================================================================
# 6. REGRESSION WITH ROBUST STANDARD ERRORS
# ============================================================================

print("\n6. REGRESSION WITH ROBUST (HC1) STANDARD ERRORS")
print("-" * 40)

# Model 3: Basic DiD with robust SEs
print("\nModel 3: Basic DiD (OLS, robust SEs)")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# ============================================================================
# 7. REGRESSION WITH COVARIATES
# ============================================================================

print("\n7. REGRESSION WITH COVARIATES")
print("-" * 40)

# Create dummy variables for categorical covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Education dummies (reference: Less than High School)
df['ED_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['ED_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['ED_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['ED_BA'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model 4: With demographic covariates
print("\nModel 4: DiD with demographic covariates (robust SEs)")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE'
model4 = smf.ols(formula4, data=df).fit(cov_type='HC1')
print(model4.summary().tables[1])

# ============================================================================
# 8. REGRESSION WITH STATE AND YEAR FIXED EFFECTS
# ============================================================================

print("\n8. REGRESSION WITH FIXED EFFECTS")
print("-" * 40)

# Create year dummies (reference: 2008)
for year in df['YEAR'].unique():
    if year != 2008:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

# Model 5: With year fixed effects
year_vars = [col for col in df.columns if col.startswith('YEAR_')]
formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(year_vars)
model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')

print("\nModel 5: DiD with year fixed effects (showing main coefficients only)")
# Show only key coefficients
key_vars = ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'MARRIED',
            'ED_HS', 'ED_SOMECOLL', 'ED_TWOYEAR', 'ED_BA', 'AGE']
print(f"{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t':>10} {'P>|t|':>10}")
print("-" * 60)
for var in key_vars:
    if var in model5.params.index:
        coef = model5.params[var]
        se = model5.bse[var]
        t = model5.tvalues[var]
        p = model5.pvalues[var]
        print(f"{var:<20} {coef:>10.4f} {se:>10.4f} {t:>10.2f} {p:>10.4f}")

# Model 6: Full model with state fixed effects
state_vars = [col for col in df.columns if col.startswith('STATE_')]
formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + ED_HS + ED_SOMECOLL + ED_TWOYEAR + ED_BA + AGE + ' + ' + '.join(year_vars) + ' + ' + ' + '.join(state_vars)

print("\nModel 6: DiD with year and state fixed effects")
model6 = smf.ols(formula6, data=df).fit(cov_type='HC1')

print(f"\n{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t':>10} {'P>|t|':>10}")
print("-" * 60)
for var in key_vars:
    if var in model6.params.index:
        coef = model6.params[var]
        se = model6.bse[var]
        t = model6.tvalues[var]
        p = model6.pvalues[var]
        print(f"{var:<20} {coef:>10.4f} {se:>10.4f} {t:>10.2f} {p:>10.4f}")

# ============================================================================
# 9. LINEAR PROBABILITY MODEL SUMMARY
# ============================================================================

print("\n9. SUMMARY OF RESULTS")
print("-" * 40)

print("\nDifference-in-Differences Estimates Across Models:")
print(f"{'Model':<50} {'DiD Estimate':>12} {'Std Error':>12} {'p-value':>10}")
print("-" * 84)

models_summary = [
    ("1. Basic OLS (unweighted)", model1),
    ("2. Basic WLS (weighted)", model2),
    ("3. Basic OLS (robust SE)", model3),
    ("4. With demographics (robust SE)", model4),
    ("5. With year FE (robust SE)", model5),
    ("6. With year & state FE (robust SE)", model6),
]

for name, model in models_summary:
    coef = model.params['ELIGIBLE_AFTER']
    se = model.bse['ELIGIBLE_AFTER']
    pval = model.pvalues['ELIGIBLE_AFTER']
    print(f"{name:<50} {coef:>12.4f} {se:>12.4f} {pval:>10.4f}")

# ============================================================================
# 10. PREFERRED SPECIFICATION
# ============================================================================

print("\n10. PREFERRED SPECIFICATION DETAILS")
print("-" * 40)

# Model 6 is the preferred specification
preferred_model = model6
print("\nPreferred Model: DiD with demographic covariates, year and state fixed effects")
print(f"\nTreatment Effect (ELIGIBLE_AFTER): {preferred_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {preferred_model.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{preferred_model.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {preferred_model.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"t-statistic: {preferred_model.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {preferred_model.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"\nSample size: {int(preferred_model.nobs):,}")
print(f"R-squared: {preferred_model.rsquared:.4f}")

# ============================================================================
# 11. ADDITIONAL DESCRIPTIVE STATISTICS FOR REPORT
# ============================================================================

print("\n11. ADDITIONAL DESCRIPTIVE STATISTICS")
print("-" * 40)

# Sample sizes by group
print("\nSample sizes by ELIGIBLE and AFTER:")
sample_sizes = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
print(sample_sizes)

# Demographics by treatment group
print("\nDemographic characteristics by treatment group:")
print("\nTreated (ELIGIBLE=1):")
treated = df[df['ELIGIBLE']==1]
print(f"  Mean age: {treated['AGE'].mean():.2f}")
print(f"  Female: {treated['FEMALE'].mean()*100:.1f}%")
print(f"  Married: {treated['MARRIED'].mean()*100:.1f}%")
print(f"  HS degree or higher: {(treated['ED_HS'] | treated['ED_SOMECOLL'] | treated['ED_TWOYEAR'] | treated['ED_BA']).mean()*100:.1f}%")

print("\nControl (ELIGIBLE=0):")
control = df[df['ELIGIBLE']==0]
print(f"  Mean age: {control['AGE'].mean():.2f}")
print(f"  Female: {control['FEMALE'].mean()*100:.1f}%")
print(f"  Married: {control['MARRIED'].mean()*100:.1f}%")
print(f"  HS degree or higher: {(control['ED_HS'] | control['ED_SOMECOLL'] | control['ED_TWOYEAR'] | control['ED_BA']).mean()*100:.1f}%")

# Employment status breakdown
print("\nEmployment Status by Group (before period):")
before = df[df['AFTER']==0]
print("\nTreated (26-30 in 2012), Before DACA:")
treated_before = before[before['ELIGIBLE']==1]
print(f"  Full-time: {treated_before['FT'].mean()*100:.1f}%")
print(f"  Employed (EMPSTAT=1): {(treated_before['EMPSTAT']==1).mean()*100:.1f}%")
print(f"  Unemployed (EMPSTAT=2): {(treated_before['EMPSTAT']==2).mean()*100:.1f}%")
print(f"  Not in labor force (EMPSTAT=3): {(treated_before['EMPSTAT']==3).mean()*100:.1f}%")

print("\nControl (31-35 in 2012), Before DACA:")
control_before = before[before['ELIGIBLE']==0]
print(f"  Full-time: {control_before['FT'].mean()*100:.1f}%")
print(f"  Employed (EMPSTAT=1): {(control_before['EMPSTAT']==1).mean()*100:.1f}%")
print(f"  Unemployed (EMPSTAT=2): {(control_before['EMPSTAT']==2).mean()*100:.1f}%")
print(f"  Not in labor force (EMPSTAT=3): {(control_before['EMPSTAT']==3).mean()*100:.1f}%")

# By year
print("\nFull-time employment rate by year:")
for year in sorted(df['YEAR'].unique()):
    year_data = df[df['YEAR']==year]
    treated_rate = year_data[year_data['ELIGIBLE']==1]['FT'].mean()
    control_rate = year_data[year_data['ELIGIBLE']==0]['FT'].mean()
    diff = treated_rate - control_rate
    print(f"  {year}: Treated={treated_rate:.3f}, Control={control_rate:.3f}, Diff={diff:.3f}")

# ============================================================================
# 12. SAVE RESULTS FOR LATEX TABLE
# ============================================================================

print("\n12. RESULTS FOR LATEX TABLES")
print("-" * 40)

# Create summary table
results_df = pd.DataFrame({
    'Model': [
        '(1) Basic OLS',
        '(2) Basic WLS',
        '(3) Robust SE',
        '(4) + Demographics',
        '(5) + Year FE',
        '(6) + State FE'
    ],
    'DiD_Estimate': [m.params['ELIGIBLE_AFTER'] for _, m in models_summary],
    'Std_Error': [m.bse['ELIGIBLE_AFTER'] for _, m in models_summary],
    'p_value': [m.pvalues['ELIGIBLE_AFTER'] for _, m in models_summary],
    'N': [int(m.nobs) for _, m in models_summary],
    'R_squared': [m.rsquared for _, m in models_summary]
})

results_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to regression_results.csv")

# Save cell means for DiD table
cell_means = pd.DataFrame({
    'Group': ['Treated (26-30)', 'Control (31-35)'],
    'Before': [mean_10, mean_00],
    'After': [mean_11, mean_01],
    'Difference': [mean_11 - mean_10, mean_01 - mean_00]
})
cell_means.to_csv('did_cell_means.csv', index=False)
print("DiD cell means saved to did_cell_means.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

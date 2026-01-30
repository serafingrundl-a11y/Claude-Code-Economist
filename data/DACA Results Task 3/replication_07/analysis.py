"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment
Methodology: Difference-in-Differences
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("=" * 80)
print("DACA REPLICATION STUDY - DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 80)

# Load data
print("\n1. LOADING DATA")
print("-" * 40)
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
print(f"Total observations: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Basic data checks
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Check years
print("\nYears in dataset:")
print(df['YEAR'].value_counts().sort_index())

# Check ELIGIBLE variable
print("\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts())

# Check AFTER variable
print("\nAFTER distribution:")
print(df['AFTER'].value_counts())

# Check FT variable
print("\nFT (Full-time employment) distribution:")
print(df['FT'].value_counts())

# Create 2x2 table
print("\n3. DIFFERENCE-IN-DIFFERENCES SETUP")
print("-" * 40)

# Create the 2x2 table of means
print("\nMean Full-Time Employment by Group and Period:")
table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'sum'])
print(table)

# Calculate simple DID
mean_treat_post = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()
mean_treat_pre = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
mean_control_post = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()
mean_control_pre = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()

print(f"\nTreatment group (ELIGIBLE=1):")
print(f"  Pre-DACA (2008-2011):  {mean_treat_pre:.4f}")
print(f"  Post-DACA (2013-2016): {mean_treat_post:.4f}")
print(f"  Change:                {mean_treat_post - mean_treat_pre:.4f}")

print(f"\nControl group (ELIGIBLE=0):")
print(f"  Pre-DACA (2008-2011):  {mean_control_pre:.4f}")
print(f"  Post-DACA (2013-2016): {mean_control_post:.4f}")
print(f"  Change:                {mean_control_post - mean_control_pre:.4f}")

simple_did = (mean_treat_post - mean_treat_pre) - (mean_control_post - mean_control_pre)
print(f"\nSimple DID estimate: {simple_did:.4f}")

# Regression Analysis
print("\n4. REGRESSION ANALYSIS")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DID (unweighted)
print("\nModel 1: Basic DID (Unweighted OLS)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: Basic DID (weighted)
print("\nModel 2: Basic DID (Weighted OLS using PERWT)")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])

# Model 3: DID with clustered standard errors at state level
print("\nModel 3: Basic DID with Clustered SE (State Level)")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model3.summary().tables[1])

# Model 4: Weighted DID with clustered SE
print("\nModel 4: Weighted DID with Clustered SE (State Level)")
# For WLS with clustered SE, we need to use the WLS model with a robust covariance
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model4.summary().tables[1])

# Add covariates
print("\n5. DID WITH DEMOGRAPHIC CONTROLS")
print("-" * 40)

# Check for missing values in potential covariates
print("\nChecking covariate availability:")
covariates = ['SEX', 'MARST', 'NCHILD', 'EDUC', 'STATEFIP']
for var in covariates:
    missing = df[var].isna().sum()
    print(f"  {var}: {missing} missing values")

# Model 5: DID with demographic controls
print("\nModel 5: DID with Demographic Controls (Weighted, Clustered SE)")

# Prepare categorical variables
df['SEX_female'] = (df['SEX'] == 2).astype(int)

# Education categories (EDUC codes from IPUMS)
# 0=NA, 1=None/Preschool, 2-5=Elem, 6=High School, 7-9=Some college, 10=BA, 11=Prof/Grad
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_somecol'] = df['EDUC'].isin([7, 8, 9]).astype(int)
df['educ_ba_plus'] = df['EDUC'].isin([10, 11]).astype(int)

# Marital status (1=married spouse present)
df['married'] = (df['MARST'] == 1).astype(int)

# Has children
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Model with controls
model5_formula = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + SEX_female + married + has_children + educ_hs + educ_somecol + educ_ba_plus'
model5 = smf.wls(model5_formula, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(model5.summary().tables[1])

# Model 6: DID with state fixed effects
print("\n6. DID WITH STATE FIXED EFFECTS")
print("-" * 40)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

# Build formula with state fixed effects
state_cols = [col for col in state_dummies.columns]
state_formula = ' + '.join(state_cols)
model6_formula = f'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + SEX_female + married + has_children + educ_hs + educ_somecol + educ_ba_plus + {state_formula}'

model6 = smf.wls(model6_formula, data=df_with_states, weights=df_with_states['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_with_states['STATEFIP']})
print("\nModel 6: DID with State FE, Demographics (Weighted, Clustered SE)")
print("Key coefficients:")
for var in ['Intercept', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'SEX_female', 'married', 'has_children', 'educ_hs', 'educ_somecol', 'educ_ba_plus']:
    if var in model6.params.index:
        print(f"  {var:20s}: {model6.params[var]:8.4f} (SE: {model6.bse[var]:.4f}, p={model6.pvalues[var]:.4f})")

# Model 7: With year fixed effects instead of AFTER dummy
print("\n7. DID WITH YEAR FIXED EFFECTS")
print("-" * 40)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)
year_cols = [col for col in year_dummies.columns]
year_formula = ' + '.join(year_cols)

model7_formula = f'FT ~ ELIGIBLE + ELIGIBLE_AFTER + SEX_female + married + has_children + educ_hs + educ_somecol + educ_ba_plus + {year_formula}'
model7 = smf.wls(model7_formula, data=df_with_years, weights=df_with_years['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_with_years['STATEFIP']})
print("\nModel 7: DID with Year FE, Demographics (Weighted, Clustered SE)")
print("Key coefficients:")
for var in ['Intercept', 'ELIGIBLE', 'ELIGIBLE_AFTER', 'SEX_female', 'married', 'has_children', 'educ_hs', 'educ_somecol', 'educ_ba_plus']:
    if var in model7.params.index:
        print(f"  {var:20s}: {model7.params[var]:8.4f} (SE: {model7.bse[var]:.4f}, p={model7.pvalues[var]:.4f})")

# Additional analysis - trends by year
print("\n8. TRENDS ANALYSIS")
print("-" * 40)

# Calculate FT rates by year and group
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'ft_rate': np.average(x['FT'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nFull-Time Employment Rate by Year and Group:")
print(trends.pivot(index='YEAR', columns='ELIGIBLE', values='ft_rate'))

# Subgroup analysis by sex
print("\n9. SUBGROUP ANALYSIS BY SEX")
print("-" * 40)

for sex_val, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub, weights=df_sub['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': df_sub['STATEFIP']})
    print(f"\n{sex_label}s (n={len(df_sub):,}):")
    print(f"  DID Estimate: {model_sub.params['ELIGIBLE_AFTER']:.4f}")
    print(f"  SE: {model_sub.bse['ELIGIBLE_AFTER']:.4f}")
    print(f"  95% CI: [{model_sub.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_sub.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
    print(f"  p-value: {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}")

# Summary statistics for report
print("\n10. SUMMARY STATISTICS FOR REPORT")
print("-" * 40)

print("\nSample Statistics:")
print(f"  Total N: {len(df):,}")
print(f"  Treatment group (ELIGIBLE=1): {(df['ELIGIBLE']==1).sum():,}")
print(f"  Control group (ELIGIBLE=0): {(df['ELIGIBLE']==0).sum():,}")
print(f"  Pre-period observations: {(df['AFTER']==0).sum():,}")
print(f"  Post-period observations: {(df['AFTER']==1).sum():,}")

print("\nOverall full-time employment rate:")
print(f"  Unweighted: {df['FT'].mean():.4f}")
print(f"  Weighted: {np.average(df['FT'], weights=df['PERWT']):.4f}")

print("\nDemographic Summary (weighted):")
print(f"  Female proportion: {np.average(df['SEX']==2, weights=df['PERWT']):.4f}")
print(f"  Married proportion: {np.average(df['MARST']==1, weights=df['PERWT']):.4f}")
print(f"  Has children: {np.average(df['NCHILD']>0, weights=df['PERWT']):.4f}")

# Age distribution
print("\nAge distribution by group (AGE_IN_JUNE_2012):")
for eligible in [0, 1]:
    subset = df[df['ELIGIBLE'] == eligible]
    label = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
    print(f"  {label}: mean = {subset['AGE_IN_JUNE_2012'].mean():.2f}, min = {subset['AGE_IN_JUNE_2012'].min()}, max = {subset['AGE_IN_JUNE_2012'].max()}")

# Save key results
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE (Model 4: Weighted DID with Clustered SE)")
print("=" * 80)
print(f"\nDID Coefficient (ELIGIBLE × AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {len(df):,}")
print(f"R-squared: {model4.rsquared:.4f}")

# Export results for LaTeX
results_dict = {
    'Model': ['(1) OLS', '(2) WLS', '(3) OLS Clustered', '(4) WLS Clustered', '(5) WLS + Controls', '(6) State FE', '(7) Year FE'],
    'DID_Estimate': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model6.params['ELIGIBLE_AFTER'],
        model7.params['ELIGIBLE_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model6.bse['ELIGIBLE_AFTER'],
        model7.bse['ELIGIBLE_AFTER']
    ],
    'p_value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER'],
        model6.pvalues['ELIGIBLE_AFTER'],
        model7.pvalues['ELIGIBLE_AFTER']
    ],
    'N': [len(df)] * 7,
    'R2': [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared,
        model5.rsquared,
        model6.rsquared,
        model7.rsquared
    ],
    'Weights': ['No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Clustered_SE': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Controls': ['No', 'No', 'No', 'No', 'Yes', 'Yes + State FE', 'Yes + Year FE']
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("\nResults saved to regression_results.csv")

# Create trend data for visualization
trends.to_csv('trends_data.csv', index=False)
print("Trends data saved to trends_data.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

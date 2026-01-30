"""
DACA Replication Study: Effect of DACA on Full-Time Employment
Difference-in-Differences Analysis

Research Question: Among ethnically Hispanic-Mexican Mexican-born people living in the United States,
what was the causal impact of eligibility for DACA (treatment) on the probability of full-time employment?

Treatment Group: Eligible people aged 26-30 at the time DACA went into place (June 2012)
Control Group: People aged 31-35 at the time, who otherwise would have been eligible if not for age
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS LOG")
print("=" * 80)

data_path = r"C:\Users\seraf\DACA Results Task 3\replication_50\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {len(df)} observations")
print(f"Years in dataset: {sorted(df['YEAR'].unique())}")
print(f"ELIGIBLE variable values: {df['ELIGIBLE'].unique()}")
print(f"AFTER variable values: {df['AFTER'].unique()}")
print(f"FT variable values: {df['FT'].unique()}")

# Basic data exploration
print("\n" + "=" * 80)
print("1. DATA EXPLORATION")
print("=" * 80)

print("\nSample size by ELIGIBLE and AFTER:")
print(df.groupby(['ELIGIBLE', 'AFTER']).size().unstack())

print("\nSample size by Year and ELIGIBLE:")
print(df.groupby(['YEAR', 'ELIGIBLE']).size().unstack())

# Summary statistics
print("\n" + "=" * 80)
print("2. SUMMARY STATISTICS")
print("=" * 80)

# Key outcome variable
print("\nFull-time employment (FT) by group and period:")
ft_summary = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(ft_summary)

# Calculate the simple DiD estimate
print("\n" + "=" * 80)
print("3. SIMPLE DIFFERENCE-IN-DIFFERENCES ESTIMATE")
print("=" * 80)

# Calculate group means
treated_before = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'].mean()
treated_after = df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'].mean()
control_before = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'].mean()
control_after = df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'].mean()

print(f"\nTreated (Eligible, ages 26-30) Before: {treated_before:.4f}")
print(f"Treated (Eligible, ages 26-30) After:  {treated_after:.4f}")
print(f"Control (Not Eligible, ages 31-35) Before: {control_before:.4f}")
print(f"Control (Not Eligible, ages 31-35) After:  {control_after:.4f}")

# DiD estimate
treated_diff = treated_after - treated_before
control_diff = control_after - control_before
did_estimate = treated_diff - control_diff

print(f"\nChange in treated group: {treated_diff:.4f}")
print(f"Change in control group: {control_diff:.4f}")
print(f"\nDifference-in-Differences Estimate: {did_estimate:.4f}")

# 4. REGRESSION ANALYSIS
print("\n" + "=" * 80)
print("4. REGRESSION ANALYSIS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD regression (no covariates)
print("\n--- Model 1: Basic DiD (no covariates) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])

# Model 2: DiD with robust standard errors
print("\n--- Model 2: DiD with robust standard errors ---")
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model2.summary().tables[1])

# Model 3: DiD with demographic controls
print("\n--- Model 3: DiD with demographic controls ---")
# Check which variables are available and appropriate
print("\nAvailable control variables:")
print(f"SEX values: {df['SEX'].unique()}")
print(f"MARST values: {sorted(df['MARST'].unique())}")

# Create control variables
df['MALE'] = (df['SEX'] == 1).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)  # Married (spouse present or absent)

# Check EDUC_RECODE
print(f"\nEDUC_RECODE values: {df['EDUC_RECODE'].unique()}")

# Create education dummies
df['HS_OR_LESS'] = df['EDUC_RECODE'].isin(['Less than High School', 'High School Degree']).astype(int)
df['SOME_COLLEGE'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['TWO_YEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['BA_PLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)

# Model with controls
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE',
                 data=df).fit(cov_type='HC1')
print(model3.summary().tables[1])

# Model 4: DiD with year fixed effects
print("\n--- Model 4: DiD with year fixed effects ---")
# Create year dummies (excluding one as reference)
year_dummies = pd.get_dummies(df['YEAR'], prefix='Y', drop_first=True)
df = pd.concat([df, year_dummies], axis=1)

# Build model with year FE
year_cols = [col for col in df.columns if col.startswith('Y_')]
formula4 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE + ' + ' + '.join(year_cols)
model4 = smf.ols(formula4, data=df).fit(cov_type='HC1')

print("\nKey coefficients:")
print(f"ELIGIBLE: {model4.params['ELIGIBLE']:.4f} (SE: {model4.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER (DiD estimate): {model4.params['ELIGIBLE_AFTER']:.4f} (SE: {model4.bse['ELIGIBLE_AFTER']:.4f})")

# Model 5: DiD with state fixed effects
print("\n--- Model 5: DiD with state and year fixed effects ---")
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='ST', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

state_cols = [col for col in df.columns if col.startswith('ST_')]
formula5 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
model5 = smf.ols(formula5, data=df).fit(cov_type='HC1')

print("\nKey coefficients:")
print(f"ELIGIBLE: {model5.params['ELIGIBLE']:.4f} (SE: {model5.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER (DiD estimate): {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")

# 5. PARALLEL TRENDS CHECK
print("\n" + "=" * 80)
print("5. PARALLEL TRENDS CHECK")
print("=" * 80)

# Calculate yearly means by group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_means.columns = ['Control (31-35)', 'Treated (26-30)']
print("\nYearly Full-Time Employment Rates by Group:")
print(yearly_means)

# Calculate pre-trends
pre_data = df[df['AFTER'] == 0].copy()
pre_data['YEAR_ELIGIBLE'] = pre_data['YEAR'] * pre_data['ELIGIBLE']
pre_model = smf.ols('FT ~ ELIGIBLE + YEAR + YEAR_ELIGIBLE', data=pre_data).fit(cov_type='HC1')
print("\nPre-trend test (interaction of ELIGIBLE with YEAR in pre-period):")
print(f"YEAR:ELIGIBLE coefficient: {pre_model.params['YEAR_ELIGIBLE']:.6f}")
print(f"p-value: {pre_model.pvalues['YEAR_ELIGIBLE']:.4f}")

# 6. EVENT STUDY
print("\n" + "=" * 80)
print("6. EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year-specific treatment effects (relative to 2011)
years_in_data = sorted(df['YEAR'].unique())
print(f"\nYears in data: {years_in_data}")

# Create interactions for each year with ELIGIBLE
for year in years_in_data:
    df[f'YEAR_{year}_ELIGIBLE'] = ((df['YEAR'] == year) & (df['ELIGIBLE'] == 1)).astype(int)

# Exclude 2011 as reference year
year_interaction_cols = [f'YEAR_{y}_ELIGIBLE' for y in years_in_data if y != 2011]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(year_interaction_cols)
model_event = smf.ols(formula_event, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (relative to 2011):")
for year in sorted(years_in_data):
    if year != 2011:
        col = f'YEAR_{year}_ELIGIBLE'
        print(f"Year {year}: {model_event.params[col]:.4f} (SE: {model_event.bse[col]:.4f}, p={model_event.pvalues[col]:.4f})")

# 7. ROBUSTNESS CHECKS
print("\n" + "=" * 80)
print("7. ROBUSTNESS CHECKS")
print("=" * 80)

# 7a. Weighted regression using person weights
print("\n--- 7a. Weighted regression (using PERWT) ---")
model_weighted = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED + NCHILD + AGE',
                         data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"ELIGIBLE_AFTER (DiD estimate): {model_weighted.params['ELIGIBLE_AFTER']:.4f} (SE: {model_weighted.bse['ELIGIBLE_AFTER']:.4f})")

# 7b. By gender
print("\n--- 7b. By gender ---")
for sex, sex_label in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sex).fit(cov_type='HC1')
    print(f"{sex_label}: DiD = {model_sex.params['ELIGIBLE_AFTER']:.4f} (SE: {model_sex.bse['ELIGIBLE_AFTER']:.4f})")

# 7c. Probit model (for binary outcome)
print("\n--- 7c. Probit model ---")
try:
    probit_model = smf.probit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED', data=df).fit(disp=0)
    print(f"ELIGIBLE_AFTER coefficient: {probit_model.params['ELIGIBLE_AFTER']:.4f} (SE: {probit_model.bse['ELIGIBLE_AFTER']:.4f})")
    # Calculate marginal effect
    print("(Note: Probit coefficients are not directly comparable to linear probability model)")
except Exception as e:
    print(f"Probit model failed: {e}")

# 7d. Logit model
print("\n--- 7d. Logit model ---")
try:
    logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + MALE + MARRIED', data=df).fit(disp=0)
    print(f"ELIGIBLE_AFTER coefficient: {logit_model.params['ELIGIBLE_AFTER']:.4f} (SE: {logit_model.bse['ELIGIBLE_AFTER']:.4f})")
    # Calculate odds ratio
    odds_ratio = np.exp(logit_model.params['ELIGIBLE_AFTER'])
    print(f"Odds ratio: {odds_ratio:.4f}")
except Exception as e:
    print(f"Logit model failed: {e}")

# 8. SAMPLE CHARACTERISTICS
print("\n" + "=" * 80)
print("8. SAMPLE CHARACTERISTICS")
print("=" * 80)

print("\nDemographic characteristics by treatment group:")
for eligible in [0, 1]:
    print(f"\n{'Treated (26-30)' if eligible == 1 else 'Control (31-35)'} group:")
    subset = df[df['ELIGIBLE'] == eligible]
    print(f"  N = {len(subset)}")
    print(f"  Mean Age: {subset['AGE'].mean():.2f}")
    print(f"  Male %: {(subset['SEX'] == 1).mean() * 100:.1f}%")
    print(f"  Married %: {subset['MARRIED'].mean() * 100:.1f}%")
    print(f"  Mean children: {subset['NCHILD'].mean():.2f}")
    print(f"  Full-time employment %: {subset['FT'].mean() * 100:.1f}%")

# 9. PREFERRED ESTIMATE SUMMARY
print("\n" + "=" * 80)
print("9. PREFERRED ESTIMATE SUMMARY")
print("=" * 80)

# Use Model 4 (year FE, demographic controls) as preferred specification
print("\nPreferred Specification: DiD with year fixed effects and demographic controls")
print(f"\nEffect Size: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
ci_low = model4.params['ELIGIBLE_AFTER'] - 1.96 * model4.bse['ELIGIBLE_AFTER']
ci_high = model4.params['ELIGIBLE_AFTER'] + 1.96 * model4.bse['ELIGIBLE_AFTER']
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {model4.nobs:.0f}")

# Save key results to file for LaTeX report
results = {
    'n_total': len(df),
    'n_treated_before': len(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]),
    'n_treated_after': len(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]),
    'n_control_before': len(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]),
    'n_control_after': len(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]),
    'ft_treated_before': treated_before,
    'ft_treated_after': treated_after,
    'ft_control_before': control_before,
    'ft_control_after': control_after,
    'did_simple': did_estimate,
    'did_model1_coef': model1.params['ELIGIBLE_AFTER'],
    'did_model1_se': model1.bse['ELIGIBLE_AFTER'],
    'did_model2_coef': model2.params['ELIGIBLE_AFTER'],
    'did_model2_se': model2.bse['ELIGIBLE_AFTER'],
    'did_model3_coef': model3.params['ELIGIBLE_AFTER'],
    'did_model3_se': model3.bse['ELIGIBLE_AFTER'],
    'did_model4_coef': model4.params['ELIGIBLE_AFTER'],
    'did_model4_se': model4.bse['ELIGIBLE_AFTER'],
    'did_model5_coef': model5.params['ELIGIBLE_AFTER'],
    'did_model5_se': model5.bse['ELIGIBLE_AFTER'],
    'did_weighted_coef': model_weighted.params['ELIGIBLE_AFTER'],
    'did_weighted_se': model_weighted.bse['ELIGIBLE_AFTER'],
}

import json
with open(r'C:\Users\seraf\DACA Results Task 3\replication_50\results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nResults saved to results.json")

# Save model summaries
print("\n" + "=" * 80)
print("FULL MODEL SUMMARIES")
print("=" * 80)

print("\n--- Full Model 1 Summary (Basic DiD) ---")
print(model1.summary())

print("\n--- Full Model 4 Summary (Preferred: Year FE + Controls) ---")
print(model4.summary())

# Create visualization data
print("\n" + "=" * 80)
print("VISUALIZATION DATA")
print("=" * 80)

yearly_data = df.groupby(['YEAR', 'ELIGIBLE']).agg({
    'FT': ['mean', 'std', 'count']
}).reset_index()
yearly_data.columns = ['Year', 'Eligible', 'FT_Mean', 'FT_Std', 'N']
yearly_data['SE'] = yearly_data['FT_Std'] / np.sqrt(yearly_data['N'])

print("\nYearly data for parallel trends plot:")
print(yearly_data.to_string(index=False))

yearly_data.to_csv(r'C:\Users\seraf\DACA Results Task 3\replication_50\yearly_means.csv', index=False)
print("\nYearly means saved to yearly_means.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

"""
DACA Effect on Full-Time Employment: Replication Analysis
=========================================================
This script implements a difference-in-differences (DiD) analysis to estimate the
causal effect of DACA eligibility on full-time employment among Hispanic-Mexican,
Mexican-born individuals in the United States.

Research Design:
- Treatment group: DACA-eligible individuals aged 26-30 as of June 15, 2012
- Control group: Individuals aged 31-35 as of June 15, 2012 (otherwise eligible if not for age)
- Pre-period: 2008-2011
- Post-period: 2013-2016
- Outcome: Full-time employment (working 35+ hours per week)
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

#%% Load Data
print("=" * 80)
print("DACA EFFECT ON FULL-TIME EMPLOYMENT: REPLICATION ANALYSIS")
print("=" * 80)

# Load the data
data = pd.read_csv('data/prepared_data_numeric_version.csv')
data_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

print(f"\nDataset dimensions: {data.shape[0]} observations, {data.shape[1]} variables")

#%% Explore Key Variables
print("\n" + "=" * 80)
print("DATA EXPLORATION")
print("=" * 80)

# Check years in data
print("\nYears in data:")
print(data['YEAR'].value_counts().sort_index())

# Check ELIGIBLE variable
print("\nELIGIBLE variable distribution:")
print(data['ELIGIBLE'].value_counts())
print(f"  - Eligible (treatment group): {(data['ELIGIBLE'] == 1).sum()}")
print(f"  - Control group: {(data['ELIGIBLE'] == 0).sum()}")

# Check AFTER variable
print("\nAFTER variable distribution:")
print(data['AFTER'].value_counts())
print(f"  - Pre-DACA (2008-2011): {(data['AFTER'] == 0).sum()}")
print(f"  - Post-DACA (2013-2016): {(data['AFTER'] == 1).sum()}")

# Check FT variable
print("\nFT (Full-Time Employment) distribution:")
print(data['FT'].value_counts())
print(f"  - Full-time employed: {(data['FT'] == 1).sum()}")
print(f"  - Not full-time employed: {(data['FT'] == 0).sum()}")
print(f"  - Overall FT rate: {data['FT'].mean():.4f}")

# Age distribution at June 2012
print("\nAGE_IN_JUNE_2012 distribution:")
print(data['AGE_IN_JUNE_2012'].describe())

# Check age groups match expectations
print("\nAge groups by ELIGIBLE status (summary):")
eligible_ages = data[data['ELIGIBLE'] == 1]['AGE_IN_JUNE_2012']
control_ages = data[data['ELIGIBLE'] == 0]['AGE_IN_JUNE_2012']
print(f"Eligible group (ages 26-30): min={eligible_ages.min():.2f}, max={eligible_ages.max():.2f}, mean={eligible_ages.mean():.2f}")
print(f"Control group (ages 31-35): min={control_ages.min():.2f}, max={control_ages.max():.2f}, mean={control_ages.mean():.2f}")

#%% Summary Statistics Table
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Create summary statistics by group and period
summary_data = []
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = data[(data['ELIGIBLE'] == eligible) & (data['AFTER'] == after)]
        group_name = 'Treatment' if eligible == 1 else 'Control'
        period = 'Post' if after == 1 else 'Pre'

        summary_data.append({
            'Group': group_name,
            'Period': period,
            'N': len(subset),
            'FT_Rate': subset['FT'].mean(),
            'FT_SE': subset['FT'].std() / np.sqrt(len(subset)),
            'Mean_Age': subset['AGE'].mean(),
            'Pct_Male': (subset['SEX'] == 1).mean() * 100,
            'Pct_Married': (subset['MARST'] == 1).mean() * 100,
        })

summary_df = pd.DataFrame(summary_data)
print("\nSummary Statistics by Group and Period:")
print(summary_df.to_string(index=False))

# Calculate simple DiD estimate
pre_treat = data[(data['ELIGIBLE'] == 1) & (data['AFTER'] == 0)]['FT'].mean()
post_treat = data[(data['ELIGIBLE'] == 1) & (data['AFTER'] == 1)]['FT'].mean()
pre_control = data[(data['ELIGIBLE'] == 0) & (data['AFTER'] == 0)]['FT'].mean()
post_control = data[(data['ELIGIBLE'] == 0) & (data['AFTER'] == 1)]['FT'].mean()

did_simple = (post_treat - pre_treat) - (post_control - pre_control)

print("\n" + "-" * 50)
print("SIMPLE DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("-" * 50)
print(f"Treatment group (ages 26-30):")
print(f"  Pre-DACA FT rate:  {pre_treat:.4f}")
print(f"  Post-DACA FT rate: {post_treat:.4f}")
print(f"  Change:            {post_treat - pre_treat:.4f}")
print(f"\nControl group (ages 31-35):")
print(f"  Pre-DACA FT rate:  {pre_control:.4f}")
print(f"  Post-DACA FT rate: {post_control:.4f}")
print(f"  Change:            {post_control - pre_control:.4f}")
print(f"\nDiD estimate (simple):  {did_simple:.4f}")

#%% Prepare Variables for Regression
print("\n" + "=" * 80)
print("PREPARING VARIABLES FOR REGRESSION")
print("=" * 80)

# Create interaction term
data['ELIGIBLE_X_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Create dummy variables for covariates
# SEX: 1=Male, 2=Female
data['MALE'] = (data['SEX'] == 1).astype(int)

# MARST: 1=Married spouse present
data['MARRIED'] = (data['MARST'] == 1).astype(int)

# Education: Create clean dummy variable names
print("\nEducation categories from labelled data:")
print(data_labels['EDUC_RECODE'].value_counts())

# Handle missing values in education
data_labels['EDUC_RECODE'] = data_labels['EDUC_RECODE'].fillna('Unknown')

# Create numeric codes for education
educ_categories = sorted([x for x in data_labels['EDUC_RECODE'].unique() if pd.notna(x)])
educ_map = {val: i for i, val in enumerate(educ_categories)}
data['EDUC_CODE'] = data_labels['EDUC_RECODE'].map(educ_map)
print(f"\nEducation categories: {educ_categories}")

for i, cat in enumerate(educ_categories[1:], 1):  # Skip first as reference
    clean_name = f'EDUC_{i}'
    data[clean_name] = (data['EDUC_CODE'] == i).astype(int)

educ_cols = [f'EDUC_{i}' for i in range(1, len(educ_categories))]
print(f"Education dummy columns created: {educ_cols}")

# Create year dummies (reference: 2008)
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    data[f'YEAR_{year}'] = (data['YEAR'] == year).astype(int)
year_cols = [f'YEAR_{year}' for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]]

# Create state dummies (reference: first state)
states = sorted(data['STATEFIP'].unique())
for state in states[1:]:  # Skip first as reference
    data[f'STATE_{state}'] = (data['STATEFIP'] == state).astype(int)
state_cols = [f'STATE_{state}' for state in states[1:]]

#%% Main DiD Regression Analysis
print("\n" + "=" * 80)
print("MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("=" * 80)

# Model 1: Basic DiD without weights
print("\n--- Model 1: Basic DiD (unweighted, no controls) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=data).fit()
print(model1.summary())

# Model 2: DiD with survey weights
print("\n--- Model 2: DiD with survey weights (PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                 data=data, weights=data['PERWT']).fit()
print(model2.summary())

# Model 3: DiD with robust standard errors
print("\n--- Model 3: DiD with robust (HC1) standard errors ---")
model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=data).fit(cov_type='HC1')
print(model3.summary())

#%% Add Covariates
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES WITH COVARIATES")
print("=" * 80)

# Model 4: DiD with demographic controls
print("\n--- Model 4: DiD with demographic controls ---")
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + MALE + MARRIED + AGE'
model4 = smf.ols(formula4, data=data).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with demographic and education controls
print("\n--- Model 5: DiD with demographic and education controls ---")
# Use matrix-based approach to avoid formula parsing issues
y = data['FT']
X_basic = data[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'AGE']].copy()
X_educ = data[educ_cols].copy()
X5 = pd.concat([X_basic, X_educ], axis=1)
X5 = sm.add_constant(X5)
model5 = sm.OLS(y, X5).fit(cov_type='HC1')
print(model5.summary())

#%% Year Fixed Effects
print("\n" + "=" * 80)
print("MODELS WITH YEAR FIXED EFFECTS")
print("=" * 80)

# Model 6: DiD with year fixed effects
print("\n--- Model 6: DiD with year fixed effects ---")
X6 = data[['ELIGIBLE', 'ELIGIBLE_X_AFTER'] + year_cols].copy()
X6 = sm.add_constant(X6)
model6 = sm.OLS(y, X6).fit(cov_type='HC1')
print(model6.summary())

# Model 7: Full model with year FE and covariates
print("\n--- Model 7: DiD with year FE and controls ---")
X7 = data[['ELIGIBLE', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'AGE'] + educ_cols + year_cols].copy()
X7 = sm.add_constant(X7)
model7 = sm.OLS(y, X7).fit(cov_type='HC1')
print(model7.summary())

#%% State Fixed Effects
print("\n" + "=" * 80)
print("MODELS WITH STATE FIXED EFFECTS")
print("=" * 80)

# Model 8: DiD with state and year fixed effects
print("\n--- Model 8: DiD with state and year fixed effects ---")
X8 = data[['ELIGIBLE', 'ELIGIBLE_X_AFTER'] + year_cols + state_cols].copy()
X8 = sm.add_constant(X8)
model8 = sm.OLS(y, X8).fit(cov_type='HC1')
print("Key coefficients:")
print(f"  ELIGIBLE_X_AFTER: {model8.params['ELIGIBLE_X_AFTER']:.5f} (SE: {model8.bse['ELIGIBLE_X_AFTER']:.5f})")
print(f"  t-stat: {model8.tvalues['ELIGIBLE_X_AFTER']:.3f}, p-value: {model8.pvalues['ELIGIBLE_X_AFTER']:.4f}")
ci8 = model8.conf_int().loc['ELIGIBLE_X_AFTER']
print(f"  95% CI: [{ci8[0]:.5f}, {ci8[1]:.5f}]")

# Model 9: Full model with state FE, year FE, and covariates
print("\n--- Model 9: Full model with state FE, year FE, and controls ---")
X9 = data[['ELIGIBLE', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'AGE'] + educ_cols + year_cols + state_cols].copy()
X9 = sm.add_constant(X9)
model9 = sm.OLS(y, X9).fit(cov_type='HC1')
print("Key coefficients:")
print(f"  ELIGIBLE_X_AFTER: {model9.params['ELIGIBLE_X_AFTER']:.5f} (SE: {model9.bse['ELIGIBLE_X_AFTER']:.5f})")
print(f"  t-stat: {model9.tvalues['ELIGIBLE_X_AFTER']:.3f}, p-value: {model9.pvalues['ELIGIBLE_X_AFTER']:.4f}")
ci9 = model9.conf_int().loc['ELIGIBLE_X_AFTER']
print(f"  95% CI: [{ci9[0]:.5f}, {ci9[1]:.5f}]")
print(f"  R-squared: {model9.rsquared:.4f}")
print(f"  N: {model9.nobs:.0f}")

#%% Create Results Summary Table
print("\n" + "=" * 80)
print("RESULTS SUMMARY TABLE")
print("=" * 80)

results_summary = []
models_list = [
    ('Model 1: Basic DiD', model1),
    ('Model 2: Weighted DiD', model2),
    ('Model 3: Robust SE', model3),
    ('Model 4: Demographic controls', model4),
    ('Model 5: + Education', model5),
    ('Model 6: Year FE', model6),
    ('Model 7: Year FE + controls', model7),
    ('Model 8: State + Year FE', model8),
    ('Model 9: Full model', model9),
]

for name, model in models_list:
    coef = model.params['ELIGIBLE_X_AFTER']
    se = model.bse['ELIGIBLE_X_AFTER']
    pval = model.pvalues['ELIGIBLE_X_AFTER']
    ci = model.conf_int().loc['ELIGIBLE_X_AFTER']
    results_summary.append({
        'Model': name,
        'DiD Estimate': f"{coef:.4f}",
        'SE': f"{se:.4f}",
        'p-value': f"{pval:.4f}",
        '95% CI': f"[{ci[0]:.4f}, {ci[1]:.4f}]",
        'N': int(model.nobs)
    })

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

#%% Parallel Trends Analysis
print("\n" + "=" * 80)
print("PARALLEL TRENDS ANALYSIS")
print("=" * 80)

# Calculate FT rates by year and group
trends = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].agg(['mean', 'std', 'count']).reset_index()
trends.columns = ['Year', 'Eligible', 'FT_Rate', 'SD', 'N']
trends['SE'] = trends['SD'] / np.sqrt(trends['N'])

print("\nFull-Time Employment Rates by Year and Group:")
print(trends.to_string(index=False))

# Test for differential pre-trends
print("\nPre-trend analysis (testing for parallel trends in pre-period):")
pre_data = data[data['AFTER'] == 0].copy()
pre_data['YEAR_CENTERED'] = pre_data['YEAR'] - 2008

# Regression of FT on year trend interacted with eligible
pre_data['YEAR_X_ELIGIBLE'] = pre_data['YEAR_CENTERED'] * pre_data['ELIGIBLE']
pretrend_model = smf.ols('FT ~ YEAR_CENTERED + ELIGIBLE + YEAR_X_ELIGIBLE', data=pre_data).fit(cov_type='HC1')
print(pretrend_model.summary())

print(f"\nDifferential pre-trend coefficient (YEAR_X_ELIGIBLE): {pretrend_model.params['YEAR_X_ELIGIBLE']:.5f}")
print(f"  SE: {pretrend_model.bse['YEAR_X_ELIGIBLE']:.5f}")
print(f"  p-value: {pretrend_model.pvalues['YEAR_X_ELIGIBLE']:.4f}")

#%% Event Study Analysis
print("\n" + "=" * 80)
print("EVENT STUDY ANALYSIS")
print("=" * 80)

# Create year-specific treatment effects (event study)
# Reference year: 2011 (last pre-treatment year)
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]

# Create interaction terms for each year (excluding 2011 as reference)
for year in event_years:
    data[f'ELIGIBLE_X_{year}'] = (data['ELIGIBLE'] * (data['YEAR'] == year)).astype(int)

event_cols_reg = [f'ELIGIBLE_X_{year}' for year in event_years]
X_event = data[['ELIGIBLE'] + event_cols_reg + year_cols].copy()
X_event = sm.add_constant(X_event)
event_model = sm.OLS(y, X_event).fit(cov_type='HC1')

print("Event Study Results (Reference: 2011):")
print("-" * 60)
event_results = []
for year in event_years:
    col = f'ELIGIBLE_X_{year}'
    coef = event_model.params[col]
    se = event_model.bse[col]
    pval = event_model.pvalues[col]
    ci = event_model.conf_int().loc[col]
    event_results.append({
        'Year': year,
        'Period': 'Pre' if year < 2012 else 'Post',
        'Coefficient': f"{coef:.4f}",
        'SE': f"{se:.4f}",
        'p-value': f"{pval:.4f}",
        '95% CI': f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    })

event_df = pd.DataFrame(event_results)
print(event_df.to_string(index=False))

#%% Subgroup Analysis
print("\n" + "=" * 80)
print("SUBGROUP ANALYSIS")
print("=" * 80)

# By sex
print("\n--- Subgroup: By Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    subset = data[data['SEX'] == sex_val]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=subset).fit(cov_type='HC1')
    print(f"\n{sex_name}:")
    print(f"  N = {len(subset)}")
    print(f"  DiD estimate: {model_sub.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  SE: {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  p-value: {model_sub.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# By education
print("\n--- Subgroup: By Education Level ---")
educ_labels = {i: cat for i, cat in enumerate(educ_categories)}
for educ_val in sorted(data['EDUC_CODE'].dropna().unique()):
    subset = data[data['EDUC_CODE'] == educ_val]
    if len(subset) > 100:  # Only analyze if sufficient sample size
        model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=subset).fit(cov_type='HC1')
        label = educ_labels.get(int(educ_val), f"Category {int(educ_val)}")
        print(f"\nEducation = {label}:")
        print(f"  N = {len(subset)}")
        print(f"  DiD estimate: {model_sub.params['ELIGIBLE_X_AFTER']:.4f}")
        print(f"  SE: {model_sub.bse['ELIGIBLE_X_AFTER']:.4f}")
        print(f"  p-value: {model_sub.pvalues['ELIGIBLE_X_AFTER']:.4f}")

#%% Robustness Checks
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# 1. Alternative age bandwidth
print("\n--- Robustness 1: Narrower age bandwidth (27-29 vs 32-34) ---")
# Create narrow bandwidth sample (ages are fractional based on birth quarter)
data_narrow = data[
    ((data['AGE_IN_JUNE_2012'] >= 27) & (data['AGE_IN_JUNE_2012'] < 30) & (data['ELIGIBLE'] == 1)) |
    ((data['AGE_IN_JUNE_2012'] >= 32) & (data['AGE_IN_JUNE_2012'] < 35) & (data['ELIGIBLE'] == 0))
]
if len(data_narrow) > 0:
    model_narrow = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=data_narrow).fit(cov_type='HC1')
    print(f"N = {len(data_narrow)}")
    print(f"DiD estimate: {model_narrow.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"SE: {model_narrow.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"p-value: {model_narrow.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 2. Placebo test: Use 2008-2009 vs 2010-2011
print("\n--- Robustness 2: Placebo test (pre-period only, 2010 as fake treatment) ---")
pre_only = data[data['AFTER'] == 0].copy()
pre_only['FAKE_AFTER'] = (pre_only['YEAR'] >= 2010).astype(int)
pre_only['ELIGIBLE_X_FAKE_AFTER'] = pre_only['ELIGIBLE'] * pre_only['FAKE_AFTER']
placebo_model = smf.ols('FT ~ ELIGIBLE + FAKE_AFTER + ELIGIBLE_X_FAKE_AFTER', data=pre_only).fit(cov_type='HC1')
print(f"N = {len(pre_only)}")
print(f"Placebo DiD estimate: {placebo_model.params['ELIGIBLE_X_FAKE_AFTER']:.4f}")
print(f"SE: {placebo_model.bse['ELIGIBLE_X_FAKE_AFTER']:.4f}")
print(f"p-value: {placebo_model.pvalues['ELIGIBLE_X_FAKE_AFTER']:.4f}")

# 3. With state-level controls
print("\n--- Robustness 3: With state-level policy controls ---")
state_policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID',
                     'EVERIFY', 'SECURECOMMUNITIES', 'LFPR', 'UNEMP']
available_policy_vars = [v for v in state_policy_vars if v in data.columns]
print(f"Available policy variables: {available_policy_vars}")

if available_policy_vars:
    data_policy = data.dropna(subset=available_policy_vars)
    y_policy = data_policy['FT']
    X_policy = data_policy[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER'] + available_policy_vars].copy()
    X_policy = sm.add_constant(X_policy)
    model_policy = sm.OLS(y_policy, X_policy).fit(cov_type='HC1')
    print(f"N = {int(model_policy.nobs)}")
    print(f"DiD estimate: {model_policy.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"SE: {model_policy.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"p-value: {model_policy.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# 4. Weighted regression with survey weights
print("\n--- Robustness 4: Weighted regression with full controls ---")
X_weighted = data[['ELIGIBLE', 'AFTER', 'ELIGIBLE_X_AFTER', 'MALE', 'MARRIED', 'AGE'] + year_cols].copy()
X_weighted = sm.add_constant(X_weighted)
model_weighted = sm.WLS(y, X_weighted, weights=data['PERWT']).fit(cov_type='HC1')
print(f"N = {int(model_weighted.nobs)}")
print(f"DiD estimate: {model_weighted.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"SE: {model_weighted.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"p-value: {model_weighted.pvalues['ELIGIBLE_X_AFTER']:.4f}")

#%% Save results for report
print("\n" + "=" * 80)
print("PREFERRED SPECIFICATION SUMMARY")
print("=" * 80)

# Model 7 (Year FE + controls) as preferred specification
preferred = model7
print(f"""
PREFERRED ESTIMATE (Model 7: DiD with year FE and demographic/education controls)
================================================================================
DiD Effect (ELIGIBLE_X_AFTER): {preferred.params['ELIGIBLE_X_AFTER']:.5f}
Standard Error (robust):       {preferred.bse['ELIGIBLE_X_AFTER']:.5f}
t-statistic:                   {preferred.tvalues['ELIGIBLE_X_AFTER']:.3f}
p-value:                       {preferred.pvalues['ELIGIBLE_X_AFTER']:.4f}
95% Confidence Interval:       [{preferred.conf_int().loc['ELIGIBLE_X_AFTER'][0]:.5f}, {preferred.conf_int().loc['ELIGIBLE_X_AFTER'][1]:.5f}]

Sample Size:                   {int(preferred.nobs)}
R-squared:                     {preferred.rsquared:.4f}

Interpretation:
DACA eligibility is associated with a {abs(preferred.params['ELIGIBLE_X_AFTER'])*100:.2f} percentage point
{'increase' if preferred.params['ELIGIBLE_X_AFTER'] > 0 else 'decrease'} in full-time employment among
eligible individuals (ages 26-30) compared to the control group (ages 31-35).
The effect is {'statistically significant' if preferred.pvalues['ELIGIBLE_X_AFTER'] < 0.05 else 'not statistically significant'} at the 5% level.
""")

# Export key results to a CSV for the report
key_results = pd.DataFrame({
    'Specification': ['Basic DiD', 'With Controls', 'Year FE + Controls', 'State + Year FE', 'Full Model'],
    'Estimate': [model1.params['ELIGIBLE_X_AFTER'], model5.params['ELIGIBLE_X_AFTER'],
                 model7.params['ELIGIBLE_X_AFTER'], model8.params['ELIGIBLE_X_AFTER'],
                 model9.params['ELIGIBLE_X_AFTER']],
    'SE': [model1.bse['ELIGIBLE_X_AFTER'], model5.bse['ELIGIBLE_X_AFTER'],
           model7.bse['ELIGIBLE_X_AFTER'], model8.bse['ELIGIBLE_X_AFTER'],
           model9.bse['ELIGIBLE_X_AFTER']],
    'p_value': [model1.pvalues['ELIGIBLE_X_AFTER'], model5.pvalues['ELIGIBLE_X_AFTER'],
                model7.pvalues['ELIGIBLE_X_AFTER'], model8.pvalues['ELIGIBLE_X_AFTER'],
                model9.pvalues['ELIGIBLE_X_AFTER']],
    'N': [int(model1.nobs), int(model5.nobs), int(model7.nobs), int(model8.nobs), int(model9.nobs)]
})
key_results.to_csv('results_summary.csv', index=False)
print("\nResults saved to results_summary.csv")

# Save detailed model output
with open('model_output.txt', 'w') as f:
    f.write("DACA Effect on Full-Time Employment: Detailed Regression Output\n")
    f.write("=" * 80 + "\n\n")
    f.write("Model 7 (Preferred): DiD with Year FE and Controls\n")
    f.write("-" * 50 + "\n")
    f.write(str(model7.summary()))
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("\nModel 9: Full Model with State FE, Year FE, and Controls\n")
    f.write("-" * 50 + "\n")
    f.write(f"ELIGIBLE_X_AFTER: {model9.params['ELIGIBLE_X_AFTER']:.5f} (SE: {model9.bse['ELIGIBLE_X_AFTER']:.5f})\n")
    f.write(f"t-stat: {model9.tvalues['ELIGIBLE_X_AFTER']:.3f}, p-value: {model9.pvalues['ELIGIBLE_X_AFTER']:.4f}\n")

print("Detailed output saved to model_output.txt")

#%% Create figures data for LaTeX
print("\n" + "=" * 80)
print("GENERATING DATA FOR FIGURES")
print("=" * 80)

# Event study coefficients for plotting
event_study_data = []
event_study_data.append({'Year': 2011, 'Coefficient': 0, 'SE': 0, 'CI_lower': 0, 'CI_upper': 0})  # Reference year
for year in event_years:
    col = f'ELIGIBLE_X_{year}'
    coef = event_model.params[col]
    se = event_model.bse[col]
    ci = event_model.conf_int().loc[col]
    event_study_data.append({
        'Year': year,
        'Coefficient': coef,
        'SE': se,
        'CI_lower': ci[0],
        'CI_upper': ci[1]
    })

event_study_df = pd.DataFrame(event_study_data).sort_values('Year')
event_study_df.to_csv('event_study_results.csv', index=False)
print("\nEvent study data saved to event_study_results.csv")

# Trends data for plotting
trends.to_csv('trends_data.csv', index=False)
print("Trends data saved to trends_data.csv")

# Save summary statistics
summary_df.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

"""
DACA Replication Analysis
Effect of DACA eligibility on full-time employment among Hispanic-Mexican immigrants
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
print("DACA REPLICATION ANALYSIS")
print("=" * 80)

data = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nTotal observations: {len(data)}")
print(f"Years in data: {sorted(data['YEAR'].unique())}")

# Check key variables
print("\n" + "=" * 80)
print("KEY VARIABLE DISTRIBUTIONS")
print("=" * 80)

print(f"\nELIGIBLE distribution:")
print(data['ELIGIBLE'].value_counts().sort_index())

print(f"\nAFTER distribution:")
print(data['AFTER'].value_counts().sort_index())

print(f"\nFT (Full-time) distribution:")
print(data['FT'].value_counts().sort_index())

print(f"\nYear distribution:")
print(data['YEAR'].value_counts().sort_index())

# Summary statistics by group
print("\n" + "=" * 80)
print("SAMPLE SIZES BY GROUP")
print("=" * 80)

print("\nObservations by ELIGIBLE and AFTER:")
crosstab = pd.crosstab(data['ELIGIBLE'], data['AFTER'], margins=True)
print(crosstab)

# Mean FT by group
print("\n" + "=" * 80)
print("MEAN FULL-TIME EMPLOYMENT RATES BY GROUP")
print("=" * 80)

means = data.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
print(means)

# Manual DiD calculation
print("\n" + "=" * 80)
print("MANUAL DIFFERENCE-IN-DIFFERENCES CALCULATION")
print("=" * 80)

# Treatment group (ELIGIBLE=1)
treat_pre = data[(data['ELIGIBLE'] == 1) & (data['AFTER'] == 0)]['FT'].mean()
treat_post = data[(data['ELIGIBLE'] == 1) & (data['AFTER'] == 1)]['FT'].mean()
treat_diff = treat_post - treat_pre

# Control group (ELIGIBLE=0)
control_pre = data[(data['ELIGIBLE'] == 0) & (data['AFTER'] == 0)]['FT'].mean()
control_post = data[(data['ELIGIBLE'] == 0) & (data['AFTER'] == 1)]['FT'].mean()
control_diff = control_post - control_pre

# DiD estimate
did_estimate = treat_diff - control_diff

print(f"\nTreatment group (ages 26-30 in June 2012, ELIGIBLE=1):")
print(f"  Pre-DACA (2008-2011): {treat_pre:.4f}")
print(f"  Post-DACA (2013-2016): {treat_post:.4f}")
print(f"  Change: {treat_diff:.4f}")

print(f"\nControl group (ages 31-35 in June 2012, ELIGIBLE=0):")
print(f"  Pre-DACA (2008-2011): {control_pre:.4f}")
print(f"  Post-DACA (2013-2016): {control_post:.4f}")
print(f"  Change: {control_diff:.4f}")

print(f"\nDifference-in-Differences estimate: {did_estimate:.4f}")

# Regression-based DiD
print("\n" + "=" * 80)
print("REGRESSION-BASED DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

# Create interaction term
data['ELIGIBLE_AFTER'] = data['ELIGIBLE'] * data['AFTER']

# Model 1: Basic DiD (no covariates)
print("\n--- Model 1: Basic DiD (no covariates) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n" + "=" * 80)
print("--- Model 2: DiD with demographic controls ---")

# Check which covariates are available and suitable
# SEX: 1=Male, 2=Female per IPUMS coding
data['FEMALE'] = (data['SEX'] == 2).astype(int)

# Age (centered for interpretation)
data['AGE_centered'] = data['AGE'] - data['AGE'].mean()

# Marital status - create simplified categories
# MARST: 1=married spouse present, 2=married spouse absent, 3=separated, 4=divorced, 5=widowed, 6=never married
data['MARRIED'] = (data['MARST'].isin([1, 2])).astype(int)

# Number of children
data['HAS_CHILDREN'] = (data['NCHILD'] > 0).astype(int)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE_centered + MARRIED + HAS_CHILDREN',
                  data=data).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with demographic and education controls using matrices
print("\n" + "=" * 80)
print("--- Model 3: DiD with demographic and education controls ---")

# Recode education using EDUC variable (numeric)
# EDUC codes: 0=NA, 1=No school, 2-5=elementary, 6=HS graduate, 7-9=some college, 10=BA, 11=MA+
# Create education categories based on EDUC variable
data['educ_less_than_hs'] = (data['EDUC'] < 6).astype(int)
data['educ_hs'] = (data['EDUC'] == 6).astype(int)
data['educ_some_college'] = ((data['EDUC'] >= 7) & (data['EDUC'] <= 9)).astype(int)
data['educ_ba_plus'] = (data['EDUC'] >= 10).astype(int)

# Use matrix approach to avoid formula issues
# Prepare X matrix
X = data[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN',
          'educ_hs', 'educ_some_college', 'educ_ba_plus']].copy()
X = sm.add_constant(X)
y = data['FT']

model3 = sm.OLS(y, X).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with year fixed effects
print("\n" + "=" * 80)
print("--- Model 4: DiD with year fixed effects ---")

# Create year dummies
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    data[f'year_{year}'] = (data['YEAR'] == year).astype(int)

year_cols = [f'year_{y}' for y in [2009, 2010, 2011, 2013, 2014, 2015, 2016]]

X4 = data[['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN',
           'educ_hs', 'educ_some_college', 'educ_ba_plus'] + year_cols].copy()
X4 = sm.add_constant(X4)

model4 = sm.OLS(y, X4).fit(cov_type='HC1')
print(model4.summary())

# Model 5: DiD with state fixed effects
print("\n" + "=" * 80)
print("--- Model 5: DiD with year and state fixed effects ---")

# Create state dummies
states = sorted(data['STATEFIP'].unique())
ref_state = states[0]
for state in states[1:]:
    data[f'state_{state}'] = (data['STATEFIP'] == state).astype(int)

state_cols = [f'state_{s}' for s in states[1:]]

X5 = data[['ELIGIBLE', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN',
           'educ_hs', 'educ_some_college', 'educ_ba_plus'] + year_cols + state_cols].copy()
X5 = sm.add_constant(X5)

model5 = sm.OLS(y, X5).fit(cov_type='HC1')
print("\nModel 5 - Key coefficients:")
print(f"ELIGIBLE: {model5.params['ELIGIBLE']:.4f} (SE: {model5.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER (DiD): {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {int(model5.nobs)}")

# Model 6: Weighted regression using person weights
print("\n" + "=" * 80)
print("--- Model 6: Weighted DiD with full controls ---")

# WLS with person weights
model6 = sm.WLS(y, X5, weights=data['PERWT']).fit(cov_type='HC1')
print("\nModel 6 (Weighted) - Key coefficients:")
print(f"ELIGIBLE: {model6.params['ELIGIBLE']:.4f} (SE: {model6.bse['ELIGIBLE']:.4f})")
print(f"ELIGIBLE_AFTER (DiD): {model6.params['ELIGIBLE_AFTER']:.4f} (SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")

# Summary table of all models
print("\n" + "=" * 80)
print("SUMMARY TABLE OF ALL MODELS")
print("=" * 80)

models = [model1, model2, model3, model4, model5, model6]
model_names = ['Basic DiD', 'Demographics', 'Demo+Educ', 'Year FE', 'Year+State FE', 'Weighted']

summary_data = []
for name, model in zip(model_names, models):
    summary_data.append({
        'Model': name,
        'DiD Estimate': model.params['ELIGIBLE_AFTER'],
        'SE': model.bse['ELIGIBLE_AFTER'],
        'CI_low': model.conf_int().loc['ELIGIBLE_AFTER', 0],
        'CI_high': model.conf_int().loc['ELIGIBLE_AFTER', 1],
        'p-value': model.pvalues['ELIGIBLE_AFTER'],
        'N': int(model.nobs),
        'R-squared': model.rsquared
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary table
summary_df.to_csv('model_summary.csv', index=False)

# Robustness checks
print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS")
print("=" * 80)

# Check 1: Parallel trends - year-by-year effects
print("\n--- Check 1: Event study / Year-by-year effects ---")

# Create year-specific treatment effects
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    data[f'ELIGIBLE_Y{year}'] = data['ELIGIBLE'] * (data['YEAR'] == year).astype(int)

# Event study regression (omitting 2011 as reference)
event_cols = [f'ELIGIBLE_Y{y}' for y in [2008, 2009, 2010, 2013, 2014, 2015, 2016]]

X_event = data[['ELIGIBLE'] + event_cols + ['FEMALE', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN',
                'educ_hs', 'educ_some_college', 'educ_ba_plus'] +
               [c for c in year_cols if c != 'year_2011'] + state_cols].copy()
X_event = sm.add_constant(X_event)

model_event = sm.OLS(y, X_event).fit(cov_type='HC1')

print("\nEvent study coefficients (relative to 2011):")
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    coef = model_event.params.get(f'ELIGIBLE_Y{year}', np.nan)
    se = model_event.bse.get(f'ELIGIBLE_Y{year}', np.nan)
    pval = model_event.pvalues.get(f'ELIGIBLE_Y{year}', np.nan)
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.3f})")

# Check 2: Heterogeneity by sex
print("\n--- Check 2: Heterogeneity by sex ---")

data_male = data[data['SEX'] == 1].copy()
data_female = data[data['SEX'] == 2].copy()

# Model for subgroups
X_male = data_male[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN']].copy()
X_male = sm.add_constant(X_male)
y_male = data_male['FT']

X_female = data_female[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'AGE_centered', 'MARRIED', 'HAS_CHILDREN']].copy()
X_female = sm.add_constant(X_female)
y_female = data_female['FT']

model_male = sm.OLS(y_male, X_male).fit(cov_type='HC1')
model_female = sm.OLS(y_female, X_female).fit(cov_type='HC1')

print(f"\nMales (N={len(data_male)}):")
print(f"  DiD estimate: {model_male.params['ELIGIBLE_AFTER']:.4f} (SE: {model_male.bse['ELIGIBLE_AFTER']:.4f}, p={model_male.pvalues['ELIGIBLE_AFTER']:.3f})")

print(f"\nFemales (N={len(data_female)}):")
print(f"  DiD estimate: {model_female.params['ELIGIBLE_AFTER']:.4f} (SE: {model_female.bse['ELIGIBLE_AFTER']:.4f}, p={model_female.pvalues['ELIGIBLE_AFTER']:.3f})")

# Check 3: Placebo test - pre-trend test
print("\n--- Check 3: Pre-period placebo test ---")
# Test if there's a trend difference in the pre-period (using 2010-2011 as "post" for placebo)
data_pre = data[data['AFTER'] == 0].copy()
data_pre['PLACEBO_AFTER'] = (data_pre['YEAR'].isin([2010, 2011])).astype(int)
data_pre['ELIGIBLE_PLACEBO'] = data_pre['ELIGIBLE'] * data_pre['PLACEBO_AFTER']

X_placebo = data_pre[['ELIGIBLE', 'PLACEBO_AFTER', 'ELIGIBLE_PLACEBO']].copy()
X_placebo = sm.add_constant(X_placebo)
y_placebo = data_pre['FT']

model_placebo = sm.OLS(y_placebo, X_placebo).fit(cov_type='HC1')
print(f"\nPlacebo DiD (2008-2009 vs 2010-2011, pre-DACA):")
print(f"  Estimate: {model_placebo.params['ELIGIBLE_PLACEBO']:.4f} (SE: {model_placebo.bse['ELIGIBLE_PLACEBO']:.4f}, p={model_placebo.pvalues['ELIGIBLE_PLACEBO']:.3f})")
print("  (A significant effect here would suggest pre-existing trends)")

# Descriptive statistics for report
print("\n" + "=" * 80)
print("DESCRIPTIVE STATISTICS FOR REPORT")
print("=" * 80)

# Overall sample characteristics
print("\n--- Sample characteristics ---")
print(f"Total N: {len(data)}")
print(f"Treated (ELIGIBLE=1): {(data['ELIGIBLE']==1).sum()}")
print(f"Control (ELIGIBLE=0): {(data['ELIGIBLE']==0).sum()}")
print(f"Pre-period: {(data['AFTER']==0).sum()}")
print(f"Post-period: {(data['AFTER']==1).sum()}")

print("\n--- Demographics ---")
print(f"Female proportion: {data['FEMALE'].mean():.3f}")
print(f"Mean age: {data['AGE'].mean():.1f} (SD: {data['AGE'].std():.1f})")
print(f"Married proportion: {data['MARRIED'].mean():.3f}")
print(f"Has children proportion: {data['HAS_CHILDREN'].mean():.3f}")
print(f"Mean full-time employment: {data['FT'].mean():.3f}")

print("\n--- Education distribution ---")
print(f"Less than HS: {data['educ_less_than_hs'].mean():.3f}")
print(f"HS degree: {data['educ_hs'].mean():.3f}")
print(f"Some college: {data['educ_some_college'].mean():.3f}")
print(f"BA+: {data['educ_ba_plus'].mean():.3f}")

# Save event study results
event_results = []
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    event_results.append({
        'Year': year,
        'Coefficient': model_event.params.get(f'ELIGIBLE_Y{year}', np.nan),
        'SE': model_event.bse.get(f'ELIGIBLE_Y{year}', np.nan),
        'CI_low': model_event.conf_int().loc[f'ELIGIBLE_Y{year}', 0] if f'ELIGIBLE_Y{year}' in model_event.params else np.nan,
        'CI_high': model_event.conf_int().loc[f'ELIGIBLE_Y{year}', 1] if f'ELIGIBLE_Y{year}' in model_event.params else np.nan
    })
event_df = pd.DataFrame(event_results)
event_df.to_csv('event_study_results.csv', index=False)

# Create balance table
print("\n" + "=" * 80)
print("BALANCE TABLE - PRE-TREATMENT CHARACTERISTICS")
print("=" * 80)

data_pre_balance = data[(data['AFTER'] == 0)].copy()

balance_vars = ['AGE', 'FEMALE', 'MARRIED', 'HAS_CHILDREN', 'FT']
balance_table = []

for var in balance_vars:
    treat_mean = data_pre_balance[data_pre_balance['ELIGIBLE'] == 1][var].mean()
    treat_sd = data_pre_balance[data_pre_balance['ELIGIBLE'] == 1][var].std()
    control_mean = data_pre_balance[data_pre_balance['ELIGIBLE'] == 0][var].mean()
    control_sd = data_pre_balance[data_pre_balance['ELIGIBLE'] == 0][var].std()

    # t-test for difference
    treat_vals = data_pre_balance[data_pre_balance['ELIGIBLE'] == 1][var]
    control_vals = data_pre_balance[data_pre_balance['ELIGIBLE'] == 0][var]
    t_stat, p_val = stats.ttest_ind(treat_vals, control_vals)

    balance_table.append({
        'Variable': var,
        'Treated_Mean': treat_mean,
        'Treated_SD': treat_sd,
        'Control_Mean': control_mean,
        'Control_SD': control_sd,
        'Difference': treat_mean - control_mean,
        'p_value': p_val
    })

balance_df = pd.DataFrame(balance_table)
print(balance_df.to_string(index=False))
balance_df.to_csv('balance_table.csv', index=False)

# Final preferred estimate
print("\n" + "=" * 80)
print("PREFERRED ESTIMATE")
print("=" * 80)

# Model 5 (unweighted with full FE) as preferred
print(f"\nPreferred Model: DiD with year and state fixed effects, demographic and education controls")
print(f"Effect size (DiD estimate): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard error: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% Confidence interval: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample size: {int(model5.nobs)}")

# Year-by-year analysis for report
print("\n" + "=" * 80)
print("YEAR-BY-YEAR FULL-TIME EMPLOYMENT RATES")
print("=" * 80)

yearly_stats = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].agg(['mean', 'std', 'count']).reset_index()
print(yearly_stats.to_string(index=False))
yearly_stats.to_csv('yearly_stats.csv', index=False)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

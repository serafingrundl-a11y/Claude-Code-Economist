"""
DACA Replication Study - Analysis Script
Replication 48

Research Question: Effect of DACA eligibility on full-time employment among
Mexican-born Hispanic individuals.

Research Design: Difference-in-Differences
- Treatment: DACA-eligible (ages 26-30 in June 2012)
- Control: Ages 31-35 in June 2012 (otherwise would have been eligible)
- Pre-period: 2008-2011
- Post-period: 2013-2016
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

print("=" * 80)
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# =============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 1: DATA LOADING AND INITIAL EXPLORATION")
print("=" * 80)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nDataset Shape: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"\nYears in data: {sorted(df['YEAR'].unique())}")
print(f"\nKey variable summary:")
print(f"  - FT (Full-time): {df['FT'].value_counts().to_dict()}")
print(f"  - ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"  - AFTER: {df['AFTER'].value_counts().to_dict()}")

# Check for missing values in key variables
print(f"\nMissing values in key variables:")
key_vars = ['FT', 'ELIGIBLE', 'AFTER', 'YEAR', 'PERWT', 'SEX', 'AGE', 'EDUC']
for var in key_vars:
    missing = df[var].isna().sum()
    print(f"  - {var}: {missing}")

# =============================================================================
# 2. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 2: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Create interaction term
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Sample sizes by group
print("\n2.1 Sample Sizes by Treatment Group and Period:")
print("-" * 60)
sample_sizes = df.groupby(['ELIGIBLE', 'AFTER']).size().unstack()
sample_sizes.columns = ['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)']
sample_sizes.index = ['Control (ages 31-35)', 'Treatment (ages 26-30)']
print(sample_sizes)
print(f"\nTotal sample size: {len(df)}")

# Sample sizes by year
print("\n2.2 Sample Sizes by Year and Treatment Group:")
print("-" * 60)
yearly_samples = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
yearly_samples.columns = ['Control', 'Treatment']
print(yearly_samples)

# Full-time employment rates by group (unweighted)
print("\n2.3 Full-Time Employment Rates (Unweighted):")
print("-" * 60)
ft_rates_unweighted = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()
ft_rates_unweighted.columns = ['Pre-DACA', 'Post-DACA']
ft_rates_unweighted.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_rates_unweighted.round(4))

# Full-time employment rates (weighted)
print("\n2.4 Full-Time Employment Rates (Weighted):")
print("-" * 60)

def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

ft_rates_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT'), include_groups=False
).unstack()
ft_rates_weighted.columns = ['Pre-DACA', 'Post-DACA']
ft_rates_weighted.index = ['Control (31-35)', 'Treatment (26-30)']
print(ft_rates_weighted.round(4))

# Calculate simple DiD estimate
print("\n2.5 Simple Difference-in-Differences Calculation:")
print("-" * 60)
pre_treat = ft_rates_weighted.loc['Treatment (26-30)', 'Pre-DACA']
post_treat = ft_rates_weighted.loc['Treatment (26-30)', 'Post-DACA']
pre_ctrl = ft_rates_weighted.loc['Control (31-35)', 'Pre-DACA']
post_ctrl = ft_rates_weighted.loc['Control (31-35)', 'Post-DACA']

diff_treat = post_treat - pre_treat
diff_ctrl = post_ctrl - pre_ctrl
did_estimate = diff_treat - diff_ctrl

print(f"Treatment group change (post - pre): {diff_treat:.4f}")
print(f"Control group change (post - pre):   {diff_ctrl:.4f}")
print(f"Difference-in-Differences estimate:  {did_estimate:.4f}")

# Detailed characteristics by group
print("\n2.6 Sample Characteristics by Treatment Group (Pre-Period):")
print("-" * 60)

# Filter to pre-period
df_pre = df[df['AFTER'] == 0]

# Calculate weighted means for demographic variables
demographics = {}
for group in [0, 1]:
    group_data = df_pre[df_pre['ELIGIBLE'] == group]
    demographics[group] = {
        'Age (mean)': np.average(group_data['AGE'], weights=group_data['PERWT']),
        'Male (%)': np.average(group_data['SEX'] == 1, weights=group_data['PERWT']) * 100,
        'Married (%)': np.average(group_data['MARST'] == 1, weights=group_data['PERWT']) * 100,
        'Has Children (%)': np.average(group_data['NCHILD'] > 0, weights=group_data['PERWT']) * 100,
        'N Observations': len(group_data)
    }

demo_df = pd.DataFrame(demographics).T
demo_df.index = ['Control (31-35)', 'Treatment (26-30)']
print(demo_df.round(2))

# Education distribution
print("\n2.7 Education Distribution (Pre-Period, Weighted %):")
print("-" * 60)

# Load labelled version for education categories
df_labelled = pd.read_csv('data/prepared_data_labelled_version.csv')
df_labelled['ELIGIBLE_X_AFTER'] = df_labelled['ELIGIBLE'] * df_labelled['AFTER']

# Education by group
for group, name in [(0, 'Control (31-35)'), (1, 'Treatment (26-30)')]:
    group_data = df_labelled[(df_labelled['AFTER'] == 0) & (df_labelled['ELIGIBLE'] == group)]
    print(f"\n{name}:")
    if 'EDUC_RECODE' in group_data.columns:
        educ_dist = group_data.groupby('EDUC_RECODE').apply(
            lambda x: np.sum(x['PERWT']) / np.sum(group_data['PERWT']) * 100,
            include_groups=False
        )
        for cat, pct in educ_dist.items():
            print(f"  {cat}: {pct:.1f}%")

# =============================================================================
# 3. PRE-TREND ANALYSIS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 3: PRE-TREND ANALYSIS")
print("=" * 80)

# Calculate yearly FT rates by treatment group
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT'), include_groups=False
).unstack()
yearly_ft.columns = ['Control', 'Treatment']

print("\n3.1 Full-Time Employment Rates by Year:")
print("-" * 60)
print(yearly_ft.round(4))

# Pre-period trends
pre_years = [2008, 2009, 2010, 2011]
yearly_ft_pre = yearly_ft.loc[pre_years]

print("\n3.2 Pre-Period Trend Analysis:")
print("-" * 60)

# Linear trend test for each group
for col in ['Control', 'Treatment']:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        pre_years, yearly_ft_pre[col]
    )
    print(f"{col}: slope = {slope:.4f}, p-value = {p_value:.4f}")

# Difference in trends
yearly_ft_pre['Difference'] = yearly_ft_pre['Treatment'] - yearly_ft_pre['Control']
slope, intercept, r_value, p_value, std_err = stats.linregress(
    pre_years, yearly_ft_pre['Difference']
)
print(f"Difference: slope = {slope:.4f}, p-value = {p_value:.4f}")

# Create visualization for pre-trends
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(yearly_ft.index, yearly_ft['Treatment'], 'b-o', label='Treatment (26-30)', linewidth=2)
ax.plot(yearly_ft.index, yearly_ft['Control'], 'r-s', label='Control (31-35)', linewidth=2)
ax.axvline(x=2012, color='gray', linestyle='--', label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(2007.5, 2016.5)
plt.tight_layout()
plt.savefig('figures/pretrends.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nPre-trends figure saved to figures/pretrends.png")

# =============================================================================
# 4. MAIN DIFFERENCE-IN-DIFFERENCES REGRESSION
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 4: MAIN DiD REGRESSION ANALYSIS")
print("=" * 80)

# Model 1: Basic DiD without covariates (unweighted)
print("\n4.1 Model 1: Basic DiD (Unweighted)")
print("-" * 60)

model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model1.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model1.rsquared:.4f}")
print(f"N: {model1.nobs:.0f}")

# Model 2: Basic DiD with weights
print("\n4.2 Model 2: Basic DiD (Weighted)")
print("-" * 60)

model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model2.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model2.rsquared:.4f}")
print(f"N: {model2.nobs:.0f}")

# Model 3: DiD with demographic covariates
print("\n4.3 Model 3: DiD with Demographic Covariates (Weighted)")
print("-" * 60)

# Create dummy variables for categorical covariates
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model3.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model3.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model3.rsquared:.4f}")
print(f"N: {model3.nobs:.0f}")

# Model 4: DiD with education controls
print("\n4.4 Model 4: DiD with Education Controls (Weighted)")
print("-" * 60)

# Create education dummies (using EDUC)
# EDUC: 0-5 = less than HS, 6 = HS, 7+ = some college or more
df['HS_GRAD'] = (df['EDUC'] == 6).astype(int)
df['SOME_COLLEGE_PLUS'] = (df['EDUC'] > 6).astype(int)

model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS',
                  data=df, weights=df['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model4.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model4.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model4.rsquared:.4f}")
print(f"N: {model4.nobs:.0f}")

# Model 5: DiD with state fixed effects
print("\n4.5 Model 5: DiD with State Fixed Effects (Weighted)")
print("-" * 60)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='STATE', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)
state_vars = [col for col in df_with_states.columns if col.startswith('STATE_')]

formula5 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS + ' + ' + '.join(state_vars)
model5 = smf.wls(formula5, data=df_with_states, weights=df_with_states['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model5.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model5.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")
print(f"N: {model5.nobs:.0f}")

# Model 6: DiD with year fixed effects
print("\n4.6 Model 6: DiD with Year Fixed Effects (Weighted)")
print("-" * 60)

year_dummies = pd.get_dummies(df['YEAR'], prefix='YEAR', drop_first=True)
df_with_years = pd.concat([df, year_dummies], axis=1)
year_vars = [col for col in df_with_years.columns if col.startswith('YEAR_')]

formula6 = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS + ' + ' + '.join(year_vars)
model6 = smf.wls(formula6, data=df_with_years, weights=df_with_years['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model6.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model6.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model6.rsquared:.4f}")
print(f"N: {model6.nobs:.0f}")

# Model 7: Full model with state and year FE
print("\n4.7 Model 7: Full Model with State and Year FE (Weighted)")
print("-" * 60)

df_full = pd.concat([df, state_dummies, year_dummies], axis=1)
formula7 = 'FT ~ ELIGIBLE + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS + ' + ' + '.join(state_vars) + ' + ' + ' + '.join(year_vars)
model7 = smf.wls(formula7, data=df_full, weights=df_full['PERWT']).fit()
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model7.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model7.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model7.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model7.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model7.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"R-squared: {model7.rsquared:.4f}")
print(f"N: {model7.nobs:.0f}")

# =============================================================================
# 5. ROBUST STANDARD ERRORS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 5: ROBUST AND CLUSTERED STANDARD ERRORS")
print("=" * 80)

# Model with heteroskedasticity-robust standard errors (HC1)
print("\n5.1 Preferred Model with Robust Standard Errors (HC1)")
print("-" * 60)

model_robust = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS',
                        data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model_robust.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Robust Standard Error: {model_robust.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model_robust.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# Clustered standard errors by state
print("\n5.2 Model with State-Clustered Standard Errors")
print("-" * 60)

model_clustered = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS',
                           data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})
print(f"DiD Coefficient (ELIGIBLE_X_AFTER): {model_clustered.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Clustered Standard Error: {model_clustered.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% CI: [{model_clustered.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_clustered.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model_clustered.pvalues['ELIGIBLE_X_AFTER']:.4f}")

# =============================================================================
# 6. ROBUSTNESS CHECKS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 6: ROBUSTNESS CHECKS")
print("=" * 80)

# 6.1 Subgroup analysis by sex
print("\n6.1 Subgroup Analysis by Sex:")
print("-" * 60)

for sex, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sex = df[df['SEX'] == sex]
    model_sex = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS',
                         data=df_sex, weights=df_sex['PERWT']).fit(cov_type='HC1')
    print(f"\n{sex_name}:")
    print(f"  DiD Coefficient: {model_sex.params['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  Robust SE: {model_sex.bse['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  P-value: {model_sex.pvalues['ELIGIBLE_X_AFTER']:.4f}")
    print(f"  N: {model_sex.nobs:.0f}")

# 6.2 Event study analysis
print("\n6.2 Event Study Analysis:")
print("-" * 60)

# Create year indicators interacted with ELIGIBLE
years = sorted(df['YEAR'].unique())
base_year = 2011  # Use 2011 as reference year

for year in years:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    if year != base_year:
        df[f'ELIGIBLE_X_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

year_interactions = [f'ELIGIBLE_X_YEAR_{year}' for year in years if year != base_year]
year_dummies_list = [f'YEAR_{year}' for year in years if year != base_year]

formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join(year_dummies_list) + ' + ' + ' + '.join(year_interactions) + ' + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS'
model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='HC1')

print("Event Study Coefficients (relative to 2011):")
for year in years:
    if year != base_year:
        coef = model_event.params[f'ELIGIBLE_X_YEAR_{year}']
        se = model_event.bse[f'ELIGIBLE_X_YEAR_{year}']
        pval = model_event.pvalues[f'ELIGIBLE_X_YEAR_{year}']
        print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p: {pval:.4f})")
    else:
        print(f"  {year}: 0.0000 (reference)")

# Create event study plot
fig, ax = plt.subplots(figsize=(10, 6))
event_coefs = []
event_ses = []
event_years = []

for year in years:
    event_years.append(year)
    if year != base_year:
        event_coefs.append(model_event.params[f'ELIGIBLE_X_YEAR_{year}'])
        event_ses.append(model_event.bse[f'ELIGIBLE_X_YEAR_{year}'])
    else:
        event_coefs.append(0)
        event_ses.append(0)

event_coefs = np.array(event_coefs)
event_ses = np.array(event_ses)

ax.errorbar(event_years, event_coefs, yerr=1.96*event_ses, fmt='o-', capsize=5,
            color='blue', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nEvent study figure saved to figures/event_study.png")

# 6.3 Placebo test with pre-treatment periods
print("\n6.3 Placebo Test (2008-2009 vs 2010-2011):")
print("-" * 60)

df_pre = df[df['YEAR'].isin([2008, 2009, 2010, 2011])].copy()
df_pre['PLACEBO_POST'] = (df_pre['YEAR'].isin([2010, 2011])).astype(int)
df_pre['ELIGIBLE_X_PLACEBO'] = df_pre['ELIGIBLE'] * df_pre['PLACEBO_POST']

model_placebo = smf.wls('FT ~ ELIGIBLE + PLACEBO_POST + ELIGIBLE_X_PLACEBO + FEMALE + MARRIED + HAS_CHILDREN + AGE + HS_GRAD + SOME_COLLEGE_PLUS',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')
print(f"Placebo DiD Coefficient: {model_placebo.params['ELIGIBLE_X_PLACEBO']:.4f}")
print(f"Robust SE: {model_placebo.bse['ELIGIBLE_X_PLACEBO']:.4f}")
print(f"P-value: {model_placebo.pvalues['ELIGIBLE_X_PLACEBO']:.4f}")

# 6.4 Different age bandwidths
print("\n6.4 Analysis Notes:")
print("-" * 60)
print("Note: The analysis uses the pre-defined ELIGIBLE variable which identifies")
print("      the treatment (ages 26-30 in June 2012) and control (ages 31-35) groups.")
print("      Age bandwidth robustness checks would require modifying this variable,")
print("      which is outside the scope as per instructions.")

# =============================================================================
# 7. SUMMARY OF RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 7: SUMMARY OF RESULTS")
print("=" * 80)

print("\n7.1 Summary Table of DiD Estimates:")
print("-" * 80)
print(f"{'Model':<45} {'Coef':>10} {'SE':>10} {'P-value':>10}")
print("-" * 80)
print(f"{'(1) Basic DiD (unweighted)':<45} {model1.params['ELIGIBLE_X_AFTER']:>10.4f} {model1.bse['ELIGIBLE_X_AFTER']:>10.4f} {model1.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(2) Basic DiD (weighted)':<45} {model2.params['ELIGIBLE_X_AFTER']:>10.4f} {model2.bse['ELIGIBLE_X_AFTER']:>10.4f} {model2.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(3) + Demographics':<45} {model3.params['ELIGIBLE_X_AFTER']:>10.4f} {model3.bse['ELIGIBLE_X_AFTER']:>10.4f} {model3.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(4) + Education':<45} {model4.params['ELIGIBLE_X_AFTER']:>10.4f} {model4.bse['ELIGIBLE_X_AFTER']:>10.4f} {model4.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(5) + State FE':<45} {model5.params['ELIGIBLE_X_AFTER']:>10.4f} {model5.bse['ELIGIBLE_X_AFTER']:>10.4f} {model5.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(6) + Year FE':<45} {model6.params['ELIGIBLE_X_AFTER']:>10.4f} {model6.bse['ELIGIBLE_X_AFTER']:>10.4f} {model6.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(7) + State + Year FE':<45} {model7.params['ELIGIBLE_X_AFTER']:>10.4f} {model7.bse['ELIGIBLE_X_AFTER']:>10.4f} {model7.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(4R) Preferred w/ Robust SE':<45} {model_robust.params['ELIGIBLE_X_AFTER']:>10.4f} {model_robust.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_robust.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print(f"{'(4C) Preferred w/ Clustered SE':<45} {model_clustered.params['ELIGIBLE_X_AFTER']:>10.4f} {model_clustered.bse['ELIGIBLE_X_AFTER']:>10.4f} {model_clustered.pvalues['ELIGIBLE_X_AFTER']:>10.4f}")
print("-" * 80)

# Preferred estimate summary
print("\n7.2 PREFERRED ESTIMATE (Model 4 with Robust SE):")
print("-" * 60)
print(f"Effect Size: {model_robust.params['ELIGIBLE_X_AFTER']:.4f}")
print(f"Standard Error: {model_robust.bse['ELIGIBLE_X_AFTER']:.4f}")
print(f"95% Confidence Interval: [{model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 0]:.4f}, {model_robust.conf_int().loc['ELIGIBLE_X_AFTER', 1]:.4f}]")
print(f"P-value: {model_robust.pvalues['ELIGIBLE_X_AFTER']:.4f}")
print(f"Sample Size: {int(model_robust.nobs)}")

# Interpretation
effect_pct = model_robust.params['ELIGIBLE_X_AFTER'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.2f} percentage point")
print(f"{'increase' if effect_pct > 0 else 'decrease'} in the probability of full-time employment.")

# Save full regression results
print("\n7.3 Full Regression Output (Preferred Model):")
print("-" * 60)
print(model_robust.summary())

# =============================================================================
# 8. SAVE RESULTS FOR REPORT
# =============================================================================
print("\n" + "=" * 80)
print("SECTION 8: SAVING RESULTS")
print("=" * 80)

# Save key results to a CSV for the report
results_dict = {
    'Model': [
        'Basic DiD (unweighted)', 'Basic DiD (weighted)',
        'Demographics', 'Education', 'State FE', 'Year FE',
        'State + Year FE', 'Preferred (Robust SE)', 'Clustered SE'
    ],
    'Coefficient': [
        model1.params['ELIGIBLE_X_AFTER'], model2.params['ELIGIBLE_X_AFTER'],
        model3.params['ELIGIBLE_X_AFTER'], model4.params['ELIGIBLE_X_AFTER'],
        model5.params['ELIGIBLE_X_AFTER'], model6.params['ELIGIBLE_X_AFTER'],
        model7.params['ELIGIBLE_X_AFTER'], model_robust.params['ELIGIBLE_X_AFTER'],
        model_clustered.params['ELIGIBLE_X_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_X_AFTER'], model2.bse['ELIGIBLE_X_AFTER'],
        model3.bse['ELIGIBLE_X_AFTER'], model4.bse['ELIGIBLE_X_AFTER'],
        model5.bse['ELIGIBLE_X_AFTER'], model6.bse['ELIGIBLE_X_AFTER'],
        model7.bse['ELIGIBLE_X_AFTER'], model_robust.bse['ELIGIBLE_X_AFTER'],
        model_clustered.bse['ELIGIBLE_X_AFTER']
    ],
    'P_value': [
        model1.pvalues['ELIGIBLE_X_AFTER'], model2.pvalues['ELIGIBLE_X_AFTER'],
        model3.pvalues['ELIGIBLE_X_AFTER'], model4.pvalues['ELIGIBLE_X_AFTER'],
        model5.pvalues['ELIGIBLE_X_AFTER'], model6.pvalues['ELIGIBLE_X_AFTER'],
        model7.pvalues['ELIGIBLE_X_AFTER'], model_robust.pvalues['ELIGIBLE_X_AFTER'],
        model_clustered.pvalues['ELIGIBLE_X_AFTER']
    ],
    'N': [
        int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs),
        int(model5.nobs), int(model6.nobs), int(model7.nobs),
        int(model_robust.nobs), int(model_clustered.nobs)
    ],
    'R_squared': [
        model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared,
        model5.rsquared, model6.rsquared, model7.rsquared,
        model_robust.rsquared, model_clustered.rsquared
    ]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('results/regression_results.csv', index=False)
print("Regression results saved to results/regression_results.csv")

# Save descriptive statistics
desc_stats = {
    'Group': ['Control (31-35)', 'Treatment (26-30)', 'Control (31-35)', 'Treatment (26-30)'],
    'Period': ['Pre-DACA', 'Pre-DACA', 'Post-DACA', 'Post-DACA'],
    'FT_Rate_Weighted': [pre_ctrl, pre_treat, post_ctrl, post_treat],
    'N': [
        len(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]),
        len(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]),
        len(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]),
        len(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)])
    ]
}
desc_df = pd.DataFrame(desc_stats)
desc_df.to_csv('results/descriptive_stats.csv', index=False)
print("Descriptive statistics saved to results/descriptive_stats.csv")

# Save event study coefficients
event_results = {
    'Year': event_years,
    'Coefficient': event_coefs,
    'SE': event_ses
}
event_df = pd.DataFrame(event_results)
event_df.to_csv('results/event_study_coefficients.csv', index=False)
print("Event study coefficients saved to results/event_study_coefficients.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

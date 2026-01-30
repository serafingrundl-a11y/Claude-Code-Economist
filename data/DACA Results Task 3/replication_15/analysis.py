"""
DACA Replication Study - Difference-in-Differences Analysis
Examining the effect of DACA eligibility on full-time employment
among ethnically Hispanic-Mexican, Mexican-born individuals in the US
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load data
print("=" * 80)
print("DACA REPLICATION STUDY - DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"\nTotal observations: {len(df)}")
print(f"Number of variables: {df.shape[1]}")

# Key variables summary
print("\n" + "=" * 80)
print("KEY VARIABLES SUMMARY")
print("=" * 80)

print("\n--- YEAR Distribution ---")
print(df['YEAR'].value_counts().sort_index())

print("\n--- ELIGIBLE Distribution ---")
print(df['ELIGIBLE'].value_counts())
print(f"Eligible (treatment group ages 26-30): {df['ELIGIBLE'].sum()}")
print(f"Comparison group (ages 31-35): {(df['ELIGIBLE']==0).sum()}")

print("\n--- AFTER Distribution ---")
print(df['AFTER'].value_counts())
print(f"Pre-DACA (2008-2011): {(df['AFTER']==0).sum()}")
print(f"Post-DACA (2013-2016): {(df['AFTER']==1).sum()}")

print("\n--- FT (Full-Time Employment) Distribution ---")
print(df['FT'].value_counts())
print(f"Full-time employed: {df['FT'].sum()} ({100*df['FT'].mean():.2f}%)")

print("\n--- AGE_IN_JUNE_2012 Distribution ---")
print(df['AGE_IN_JUNE_2012'].describe())

# Create 2x2 table for DiD
print("\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES SETUP")
print("=" * 80)

# Calculate mean FT by group and period
did_table = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'std']).round(4)
print("\nMean Full-Time Employment by Group and Period:")
print(did_table)

# Calculate DiD manually
ft_eligible_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
ft_eligible_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
ft_control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
ft_control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\n--- Manual DiD Calculation ---")
print(f"Treatment group (ELIGIBLE=1) before: {ft_eligible_before:.4f}")
print(f"Treatment group (ELIGIBLE=1) after:  {ft_eligible_after:.4f}")
print(f"Change for treatment group: {ft_eligible_after - ft_eligible_before:.4f}")
print(f"\nControl group (ELIGIBLE=0) before: {ft_control_before:.4f}")
print(f"Control group (ELIGIBLE=0) after:  {ft_control_after:.4f}")
print(f"Change for control group: {ft_control_after - ft_control_before:.4f}")

did_estimate = (ft_eligible_after - ft_eligible_before) - (ft_control_after - ft_control_before)
print(f"\n*** DiD Estimate (raw): {did_estimate:.4f} ***")

# Sample sizes by group
print("\n--- Sample Sizes by Group ---")
for eligible in [0, 1]:
    for after in [0, 1]:
        n = len(df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)])
        group = "Treatment" if eligible == 1 else "Control"
        period = "Post-DACA" if after == 1 else "Pre-DACA"
        print(f"{group}, {period}: n = {n}")

# Basic DiD Regression
print("\n" + "=" * 80)
print("MODEL 1: BASIC DIFFERENCE-IN-DIFFERENCES REGRESSION")
print("FT ~ ELIGIBLE + AFTER + ELIGIBLE*AFTER")
print("=" * 80)

df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary())

# DiD with robust standard errors
print("\n" + "=" * 80)
print("MODEL 2: DiD WITH ROBUST (HC1) STANDARD ERRORS")
print("=" * 80)

model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model2.summary())

# DiD with clustered standard errors by state
print("\n" + "=" * 80)
print("MODEL 3: DiD WITH STATE-CLUSTERED STANDARD ERRORS")
print("=" * 80)

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print(model3.summary())

# DiD with year fixed effects
print("\n" + "=" * 80)
print("MODEL 4: DiD WITH YEAR FIXED EFFECTS")
print("=" * 80)

df['YEAR_factor'] = pd.Categorical(df['YEAR'])
model4 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(model4.summary())

# DiD with covariates
print("\n" + "=" * 80)
print("MODEL 5: DiD WITH COVARIATES")
print("(Sex, Education, Marital Status)")
print("=" * 80)

# Create education dummies based on EDUC_RECODE
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = df['MARST'].isin([1, 2]).astype(int)

# Education categories
print("\nEducation distribution (EDUC_RECODE):")
print(df['EDUC_RECODE'].value_counts())

model5 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + female + married + C(EDUC_RECODE)',
                 data=df).fit(cov_type='HC1')
print(model5.summary())

# DiD with more covariates including state FE
print("\n" + "=" * 80)
print("MODEL 6: DiD WITH STATE FIXED EFFECTS AND COVARIATES")
print("=" * 80)

model6 = smf.ols('FT ~ ELIGIBLE + C(YEAR) + ELIGIBLE_AFTER + female + married + C(EDUC_RECODE) + C(STATEFIP)',
                 data=df).fit(cov_type='HC1')

# Print key coefficients (full summary would be very long)
print("\nKey coefficients from Model 6:")
print(f"ELIGIBLE_AFTER (DiD estimate): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"Std Error: {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model6.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"\nR-squared: {model6.rsquared:.4f}")
print(f"Number of observations: {model6.nobs:.0f}")

# Model with person weights
print("\n" + "=" * 80)
print("MODEL 7: WEIGHTED DiD (USING PERWT)")
print("=" * 80)

import statsmodels.api as sm
X = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'female', 'married']]
X = sm.add_constant(X)
y = df['FT']
model7 = sm.WLS(y, X, weights=df['PERWT']).fit(cov_type='HC1')
print(model7.summary())

# Parallel trends examination
print("\n" + "=" * 80)
print("PARALLEL TRENDS EXAMINATION")
print("=" * 80)

# Calculate yearly means by group
yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']
print("\nYearly Full-Time Employment Rates:")
print(yearly_means.round(4))

# Pre-trends test: interaction of ELIGIBLE with year dummies in pre-period only
print("\n--- Pre-Trends Test (Pre-DACA period only) ---")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['YEAR_centered'] = df_pre['YEAR'] - 2008

model_pretrends = smf.ols('FT ~ ELIGIBLE * YEAR_centered', data=df_pre).fit(cov_type='HC1')
print(f"Interaction coefficient (ELIGIBLE:YEAR_centered): {model_pretrends.params['ELIGIBLE:YEAR_centered']:.4f}")
print(f"p-value: {model_pretrends.pvalues['ELIGIBLE:YEAR_centered']:.4f}")
if model_pretrends.pvalues['ELIGIBLE:YEAR_centered'] > 0.05:
    print("=> No statistically significant difference in pre-trends (parallel trends assumption supported)")
else:
    print("=> Warning: Significant difference in pre-trends detected")

# Event study / dynamic DiD
print("\n" + "=" * 80)
print("EVENT STUDY / DYNAMIC DIFFERENCE-IN-DIFFERENCES")
print("=" * 80)

# Create year dummies interacted with ELIGIBLE (reference: 2011)
df['year_2008'] = (df['YEAR'] == 2008).astype(int)
df['year_2009'] = (df['YEAR'] == 2009).astype(int)
df['year_2010'] = (df['YEAR'] == 2010).astype(int)
df['year_2013'] = (df['YEAR'] == 2013).astype(int)
df['year_2014'] = (df['YEAR'] == 2014).astype(int)
df['year_2015'] = (df['YEAR'] == 2015).astype(int)
df['year_2016'] = (df['YEAR'] == 2016).astype(int)

# Interactions
for yr in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_year_{yr}'] = df['ELIGIBLE'] * df[f'year_{yr}']

event_study_formula = 'FT ~ ELIGIBLE + year_2008 + year_2009 + year_2010 + year_2013 + year_2014 + year_2015 + year_2016 + ELIGIBLE_year_2008 + ELIGIBLE_year_2009 + ELIGIBLE_year_2010 + ELIGIBLE_year_2013 + ELIGIBLE_year_2014 + ELIGIBLE_year_2015 + ELIGIBLE_year_2016'

model_event = smf.ols(event_study_formula, data=df).fit(cov_type='HC1')

print("\nEvent Study Coefficients (interaction terms):")
years_event = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
for yr in years_event:
    coef = model_event.params[f'ELIGIBLE_year_{yr}']
    se = model_event.bse[f'ELIGIBLE_year_{yr}']
    ci_low = model_event.conf_int().loc[f'ELIGIBLE_year_{yr}', 0]
    ci_high = model_event.conf_int().loc[f'ELIGIBLE_year_{yr}', 1]
    pval = model_event.pvalues[f'ELIGIBLE_year_{yr}']
    print(f"Year {yr}: coef = {coef:.4f}, SE = {se:.4f}, p = {pval:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    event_coefs.append({'year': yr, 'coef': coef, 'se': se, 'ci_low': ci_low, 'ci_high': ci_high})

# Add reference year (2011)
event_coefs.append({'year': 2011, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
event_df = pd.DataFrame(event_coefs).sort_values('year')

# Demographic subgroup analysis
print("\n" + "=" * 80)
print("SUBGROUP ANALYSIS")
print("=" * 80)

print("\n--- By Sex ---")
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
    print(f"{sex_name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f}, SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}, p = {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}, n = {len(df_sub)}")

print("\n--- By Education ---")
for educ in df['EDUC_RECODE'].unique():
    df_sub = df[df['EDUC_RECODE'] == educ]
    if len(df_sub) > 100:
        model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
        print(f"{educ}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f}, SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}, p = {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}, n = {len(df_sub)}")

print("\n--- By Marital Status ---")
for mar_name, mar_condition in [('Married', df['married']==1), ('Unmarried', df['married']==0)]:
    df_sub = df[mar_condition]
    model_sub = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub).fit(cov_type='HC1')
    print(f"{mar_name}: DiD = {model_sub.params['ELIGIBLE_AFTER']:.4f}, SE = {model_sub.bse['ELIGIBLE_AFTER']:.4f}, p = {model_sub.pvalues['ELIGIBLE_AFTER']:.4f}, n = {len(df_sub)}")

# Summary statistics table
print("\n" + "=" * 80)
print("SUMMARY STATISTICS BY GROUP")
print("=" * 80)

summary_vars = ['FT', 'AGE', 'female', 'married', 'UHRSWORK', 'FAMSIZE', 'NCHILD']

print("\n--- Pre-DACA Period ---")
for eligible in [1, 0]:
    group = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
    df_sub = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==0)]
    print(f"\n{group}:")
    for var in summary_vars:
        if var in df_sub.columns:
            print(f"  {var}: mean = {df_sub[var].mean():.3f}, sd = {df_sub[var].std():.3f}")

print("\n--- Post-DACA Period ---")
for eligible in [1, 0]:
    group = "Treatment (26-30)" if eligible == 1 else "Control (31-35)"
    df_sub = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==1)]
    print(f"\n{group}:")
    for var in summary_vars:
        if var in df_sub.columns:
            print(f"  {var}: mean = {df_sub[var].mean():.3f}, sd = {df_sub[var].std():.3f}")

# State-level policy variables
print("\n" + "=" * 80)
print("STATE POLICY VARIABLES DISTRIBUTION")
print("=" * 80)

policy_vars = ['DRIVERSLICENSES', 'INSTATETUITION', 'STATEFINANCIALAID', 'EVERIFY', 'SECURECOMMUNITIES']
for var in policy_vars:
    if var in df.columns:
        print(f"{var}: {df[var].value_counts().to_dict()}")

# Robustness: Triple difference with state policy
print("\n" + "=" * 80)
print("ROBUSTNESS: DiD WITH STATE POLICY INTERACTIONS")
print("=" * 80)

# Interaction with drivers license policy
df['DL_ELIGIBLE_AFTER'] = df['DRIVERSLICENSES'] * df['ELIGIBLE_AFTER']
model_robust = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + DRIVERSLICENSES + DL_ELIGIBLE_AFTER',
                       data=df).fit(cov_type='HC1')
print("\nModel with Drivers License Policy Interaction:")
print(f"ELIGIBLE_AFTER: {model_robust.params['ELIGIBLE_AFTER']:.4f} (p={model_robust.pvalues['ELIGIBLE_AFTER']:.4f})")
print(f"DL_ELIGIBLE_AFTER: {model_robust.params['DL_ELIGIBLE_AFTER']:.4f} (p={model_robust.pvalues['DL_ELIGIBLE_AFTER']:.4f})")

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY - PREFERRED ESTIMATE")
print("=" * 80)

print("\nPreferred Model: Model 5 (DiD with basic covariates and robust SEs)")
print(f"DiD Estimate (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error (robust): {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-statistic: {model5.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
ci = model5.conf_int().loc['ELIGIBLE_AFTER']
print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
print(f"\nSample Size: {int(model5.nobs)}")
print(f"R-squared: {model5.rsquared:.4f}")

# Interpretation
effect_pct = model5.params['ELIGIBLE_AFTER'] * 100
print(f"\nInterpretation: DACA eligibility is associated with a {effect_pct:.2f} percentage point")
if effect_pct > 0:
    print("increase in the probability of full-time employment for the treatment group")
else:
    print("decrease in the probability of full-time employment for the treatment group")
print("relative to the control group, after accounting for general time trends.")

# Save key results to file
results_dict = {
    'model': ['Basic DiD', 'DiD + Robust SE', 'DiD + Clustered SE', 'DiD + Year FE',
              'DiD + Covariates', 'DiD + State FE', 'Weighted DiD'],
    'coefficient': [model1.params['ELIGIBLE_AFTER'], model2.params['ELIGIBLE_AFTER'],
                   model3.params['ELIGIBLE_AFTER'], model4.params['ELIGIBLE_AFTER'],
                   model5.params['ELIGIBLE_AFTER'], model6.params['ELIGIBLE_AFTER'],
                   model7.params['ELIGIBLE_AFTER']],
    'std_error': [model1.bse['ELIGIBLE_AFTER'], model2.bse['ELIGIBLE_AFTER'],
                 model3.bse['ELIGIBLE_AFTER'], model4.bse['ELIGIBLE_AFTER'],
                 model5.bse['ELIGIBLE_AFTER'], model6.bse['ELIGIBLE_AFTER'],
                 model7.bse['ELIGIBLE_AFTER']],
    'p_value': [model1.pvalues['ELIGIBLE_AFTER'], model2.pvalues['ELIGIBLE_AFTER'],
               model3.pvalues['ELIGIBLE_AFTER'], model4.pvalues['ELIGIBLE_AFTER'],
               model5.pvalues['ELIGIBLE_AFTER'], model6.pvalues['ELIGIBLE_AFTER'],
               model7.pvalues['ELIGIBLE_AFTER']],
    'n_obs': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs),
             int(model5.nobs), int(model6.nobs), int(model7.nobs)],
    'r_squared': [model1.rsquared, model2.rsquared, model3.rsquared, model4.rsquared,
                 model5.rsquared, model6.rsquared, model7.rsquared]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('regression_results.csv', index=False)
print("\nRegression results saved to regression_results.csv")

# Create visualizations
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Figure 1: Parallel trends plot
fig, ax = plt.subplots(figsize=(10, 6))
years = yearly_means.index
ax.plot(years, yearly_means['Control (31-35)'], 'b-o', label='Control (ages 31-35)', linewidth=2, markersize=8)
ax.plot(years, yearly_means['Treatment (26-30)'], 'r-s', label='Treatment (ages 26-30)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Group', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.7)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_parallel_trends.png")

# Figure 2: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_df['year'], event_df['coef'],
            yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8, color='darkblue')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Dynamic Treatment Effects of DACA Eligibility', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_event_study.png")

# Figure 3: DiD illustration
fig, ax = plt.subplots(figsize=(10, 6))
# Pre-period means
pre_control = ft_control_before
pre_treat = ft_eligible_before
# Post-period means
post_control = ft_control_after
post_treat = ft_eligible_after
# Counterfactual for treatment
counterfactual = pre_treat + (post_control - pre_control)

ax.plot([0, 1], [pre_control, post_control], 'b-o', label='Control Group', linewidth=2, markersize=10)
ax.plot([0, 1], [pre_treat, post_treat], 'r-s', label='Treatment Group', linewidth=2, markersize=10)
ax.plot([0, 1], [pre_treat, counterfactual], 'r--', label='Treatment Counterfactual', linewidth=2, alpha=0.5)
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.05, (post_treat + counterfactual)/2, f'DiD = {did_estimate:.3f}', fontsize=12, color='green')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Illustration', fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_did_illustration.png")

# Figure 4: Distribution of full-time work by group
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

groups = [
    ('Treatment, Pre-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==0)),
    ('Treatment, Post-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==1)),
    ('Control, Pre-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==0)),
    ('Control, Post-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==1))
]

for idx, (title, mask) in enumerate(groups):
    ax = axes[idx // 2, idx % 2]
    df_sub = df[mask]
    ft_rate = df_sub['FT'].mean()
    ax.bar(['Not Full-Time', 'Full-Time'], [1-ft_rate, ft_rate], color=['lightcoral', 'lightgreen'])
    ax.set_title(f'{title}\n(n={len(df_sub)}, FT rate={ft_rate:.3f})', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Proportion')

plt.suptitle('Full-Time Employment Distribution by Group and Period', fontsize=14)
plt.tight_layout()
plt.savefig('figure4_ft_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_ft_distribution.png")

# Figure 5: Coefficient comparison across models
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Basic\nDiD', 'Robust\nSE', 'Clustered\nSE', 'Year\nFE', 'With\nCovariates',
          'State\nFE', 'Weighted\nDiD']
coefs = results_df['coefficient'].values
ses = results_df['std_error'].values

ax.errorbar(range(len(models)), coefs, yerr=1.96*ses, fmt='o', capsize=5, capthick=2,
            markersize=10, color='darkblue')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models)
ax.set_ylabel('DiD Coefficient (95% CI)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure5_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure5_coefficient_comparison.png")

# Figure 6: Hours worked distribution
fig, ax = plt.subplots(figsize=(10, 6))
df_treat = df[df['ELIGIBLE']==1]
df_control = df[df['ELIGIBLE']==0]
ax.hist(df_treat['UHRSWORK'], bins=50, alpha=0.5, label='Treatment (26-30)', density=True)
ax.hist(df_control['UHRSWORK'], bins=50, alpha=0.5, label='Control (31-35)', density=True)
ax.axvline(x=35, color='red', linestyle='--', linewidth=2, label='Full-Time Threshold (35 hrs)')
ax.set_xlabel('Usual Hours Worked per Week', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Distribution of Hours Worked by Eligibility Group', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig('figure6_hours_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure6_hours_distribution.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

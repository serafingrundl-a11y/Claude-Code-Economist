"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment

Author: Replication 61
Date: 2026-01-27
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
print("DACA REPLICATION STUDY - ANALYSIS")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n1. LOADING DATA")
print("-" * 40)

# Load the numeric version for analysis
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"Total observations: {len(df):,}")
print(f"Variables: {len(df.columns)}")

# ============================================================================
# DATA EXPLORATION
# ============================================================================
print("\n2. DATA EXPLORATION")
print("-" * 40)

# Check key variables
print("\nKey variable summary:")
print(f"FT (full-time): {df['FT'].value_counts().to_dict()}")
print(f"ELIGIBLE: {df['ELIGIBLE'].value_counts().to_dict()}")
print(f"AFTER: {df['AFTER'].value_counts().to_dict()}")
print(f"Years in data: {sorted(df['YEAR'].unique())}")

# Sample sizes by treatment groups
print("\nSample sizes by group:")
cross_tab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
cross_tab.columns = ['Pre (2008-2011)', 'Post (2013-2016)', 'Total']
cross_tab.index = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(cross_tab)

# Check age distribution
print("\nAge distribution:")
print(df['AGE'].describe())

# Check AGE_IN_JUNE_2012 distribution
print("\nAge in June 2012 distribution:")
print(df['AGE_IN_JUNE_2012'].describe())
print(f"\nBy ELIGIBLE status:")
print(df.groupby('ELIGIBLE')['AGE_IN_JUNE_2012'].describe())

# Full-time employment rates by group
print("\nFull-time employment rates by group (unweighted):")
ft_by_group = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'count', 'sum'])
ft_by_group.columns = ['FT_Rate', 'N', 'N_FT']
print(ft_by_group)

# Weighted full-time employment rates
print("\nFull-time employment rates by group (weighted):")
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

ft_weighted = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean)
print(ft_weighted)

# ============================================================================
# DIFFERENCE-IN-DIFFERENCES: SIMPLE 2x2
# ============================================================================
print("\n3. DIFFERENCE-IN-DIFFERENCES: SIMPLE 2x2 TABLE")
print("-" * 40)

# Calculate means by group
means = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

print("\nWeighted FT rates:")
print("                      Pre        Post")
print(f"Control (31-35):    {means.loc[0, 0]:.4f}    {means.loc[0, 1]:.4f}")
print(f"Treatment (26-30):  {means.loc[1, 0]:.4f}    {means.loc[1, 1]:.4f}")

# Calculate DiD
pre_diff = means.loc[1, 0] - means.loc[0, 0]
post_diff = means.loc[1, 1] - means.loc[0, 1]
did_simple = (means.loc[1, 1] - means.loc[1, 0]) - (means.loc[0, 1] - means.loc[0, 0])

print(f"\nPre-period difference (Treat - Control): {pre_diff:.4f}")
print(f"Post-period difference (Treat - Control): {post_diff:.4f}")

print(f"\nChange for Treatment group: {means.loc[1, 1] - means.loc[1, 0]:.4f}")
print(f"Change for Control group: {means.loc[0, 1] - means.loc[0, 0]:.4f}")

print(f"\n*** DiD Estimate (simple): {did_simple:.4f} ***")

# ============================================================================
# REGRESSION-BASED DiD: LINEAR PROBABILITY MODEL
# ============================================================================
print("\n4. REGRESSION-BASED DiD: LINEAR PROBABILITY MODEL")
print("-" * 40)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD (weighted)
print("\nModel 1: Basic DiD (no covariates)")
model1 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                 data=df, weights=df['PERWT'])
results1 = model1.fit()
print(results1.summary().tables[1])

# Get robust (HC1) standard errors
results1_robust = model1.fit(cov_type='HC1')
print("\nWith robust standard errors:")
print(f"DiD coefficient: {results1_robust.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {results1_robust.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-stat: {results1_robust.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {results1_robust.pvalues['ELIGIBLE_AFTER']:.4f}")

# Model 2: DiD with state fixed effects and clustered SEs
print("\n" + "=" * 40)
print("Model 2: DiD with State Fixed Effects")
print("=" * 40)

# Create state dummies
state_dummies = pd.get_dummies(df['STATEFIP'], prefix='state', drop_first=True)
df_with_states = pd.concat([df, state_dummies], axis=1)

# Model with state FE
state_cols = [col for col in df_with_states.columns if col.startswith('state_')]
formula_state = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + ' + ' + '.join(state_cols)
model2 = smf.wls(formula_state, data=df_with_states, weights=df_with_states['PERWT'])
results2 = model2.fit(cov_type='cluster', cov_kwds={'groups': df_with_states['STATEFIP']})

print(f"\nDiD coefficient: {results2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {results2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-stat: {results2.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {results2.pvalues['ELIGIBLE_AFTER']:.4f}")
conf_int = results2.conf_int().loc['ELIGIBLE_AFTER']
print(f"95% CI: [{conf_int[0]:.4f}, {conf_int[1]:.4f}]")

# Model 3: DiD with state FE and year FE
print("\n" + "=" * 40)
print("Model 3: DiD with State and Year Fixed Effects")
print("=" * 40)

# Create year dummies
year_dummies = pd.get_dummies(df['YEAR'], prefix='year', drop_first=True)
df_full = pd.concat([df_with_states, year_dummies], axis=1)

year_cols = [col for col in df_full.columns if col.startswith('year_')]
formula_full = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + ' + ' + '.join(state_cols) + ' + ' + ' + '.join(year_cols)

model3 = smf.wls(formula_full, data=df_full, weights=df_full['PERWT'])
results3 = model3.fit(cov_type='cluster', cov_kwds={'groups': df_full['STATEFIP']})

print(f"\nDiD coefficient: {results3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {results3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-stat: {results3.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {results3.pvalues['ELIGIBLE_AFTER']:.4f}")
conf_int3 = results3.conf_int().loc['ELIGIBLE_AFTER']
print(f"95% CI: [{conf_int3[0]:.4f}, {conf_int3[1]:.4f}]")

# Model 4: DiD with covariates
print("\n" + "=" * 40)
print("Model 4: DiD with State/Year FE and Individual Covariates")
print("=" * 40)

# Add individual covariates
# Sex (1=Male, 2=Female)
df_full['FEMALE'] = (df_full['SEX'] == 2).astype(int)

# Marital status dummies
df_full['MARRIED'] = (df_full['MARST'] == 1).astype(int)

# Education recoded - create manual dummies to avoid column name issues
df_full['EDUC_HS'] = (df_full['EDUC_RECODE'] == 'High School Degree').astype(int)
df_full['EDUC_SOMECOLL'] = (df_full['EDUC_RECODE'] == 'Some College').astype(int)
df_full['EDUC_AA'] = (df_full['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df_full['EDUC_BA'] = (df_full['EDUC_RECODE'] == 'BA+').astype(int)
educ_cols = ['EDUC_HS', 'EDUC_SOMECOLL', 'EDUC_AA', 'EDUC_BA']

# Create formula with covariates
formula_cov = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE + ' + \
              ' + '.join(educ_cols) + ' + ' + ' + '.join(state_cols) + ' + ' + ' + '.join(year_cols)

model4 = smf.wls(formula_cov, data=df_full, weights=df_full['PERWT'])
results4 = model4.fit(cov_type='cluster', cov_kwds={'groups': df_full['STATEFIP']})

print(f"\nDiD coefficient: {results4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered SE: {results4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"t-stat: {results4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {results4.pvalues['ELIGIBLE_AFTER']:.4f}")
conf_int4 = results4.conf_int().loc['ELIGIBLE_AFTER']
print(f"95% CI: [{conf_int4[0]:.4f}, {conf_int4[1]:.4f}]")

# ============================================================================
# ROBUSTNESS CHECKS
# ============================================================================
print("\n" + "=" * 80)
print("5. ROBUSTNESS CHECKS")
print("=" * 80)

# 5.1 Unweighted Analysis
print("\n5.1 Unweighted Analysis")
print("-" * 40)
model_unw = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df)
results_unw = model_unw.fit(cov_type='HC1')
print(f"DiD coefficient (unweighted): {results_unw.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust SE: {results_unw.bse['ELIGIBLE_AFTER']:.4f}")
print(f"p-value: {results_unw.pvalues['ELIGIBLE_AFTER']:.4f}")

# 5.2 Logit Model
print("\n5.2 Logit Model (Marginal Effects)")
print("-" * 40)
try:
    logit_model = smf.logit('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df)
    logit_results = logit_model.fit(disp=0)

    # Get marginal effects at means
    marg_eff = logit_results.get_margeff()
    idx = list(logit_results.params.index).index('ELIGIBLE_AFTER')
    print(f"Average marginal effect of interaction: {marg_eff.margeff[idx]:.4f}")
    print(f"SE: {marg_eff.margeff_se[idx]:.4f}")
except Exception as e:
    print(f"Logit model error: {e}")

# 5.3 By Gender
print("\n5.3 Heterogeneous Effects by Gender")
print("-" * 40)

# Male only
df_male = df[df['SEX'] == 1].copy()
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                     data=df_male, weights=df_male['PERWT'])
results_male = model_male.fit(cov_type='HC1')
print(f"Male DiD coefficient: {results_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE: {results_male.bse['ELIGIBLE_AFTER']:.4f}, p={results_male.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  N = {len(df_male):,}")

# Female only
df_female = df[df['SEX'] == 2].copy()
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                       data=df_female, weights=df_female['PERWT'])
results_female = model_female.fit(cov_type='HC1')
print(f"Female DiD coefficient: {results_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  SE: {results_female.bse['ELIGIBLE_AFTER']:.4f}, p={results_female.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  N = {len(df_female):,}")

# 5.4 Event Study / Year-by-Year Effects
print("\n5.4 Event Study (Year-by-Year Interaction Effects)")
print("-" * 40)

# Create year interaction terms
for year in df['YEAR'].unique():
    df[f'ELIG_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

# Drop base year (2008) interaction
year_interact_cols = [f'ELIG_YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2008]
formula_event = 'FT ~ ELIGIBLE + ' + ' + '.join([f'ELIG_YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2008])
formula_event += ' + C(YEAR)'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT'])
results_event = model_event.fit(cov_type='HC1')

print("\nYear-specific treatment effects (relative to 2008):")
for year in sorted(df['YEAR'].unique()):
    if year != 2008:
        coef = results_event.params[f'ELIG_YEAR_{year}']
        se = results_event.bse[f'ELIG_YEAR_{year}']
        print(f"  {year}: {coef:.4f} (SE: {se:.4f})")

# 5.5 Placebo Test - Pre-trends
print("\n5.5 Placebo Test - Pre-Period Only (2008-2011)")
print("-" * 40)

df_pre = df[df['AFTER'] == 0].copy()
df_pre['POST_2010'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_POST2010'] = df_pre['ELIGIBLE'] * df_pre['POST_2010']

model_placebo = smf.wls('FT ~ ELIGIBLE + POST_2010 + ELIGIBLE_POST2010',
                        data=df_pre, weights=df_pre['PERWT'])
results_placebo = model_placebo.fit(cov_type='HC1')

print(f"Placebo DiD (2010-2011 vs 2008-2009): {results_placebo.params['ELIGIBLE_POST2010']:.4f}")
print(f"  SE: {results_placebo.bse['ELIGIBLE_POST2010']:.4f}")
print(f"  p-value: {results_placebo.pvalues['ELIGIBLE_POST2010']:.4f}")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "=" * 80)
print("6. SUMMARY OF RESULTS")
print("=" * 80)

print("\n" + "-" * 100)
print(f"{'Model':<45} {'Coefficient':>12} {'SE':>12} {'95% CI':>25} {'p-value':>12}")
print("-" * 100)

# Store results for summary
models_summary = [
    ("(1) Basic DiD", results1_robust.params['ELIGIBLE_AFTER'],
     results1_robust.bse['ELIGIBLE_AFTER'], results1_robust.conf_int().loc['ELIGIBLE_AFTER'],
     results1_robust.pvalues['ELIGIBLE_AFTER']),
    ("(2) State FE + Clustered SE", results2.params['ELIGIBLE_AFTER'],
     results2.bse['ELIGIBLE_AFTER'], results2.conf_int().loc['ELIGIBLE_AFTER'],
     results2.pvalues['ELIGIBLE_AFTER']),
    ("(3) State + Year FE", results3.params['ELIGIBLE_AFTER'],
     results3.bse['ELIGIBLE_AFTER'], results3.conf_int().loc['ELIGIBLE_AFTER'],
     results3.pvalues['ELIGIBLE_AFTER']),
    ("(4) State + Year FE + Covariates", results4.params['ELIGIBLE_AFTER'],
     results4.bse['ELIGIBLE_AFTER'], results4.conf_int().loc['ELIGIBLE_AFTER'],
     results4.pvalues['ELIGIBLE_AFTER']),
    ("(5) Unweighted", results_unw.params['ELIGIBLE_AFTER'],
     results_unw.bse['ELIGIBLE_AFTER'], results_unw.conf_int().loc['ELIGIBLE_AFTER'],
     results_unw.pvalues['ELIGIBLE_AFTER']),
]

for name, coef, se, ci, pval in models_summary:
    ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.1:
        stars = "*"
    print(f"{name:<45} {coef:>12.4f} {se:>12.4f} {ci_str:>25} {pval:>10.4f} {stars}")

print("-" * 100)
print("Notes: * p<0.1, ** p<0.05, *** p<0.01")
print(f"Sample size: N = {len(df):,}")

# ============================================================================
# PREFERRED ESTIMATE
# ============================================================================
print("\n" + "=" * 80)
print("7. PREFERRED ESTIMATE")
print("=" * 80)

# Model 3 with state and year FE is preferred for causal interpretation
preferred_coef = results3.params['ELIGIBLE_AFTER']
preferred_se = results3.bse['ELIGIBLE_AFTER']
preferred_ci = results3.conf_int().loc['ELIGIBLE_AFTER']
preferred_pval = results3.pvalues['ELIGIBLE_AFTER']

print(f"\nPreferred specification: State and Year Fixed Effects (Model 3)")
print(f"DiD Estimate: {preferred_coef:.4f}")
print(f"Standard Error (clustered at state): {preferred_se:.4f}")
print(f"95% Confidence Interval: [{preferred_ci[0]:.4f}, {preferred_ci[1]:.4f}]")
print(f"p-value: {preferred_pval:.4f}")
print(f"Sample Size: {len(df):,}")

print("\nInterpretation:")
if preferred_coef > 0:
    print(f"DACA eligibility is associated with a {abs(preferred_coef)*100:.2f} percentage point")
    print("INCREASE in the probability of full-time employment.")
else:
    print(f"DACA eligibility is associated with a {abs(preferred_coef)*100:.2f} percentage point")
    print("DECREASE in the probability of full-time employment.")

if preferred_pval < 0.05:
    print("This effect is statistically significant at the 5% level.")
elif preferred_pval < 0.1:
    print("This effect is statistically significant at the 10% level.")
else:
    print("This effect is NOT statistically significant at conventional levels.")

# ============================================================================
# SAVE RESULTS FOR LATEX
# ============================================================================
print("\n" + "=" * 80)
print("8. SAVING RESULTS")
print("=" * 80)

# Create results dictionary for later use
results_dict = {
    'n_total': len(df),
    'n_treatment_pre': len(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]),
    'n_treatment_post': len(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]),
    'n_control_pre': len(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]),
    'n_control_post': len(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]),
    'ft_treat_pre': means.loc[1, 0],
    'ft_treat_post': means.loc[1, 1],
    'ft_control_pre': means.loc[0, 0],
    'ft_control_post': means.loc[0, 1],
    'did_simple': did_simple,
    'coef_model1': results1_robust.params['ELIGIBLE_AFTER'],
    'se_model1': results1_robust.bse['ELIGIBLE_AFTER'],
    'pval_model1': results1_robust.pvalues['ELIGIBLE_AFTER'],
    'coef_model2': results2.params['ELIGIBLE_AFTER'],
    'se_model2': results2.bse['ELIGIBLE_AFTER'],
    'pval_model2': results2.pvalues['ELIGIBLE_AFTER'],
    'coef_model3': results3.params['ELIGIBLE_AFTER'],
    'se_model3': results3.bse['ELIGIBLE_AFTER'],
    'pval_model3': results3.pvalues['ELIGIBLE_AFTER'],
    'ci_model3_low': conf_int3[0],
    'ci_model3_high': conf_int3[1],
    'coef_model4': results4.params['ELIGIBLE_AFTER'],
    'se_model4': results4.bse['ELIGIBLE_AFTER'],
    'pval_model4': results4.pvalues['ELIGIBLE_AFTER'],
    'coef_male': results_male.params['ELIGIBLE_AFTER'],
    'se_male': results_male.bse['ELIGIBLE_AFTER'],
    'pval_male': results_male.pvalues['ELIGIBLE_AFTER'],
    'n_male': len(df_male),
    'coef_female': results_female.params['ELIGIBLE_AFTER'],
    'se_female': results_female.bse['ELIGIBLE_AFTER'],
    'pval_female': results_female.pvalues['ELIGIBLE_AFTER'],
    'n_female': len(df_female),
    'placebo_coef': results_placebo.params['ELIGIBLE_POST2010'],
    'placebo_se': results_placebo.bse['ELIGIBLE_POST2010'],
    'placebo_pval': results_placebo.pvalues['ELIGIBLE_POST2010'],
}

# Save to file for use in LaTeX
import json
with open('results_summary.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("Results saved to results_summary.json")

# ============================================================================
# CREATE FIGURES
# ============================================================================
print("\nCreating figures...")

# Figure 1: Parallel trends
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate yearly FT rates by group
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

years = yearly_rates.index
ax.plot(years, yearly_rates[0], 'b-o', label='Control (ages 31-35 in 2012)', linewidth=2, markersize=8)
ax.plot(years, yearly_rates[1], 'r-s', label='Treatment (ages 26-30 in 2012)', linewidth=2, markersize=8)

# Add vertical line for DACA
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved: figure1_parallel_trends.png")

# Figure 2: Event Study
fig, ax = plt.subplots(figsize=(10, 6))

event_years = [y for y in sorted(df['YEAR'].unique()) if y != 2008]
coefs = []
ses = []

for year in event_years:
    coefs.append(results_event.params[f'ELIG_YEAR_{year}'])
    ses.append(results_event.bse[f'ELIG_YEAR_{year}'])

# Add 2008 as reference (0 effect)
all_years = [2008] + event_years
all_coefs = [0] + coefs
all_ses = [0] + ses

# Calculate confidence intervals
ci_low = [c - 1.96*s for c, s in zip(all_coefs, all_ses)]
ci_high = [c + 1.96*s for c, s in zip(all_coefs, all_ses)]

ax.plot(all_years, all_coefs, 'ko-', linewidth=2, markersize=8)
ax.fill_between(all_years, ci_low, ci_high, alpha=0.3, color='blue')

# Add reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (Relative to 2008)', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(all_years)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 saved: figure2_event_study.png")

# Figure 3: DiD Illustration
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left panel: Raw data
ax1 = axes[0]
x_pre = 0
x_post = 1
width = 0.35

ax1.bar(x_pre - width/2, means.loc[0, 0], width, label='Control (Pre)', color='blue', alpha=0.7)
ax1.bar(x_pre + width/2, means.loc[1, 0], width, label='Treatment (Pre)', color='red', alpha=0.7)
ax1.bar(x_post - width/2, means.loc[0, 1], width, label='Control (Post)', color='blue')
ax1.bar(x_post + width/2, means.loc[1, 1], width, label='Treatment (Post)', color='red')

ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment by Group and Period', fontsize=14)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax1.legend(loc='upper left', fontsize=9)
ax1.set_ylim(0, 0.7)

# Right panel: DiD visualization
ax2 = axes[1]
ax2.plot([0, 1], [means.loc[0, 0], means.loc[0, 1]], 'b-o', linewidth=2, markersize=10, label='Control')
ax2.plot([0, 1], [means.loc[1, 0], means.loc[1, 1]], 'r-s', linewidth=2, markersize=10, label='Treatment')

# Counterfactual line
counterfactual_post = means.loc[1, 0] + (means.loc[0, 1] - means.loc[0, 0])
ax2.plot([0, 1], [means.loc[1, 0], counterfactual_post], 'r--', linewidth=2, alpha=0.5, label='Counterfactual')

# Draw DiD arrow
ax2.annotate('', xy=(1.05, means.loc[1, 1]), xytext=(1.05, counterfactual_post),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax2.text(1.1, (means.loc[1, 1] + counterfactual_post)/2, f'DiD\n{did_simple:.3f}', fontsize=11, color='green', fontweight='bold')

ax2.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax2.set_title('Difference-in-Differences Visualization', fontsize=14)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax2.legend(loc='upper left', fontsize=9)
ax2.set_ylim(0.35, 0.65)
ax2.set_xlim(-0.2, 1.3)

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 saved: figure3_did_illustration.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

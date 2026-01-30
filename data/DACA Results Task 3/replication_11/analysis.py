"""
DACA Replication Analysis - Replication 11
Effect of DACA Eligibility on Full-Time Employment

This script performs a difference-in-differences analysis to estimate
the causal effect of DACA eligibility on full-time employment among
Hispanic-Mexican, Mexican-born individuals in the United States.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("DACA REPLICATION ANALYSIS - Study 11")
print("=" * 70)

# =============================================================================
# STEP 1: Load and Examine Data
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING AND EXAMINING DATA")
print("=" * 70)

# Load the numeric version for analysis
df = pd.read_csv('data/prepared_data_numeric_version.csv')

print(f"\nDataset shape: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"\nKey variables present:")
print(f"  - ELIGIBLE: {df['ELIGIBLE'].notna().sum()} non-missing values")
print(f"  - AFTER: {df['AFTER'].notna().sum()} non-missing values")
print(f"  - FT: {df['FT'].notna().sum()} non-missing values")
print(f"  - PERWT: {df['PERWT'].notna().sum()} non-missing values")
print(f"  - YEAR: {df['YEAR'].notna().sum()} non-missing values")

# Check variable distributions
print(f"\nELIGIBLE distribution:")
print(df['ELIGIBLE'].value_counts().sort_index())
print(f"\nAFTER distribution:")
print(df['AFTER'].value_counts().sort_index())
print(f"\nFT distribution:")
print(df['FT'].value_counts().sort_index())
print(f"\nYEAR distribution:")
print(df['YEAR'].value_counts().sort_index())

# =============================================================================
# STEP 2: Descriptive Statistics
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: DESCRIPTIVE STATISTICS")
print("=" * 70)

# Create treatment-period groups
df['group'] = np.where(
    (df['ELIGIBLE'] == 1) & (df['AFTER'] == 1), 'Treated_Post',
    np.where(
        (df['ELIGIBLE'] == 1) & (df['AFTER'] == 0), 'Treated_Pre',
        np.where(
            (df['ELIGIBLE'] == 0) & (df['AFTER'] == 1), 'Control_Post',
            'Control_Pre'
        )
    )
)

# Calculate group sizes
print("\nSample sizes by group:")
group_sizes = df.groupby(['ELIGIBLE', 'AFTER']).size().reset_index(name='n')
print(group_sizes)

# Calculate weighted means of FT by group
print("\nFull-Time Employment Rates by Group (Weighted):")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).reset_index(name='FT_Rate')
print(ft_rates)

# Calculate unweighted means for comparison
print("\nFull-Time Employment Rates by Group (Unweighted):")
ft_rates_unweighted = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().reset_index(name='FT_Rate')
print(ft_rates_unweighted)

# Detailed descriptive statistics by treatment group
print("\n" + "-" * 50)
print("Descriptive Statistics by Treatment Group")
print("-" * 50)

# Demographics
demo_vars = ['AGE', 'SEX', 'FAMSIZE', 'NCHILD']

for eligible in [0, 1]:
    group_name = "Treatment (ELIGIBLE=1)" if eligible == 1 else "Control (ELIGIBLE=0)"
    print(f"\n{group_name}:")
    subset = df[df['ELIGIBLE'] == eligible]

    for var in demo_vars:
        if var in df.columns:
            weighted_mean = np.average(subset[var], weights=subset['PERWT'])
            weighted_std = np.sqrt(np.average((subset[var] - weighted_mean)**2, weights=subset['PERWT']))
            print(f"  {var}: Mean = {weighted_mean:.3f}, SD = {weighted_std:.3f}")

# =============================================================================
# STEP 3: Pre-Treatment Parallel Trends Check
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: PARALLEL TRENDS ASSESSMENT")
print("=" * 70)

# Calculate FT rates by year and treatment group
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_Rate': np.average(x['FT'], weights=x['PERWT']),
        'n': len(x),
        'weighted_n': x['PERWT'].sum()
    })
).reset_index()

print("\nFull-Time Employment Rates by Year and Treatment Group:")
print(yearly_ft.pivot(index='YEAR', columns='ELIGIBLE', values='FT_Rate').to_string())

# Pre-treatment trends
pre_treatment = yearly_ft[yearly_ft['YEAR'] < 2012]
print("\nPre-Treatment Trends (2008-2011):")
print(pre_treatment.pivot(index='YEAR', columns='ELIGIBLE', values='FT_Rate').to_string())

# Calculate year-over-year changes in pre-period
pre_pivot = pre_treatment.pivot(index='YEAR', columns='ELIGIBLE', values='FT_Rate')
pre_pivot.columns = ['Control', 'Treatment']
pre_pivot['Difference'] = pre_pivot['Treatment'] - pre_pivot['Control']
print("\nPre-Treatment Differences (Treatment - Control):")
print(pre_pivot['Difference'].to_string())

# =============================================================================
# STEP 4: Main Difference-in-Differences Analysis
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: DIFFERENCE-IN-DIFFERENCES ANALYSIS")
print("=" * 70)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# ----- Model 1: Basic DiD (OLS, no weights) -----
print("\n--- Model 1: Basic OLS DiD (No Weights) ---")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(model1.summary().tables[1])
print(f"\nDiD Estimate (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")

# ----- Model 2: DiD with Survey Weights -----
print("\n--- Model 2: Weighted DiD (PERWT) ---")
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
print(model2.summary().tables[1])
print(f"\nDiD Estimate (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"Standard Error: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

# ----- Model 3: Weighted DiD with Robust Standard Errors -----
print("\n--- Model 3: Weighted DiD with Robust (HC1) Standard Errors ---")
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')
print(model3.summary().tables[1])
print(f"\nDiD Estimate (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"Robust Standard Error: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")

# ----- Model 4: Weighted DiD with State Clustered Standard Errors -----
print("\n--- Model 4: Weighted DiD with State-Clustered Standard Errors ---")
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)
print(model4.summary().tables[1])
print(f"\nDiD Estimate (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")

# =============================================================================
# STEP 5: DiD with Covariates
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: DIFFERENCE-IN-DIFFERENCES WITH COVARIATES")
print("=" * 70)

# Check available covariates
print("\nAvailable covariates in dataset:")
covariate_candidates = ['SEX', 'FAMSIZE', 'NCHILD', 'EDUC', 'MARST', 'METRO', 'LFPR', 'UNEMP']
for var in covariate_candidates:
    if var in df.columns:
        print(f"  - {var}: {df[var].nunique()} unique values")

# Create covariate model - Sex, education, marital status, metro status
# Note: SEX in IPUMS is coded 1=Male, 2=Female
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# ----- Model 5: DiD with demographic covariates -----
print("\n--- Model 5: Weighted DiD with Demographic Covariates ---")
model5 = smf.wls(
    'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + FAMSIZE + NCHILD + C(MARST)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Print key coefficients
print(f"\nKey Coefficients:")
print(f"  Intercept: {model5.params['Intercept']:.4f}")
print(f"  ELIGIBLE: {model5.params['ELIGIBLE']:.4f}")
print(f"  AFTER: {model5.params['AFTER']:.4f}")
print(f"  ELIGIBLE_AFTER (DiD): {model5.params['ELIGIBLE_AFTER']:.4f} (SE: {model5.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  FEMALE: {model5.params['FEMALE']:.4f}")
print(f"  FAMSIZE: {model5.params['FAMSIZE']:.4f}")
print(f"  NCHILD: {model5.params['NCHILD']:.4f}")
print(f"\n95% CI for DiD: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model5.rsquared:.4f}")

# ----- Model 6: Full model with state and year fixed effects -----
print("\n--- Model 6: DiD with State and Year Fixed Effects ---")
model6 = smf.wls(
    'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + FAMSIZE + NCHILD + C(MARST) + C(STATEFIP) + C(YEAR)',
    data=df, weights=df['PERWT']
).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

print(f"\nKey Coefficients (State and Year FE absorbed):")
print(f"  ELIGIBLE: {model6.params['ELIGIBLE']:.4f} (SE: {model6.bse['ELIGIBLE']:.4f})")
print(f"  ELIGIBLE_AFTER (DiD): {model6.params['ELIGIBLE_AFTER']:.4f} (SE: {model6.bse['ELIGIBLE_AFTER']:.4f})")
print(f"  FEMALE: {model6.params['FEMALE']:.4f}")
print(f"\n95% CI for DiD: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"R-squared: {model6.rsquared:.4f}")

# =============================================================================
# STEP 6: Additional Robustness Checks
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: ROBUSTNESS CHECKS")
print("=" * 70)

# ----- Robustness 1: Year-by-year treatment effects -----
print("\n--- Robustness Check 1: Event Study / Year-by-Year Effects ---")

# Create year dummies and interactions
years = sorted(df['YEAR'].unique())
for year in years:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression (omit 2011 as reference year)
year_vars = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in years if y != 2011])
event_formula = f'FT ~ ELIGIBLE + C(YEAR) + {year_vars}'

model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df['STATEFIP']}
)

print("\nEvent Study Coefficients (reference year = 2011):")
for year in years:
    if year != 2011:
        coef_name = f'ELIGIBLE_YEAR_{year}'
        coef = model_event.params[coef_name]
        se = model_event.bse[coef_name]
        pval = model_event.pvalues[coef_name]
        sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
        print(f"  {year}: {coef:.4f} (SE: {se:.4f}) {sig}")

# ----- Robustness 2: Placebo test with different pre-treatment cutoff -----
print("\n--- Robustness Check 2: Placebo Test (Pre-Treatment Period Only) ---")
df_pre = df[df['AFTER'] == 0].copy()
df_pre['FAKE_AFTER'] = (df_pre['YEAR'] >= 2010).astype(int)
df_pre['ELIGIBLE_FAKE_AFTER'] = df_pre['ELIGIBLE'] * df_pre['FAKE_AFTER']

model_placebo = smf.wls('FT ~ ELIGIBLE + FAKE_AFTER + ELIGIBLE_FAKE_AFTER',
                        data=df_pre, weights=df_pre['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_pre['STATEFIP']}
)
print(f"\nPlacebo DiD Effect (2010-2011 vs 2008-2009):")
print(f"  Coefficient: {model_placebo.params['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"  Standard Error: {model_placebo.bse['ELIGIBLE_FAKE_AFTER']:.4f}")
print(f"  P-value: {model_placebo.pvalues['ELIGIBLE_FAKE_AFTER']:.4f}")

# ----- Robustness 3: Heterogeneous effects by sex -----
print("\n--- Robustness Check 3: Heterogeneous Effects by Sex ---")

# Males only
df_male = df[df['SEX'] == 1]
model_male = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                     data=df_male, weights=df_male['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_male['STATEFIP']}
)
print(f"\nMales Only (SEX=1):")
print(f"  DiD Estimate: {model_male.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model_male.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_male.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_male.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  P-value: {model_male.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  N: {len(df_male)}")

# Females only
df_female = df[df['SEX'] == 2]
model_female = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                       data=df_female, weights=df_female['PERWT']).fit(
    cov_type='cluster', cov_kwds={'groups': df_female['STATEFIP']}
)
print(f"\nFemales Only (SEX=2):")
print(f"  DiD Estimate: {model_female.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error: {model_female.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% CI: [{model_female.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model_female.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  P-value: {model_female.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  N: {len(df_female)}")

# =============================================================================
# STEP 7: Generate Tables and Figures
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: GENERATING TABLES AND FIGURES")
print("=" * 70)

# ----- Figure 1: Parallel Trends -----
fig, ax = plt.subplots(figsize=(10, 6))
pivot_data = yearly_ft.pivot(index='YEAR', columns='ELIGIBLE', values='FT_Rate')
ax.plot(pivot_data.index, pivot_data[0], 'b-o', label='Control (Ages 31-35)', linewidth=2, markersize=8)
ax.plot(pivot_data.index, pivot_data[1], 'r-s', label='Treatment (Ages 26-30)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2008, 2017))
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_parallel_trends.png")

# ----- Figure 2: Event Study -----
event_years = [y for y in years if y != 2011]
event_coefs = [model_event.params[f'ELIGIBLE_YEAR_{y}'] for y in event_years]
event_ses = [model_event.bse[f'ELIGIBLE_YEAR_{y}'] for y in event_years]
event_ci_low = [c - 1.96*s for c, s in zip(event_coefs, event_ses)]
event_ci_high = [c + 1.96*s for c, s in zip(event_coefs, event_ses)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_years, event_coefs, yerr=[np.array(event_coefs)-np.array(event_ci_low),
                                             np.array(event_ci_high)-np.array(event_coefs)],
            fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8, color='darkblue')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(event_years)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_event_study.png")

# ----- Figure 3: Difference-in-Differences Visual -----
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate pre and post means for each group
pre_control = yearly_ft[(yearly_ft['ELIGIBLE'] == 0) & (yearly_ft['YEAR'] < 2012)]['FT_Rate'].mean()
post_control = yearly_ft[(yearly_ft['ELIGIBLE'] == 0) & (yearly_ft['YEAR'] > 2012)]['FT_Rate'].mean()
pre_treat = yearly_ft[(yearly_ft['ELIGIBLE'] == 1) & (yearly_ft['YEAR'] < 2012)]['FT_Rate'].mean()
post_treat = yearly_ft[(yearly_ft['ELIGIBLE'] == 1) & (yearly_ft['YEAR'] > 2012)]['FT_Rate'].mean()

# Plot the DiD diagram
ax.plot([0, 1], [pre_control, post_control], 'b-o', label='Control (Observed)', linewidth=2, markersize=10)
ax.plot([0, 1], [pre_treat, post_treat], 'r-s', label='Treatment (Observed)', linewidth=2, markersize=10)
# Counterfactual
counterfactual_post = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual_post], 'r--', label='Treatment (Counterfactual)', linewidth=2, alpha=0.5)
# DiD effect arrow
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual_post)/2, f'DiD = {post_treat - counterfactual_post:.3f}',
        fontsize=12, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.2, 1.4)
plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_did_visual.png")

# =============================================================================
# STEP 8: Summary of Results
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF MAIN RESULTS")
print("=" * 70)

print("\n*** PREFERRED SPECIFICATION: Model 4 (Weighted DiD, State-Clustered SE) ***")
print(f"\nEffect of DACA Eligibility on Full-Time Employment:")
print(f"  Point Estimate: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"  Standard Error (clustered): {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"  95% Confidence Interval: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"  t-statistic: {model4.tvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"  Sample Size: {len(df)}")
print(f"  R-squared: {model4.rsquared:.4f}")

print("\n" + "-" * 50)
print("Interpretation:")
did_effect = model4.params['ELIGIBLE_AFTER']
if did_effect > 0:
    print(f"  DACA eligibility is associated with a {did_effect*100:.2f} percentage point")
    print(f"  increase in the probability of full-time employment.")
else:
    print(f"  DACA eligibility is associated with a {abs(did_effect)*100:.2f} percentage point")
    print(f"  decrease in the probability of full-time employment.")

if model4.pvalues['ELIGIBLE_AFTER'] < 0.05:
    print("  This effect is statistically significant at the 5% level.")
elif model4.pvalues['ELIGIBLE_AFTER'] < 0.10:
    print("  This effect is statistically significant at the 10% level.")
else:
    print("  This effect is NOT statistically significant at conventional levels.")

# =============================================================================
# STEP 9: Export Results for Report
# =============================================================================
print("\n" + "=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

# Create results summary for LaTeX
results_dict = {
    'Model': ['Model 1: Basic OLS', 'Model 2: Weighted', 'Model 3: Robust SE',
              'Model 4: Clustered SE', 'Model 5: With Covariates', 'Model 6: State/Year FE'],
    'DiD_Estimate': [
        model1.params['ELIGIBLE_AFTER'],
        model2.params['ELIGIBLE_AFTER'],
        model3.params['ELIGIBLE_AFTER'],
        model4.params['ELIGIBLE_AFTER'],
        model5.params['ELIGIBLE_AFTER'],
        model6.params['ELIGIBLE_AFTER']
    ],
    'SE': [
        model1.bse['ELIGIBLE_AFTER'],
        model2.bse['ELIGIBLE_AFTER'],
        model3.bse['ELIGIBLE_AFTER'],
        model4.bse['ELIGIBLE_AFTER'],
        model5.bse['ELIGIBLE_AFTER'],
        model6.bse['ELIGIBLE_AFTER']
    ],
    'P_value': [
        model1.pvalues['ELIGIBLE_AFTER'],
        model2.pvalues['ELIGIBLE_AFTER'],
        model3.pvalues['ELIGIBLE_AFTER'],
        model4.pvalues['ELIGIBLE_AFTER'],
        model5.pvalues['ELIGIBLE_AFTER'],
        model6.pvalues['ELIGIBLE_AFTER']
    ],
    'R_squared': [
        model1.rsquared,
        model2.rsquared,
        model3.rsquared,
        model4.rsquared,
        model5.rsquared,
        model6.rsquared
    ]
}
results_df = pd.DataFrame(results_dict)
results_df.to_csv('results_summary.csv', index=False)
print("Results summary saved: results_summary.csv")

# Export descriptive statistics
desc_stats = df.groupby('ELIGIBLE').agg({
    'FT': ['mean', 'std', 'count'],
    'AGE': ['mean', 'std'],
    'FEMALE': 'mean',
    'FAMSIZE': 'mean',
    'NCHILD': 'mean'
}).round(4)
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved: descriptive_stats.csv")

# Export group means for 2x2 table
group_means = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: pd.Series({
        'FT_Rate': np.average(x['FT'], weights=x['PERWT']),
        'N': len(x),
        'Weighted_N': x['PERWT'].sum()
    })
).reset_index()
group_means.to_csv('group_means.csv', index=False)
print("Group means saved: group_means.csv")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Final output for easy extraction
print("\n\n### KEY RESULTS FOR REPORT ###")
print(f"Preferred DiD Estimate: {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"Clustered Standard Error: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"P-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"Sample Size: {len(df)}")

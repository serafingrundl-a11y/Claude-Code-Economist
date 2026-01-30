"""
DACA Replication Study - Analysis Script
Replication 94

Research Question: Effect of DACA eligibility on full-time employment
among Hispanic-Mexican, Mexican-born individuals in the US.

Design: Difference-in-differences
- Treatment group: Ages 26-30 at DACA implementation (June 15, 2012)
- Control group: Ages 31-35 at DACA implementation
- Pre-period: 2006-2011
- Post-period: 2013-2016 (2012 excluded due to mid-year implementation)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('output', exist_ok=True)

print("=" * 60)
print("DACA Replication Study - Analysis")
print("=" * 60)

# Load data
print("\nLoading data...")
data_path = "data/data.csv"

# Read in chunks to manage memory
chunks = []
chunksize = 500000
for chunk in pd.read_csv(data_path, chunksize=chunksize, low_memory=False):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
print(f"Total rows loaded: {len(df):,}")

# ============================================================
# STEP 1: Filter to Target Population
# ============================================================
print("\n" + "=" * 60)
print("Step 1: Filtering to Target Population")
print("=" * 60)

# 1. Hispanic-Mexican ethnicity (HISPAN=1)
df_mex = df[df['HISPAN'] == 1].copy()
print(f"After filtering Hispanic-Mexican (HISPAN=1): {len(df_mex):,}")

# 2. Born in Mexico (BPL=200)
df_mex = df_mex[df_mex['BPL'] == 200].copy()
print(f"After filtering born in Mexico (BPL=200): {len(df_mex):,}")

# 3. Non-citizen (CITIZEN=3) - proxy for undocumented
# Per instructions: "Assume that anyone who is not a citizen and who has not
# received immigration papers is undocumented for DACA purposes"
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()
print(f"After filtering non-citizens (CITIZEN=3): {len(df_mex):,}")

# 4. Calculate age at DACA implementation (June 15, 2012)
# For simplicity, we use mid-year 2012 (June) as reference point
# A person born in 1982 would be 30 in 2012, born in 1986 would be 26
# A person born in 1977 would be 35 in 2012, born in 1981 would be 31
df_mex['age_at_daca'] = 2012 - df_mex['BIRTHYR']
print(f"\nAge at DACA distribution in filtered sample:")
print(df_mex['age_at_daca'].describe())

# 5. Arrived before age 16 (requirement for DACA eligibility)
# This is a general eligibility requirement
df_mex['age_at_immigration'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
# YRIMMIG=0 means N/A, so we filter those out
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
print(f"After filtering valid immigration year: {len(df_mex):,}")

# Filter to those who arrived before age 16
df_mex = df_mex[df_mex['age_at_immigration'] < 16].copy()
print(f"After filtering arrived before age 16: {len(df_mex):,}")

# 6. Must have been present since June 2007 (5 years continuous presence)
# This means immigration year must be 2007 or earlier
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()
print(f"After filtering immigrated by 2007: {len(df_mex):,}")

# 7. Define treatment and control groups based on age at DACA
# Treatment: age 26-30 at DACA (born 1982-1986)
# Control: age 31-35 at DACA (born 1977-1981)
df_mex['treatment_group'] = ((df_mex['age_at_daca'] >= 26) &
                              (df_mex['age_at_daca'] <= 30)).astype(int)
df_mex['control_group'] = ((df_mex['age_at_daca'] >= 31) &
                            (df_mex['age_at_daca'] <= 35)).astype(int)

# Keep only those in treatment or control group
df_analysis = df_mex[(df_mex['treatment_group'] == 1) |
                      (df_mex['control_group'] == 1)].copy()
print(f"After filtering to age groups 26-30 and 31-35: {len(df_analysis):,}")

# Create treated indicator (1 if in treatment group)
df_analysis['treated'] = df_analysis['treatment_group']

# 8. Exclude 2012 (DACA implemented mid-year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"After excluding 2012: {len(df_analysis):,}")

# 9. Define pre/post periods
# Pre: 2006-2011, Post: 2013-2016
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)

# ============================================================
# STEP 2: Create Outcome Variable
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Creating Outcome Variable")
print("=" * 60)

# Full-time employment: usually working 35+ hours per week
# UHRSWORK: Usual hours worked per week (0 = N/A or not employed)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

print(f"\nFull-time employment rate overall: {df_analysis['fulltime'].mean():.4f}")
print(f"Full-time employment by group:")
print(f"  Treatment group: {df_analysis[df_analysis['treated']==1]['fulltime'].mean():.4f}")
print(f"  Control group: {df_analysis[df_analysis['treated']==0]['fulltime'].mean():.4f}")

# ============================================================
# STEP 3: Sample Statistics
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Sample Statistics")
print("=" * 60)

print(f"\nSample size by year and treatment status:")
crosstab = pd.crosstab(df_analysis['YEAR'], df_analysis['treated'],
                       margins=True, margins_name='Total')
crosstab.columns = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(crosstab)

print(f"\nSample size by pre/post and treatment status:")
crosstab2 = pd.crosstab(df_analysis['post'], df_analysis['treated'],
                        margins=True, margins_name='Total')
crosstab2.index = ['Pre (2006-2011)', 'Post (2013-2016)', 'Total']
crosstab2.columns = ['Control (31-35)', 'Treatment (26-30)', 'Total']
print(crosstab2)

print(f"\nFull-time employment by period and group:")
ft_by_group = df_analysis.groupby(['post', 'treated'])['fulltime'].agg(['mean', 'count'])
print(ft_by_group)

# Calculate simple difference-in-differences
pre_treat = df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==1)]['fulltime'].mean()
pre_ctrl = df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==0)]['fulltime'].mean()
post_treat = df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==1)]['fulltime'].mean()
post_ctrl = df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==0)]['fulltime'].mean()

simple_did = (post_treat - pre_treat) - (post_ctrl - pre_ctrl)

print(f"\nSimple DiD Calculation:")
print(f"  Pre-period treatment mean:  {pre_treat:.4f}")
print(f"  Pre-period control mean:    {pre_ctrl:.4f}")
print(f"  Post-period treatment mean: {post_treat:.4f}")
print(f"  Post-period control mean:   {post_ctrl:.4f}")
print(f"  Change in treatment:        {post_treat - pre_treat:.4f}")
print(f"  Change in control:          {post_ctrl - pre_ctrl:.4f}")
print(f"  Difference-in-Differences:  {simple_did:.4f}")

# ============================================================
# STEP 4: Difference-in-Differences Regression
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Difference-in-Differences Regression")
print("=" * 60)

# Create interaction term
df_analysis['treated_post'] = df_analysis['treated'] * df_analysis['post']

# Create control variables
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)
df_analysis['married'] = (df_analysis['MARST'] == 1).astype(int)
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)  # High school or more

# Model 1: Basic DiD without controls
print("\n--- Model 1: Basic DiD (No Controls) ---")
model1 = smf.ols('fulltime ~ treated + post + treated_post',
                  data=df_analysis).fit(cov_type='HC1')
print(model1.summary())

# Model 2: DiD with demographic controls
print("\n--- Model 2: DiD with Demographic Controls ---")
model2 = smf.ols('fulltime ~ treated + post + treated_post + female + married + educ_hs',
                  data=df_analysis).fit(cov_type='HC1')
print(model2.summary())

# Model 3: DiD with year fixed effects
print("\n--- Model 3: DiD with Year Fixed Effects ---")
# Create year dummies
for year in df_analysis['YEAR'].unique():
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)

# Use 2006 as reference year
year_cols = [f'year_{y}' for y in sorted(df_analysis['YEAR'].unique()) if y != 2006]
formula3 = 'fulltime ~ treated + treated_post + female + married + educ_hs + ' + ' + '.join(year_cols)
model3 = smf.ols(formula3, data=df_analysis).fit(cov_type='HC1')
print(model3.summary())

# Model 4: DiD with state fixed effects
print("\n--- Model 4: DiD with Year and State Fixed Effects ---")
# Create state dummies
states = sorted(df_analysis['STATEFIP'].unique())
ref_state = states[0]
for state in states:
    df_analysis[f'state_{state}'] = (df_analysis['STATEFIP'] == state).astype(int)

state_cols = [f'state_{s}' for s in states if s != ref_state]
formula4 = 'fulltime ~ treated + treated_post + female + married + educ_hs + ' + ' + '.join(year_cols) + ' + ' + ' + '.join(state_cols)
model4 = smf.ols(formula4, data=df_analysis).fit(cov_type='HC1')

# Print just the key coefficients
print("\nKey coefficients from Model 4:")
print(f"  treated_post (DiD estimate): {model4.params['treated_post']:.4f}")
print(f"  Standard error:              {model4.bse['treated_post']:.4f}")
print(f"  t-statistic:                 {model4.tvalues['treated_post']:.4f}")
print(f"  p-value:                     {model4.pvalues['treated_post']:.4f}")
print(f"  95% CI:                      [{model4.conf_int().loc['treated_post', 0]:.4f}, {model4.conf_int().loc['treated_post', 1]:.4f}]")

# ============================================================
# STEP 5: Weighted Analysis using PERWT
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Weighted Regression Analysis")
print("=" * 60)

# Model 5: Weighted DiD with controls
print("\n--- Model 5: Weighted DiD with Year and State Fixed Effects ---")

# Prepare data for weighted regression - ensure all numeric
X_vars = ['treated', 'treated_post', 'female', 'married', 'educ_hs'] + year_cols + state_cols
X = df_analysis[X_vars].astype(float).copy()
X = sm.add_constant(X)
y = df_analysis['fulltime'].astype(float)
weights = df_analysis['PERWT'].astype(float)

# Weighted least squares
model5 = sm.WLS(y, X, weights=weights).fit(cov_type='HC1')

print("\nKey coefficients from Weighted Model 5:")
print(f"  treated_post (DiD estimate): {model5.params['treated_post']:.4f}")
print(f"  Standard error:              {model5.bse['treated_post']:.4f}")
print(f"  t-statistic:                 {model5.tvalues['treated_post']:.4f}")
print(f"  p-value:                     {model5.pvalues['treated_post']:.4f}")
print(f"  95% CI:                      [{model5.conf_int().loc['treated_post', 0]:.4f}, {model5.conf_int().loc['treated_post', 1]:.4f}]")

# ============================================================
# STEP 6: Robustness Checks - Pre-trends
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Robustness Checks - Pre-trends Analysis")
print("=" * 60)

# Create year-specific treatment effects
years_for_trends = [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years_for_trends:
    df_analysis[f'treated_year_{year}'] = (df_analysis['treated'] * (df_analysis['YEAR'] == year)).astype(int)

trend_vars = [f'treated_year_{year}' for year in years_for_trends]
formula_trend = 'fulltime ~ treated + ' + ' + '.join(trend_vars) + ' + female + married + educ_hs + ' + ' + '.join(year_cols)
model_trend = smf.ols(formula_trend, data=df_analysis).fit(cov_type='HC1')

print("\nYear-specific treatment effects (relative to 2006):")
trend_data = []
for year in years_for_trends:
    var = f'treated_year_{year}'
    coef = model_trend.params[var]
    se = model_trend.bse[var]
    pval = model_trend.pvalues[var]
    print(f"  {year}: {coef:.4f} (SE: {se:.4f}, p={pval:.4f})")
    trend_data.append({'year': year, 'coef': coef, 'se': se, 'pval': pval})

# ============================================================
# STEP 7: Create Visualizations
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Creating Visualizations")
print("=" * 60)

# Figure 1: Trends in full-time employment by group
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate means by year and group
yearly_means = df_analysis.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']

yearly_means.plot(ax=ax1, marker='o', linewidth=2)
ax1.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-time Employment Rate', fontsize=12)
ax1.set_title('Full-time Employment Trends by Age Group\n(DACA-Eligible Hispanic-Mexican Non-Citizens)', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.5, 0.75)
plt.tight_layout()
plt.savefig('output/figure1_trends.png', dpi=150)
plt.close()
print("  Saved figure1_trends.png")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

years = [y['year'] for y in trend_data]
coefs = [y['coef'] for y in trend_data]
ses = [y['se'] for y in trend_data]

# Add 2006 as reference year with 0 effect
years = [2006] + years
coefs = [0] + coefs
ses = [0] + ses

ax2.errorbar(years, coefs, yerr=[1.96*s for s in ses], fmt='o-', capsize=5, linewidth=2, markersize=8)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax2.axvline(x=2012, color='red', linestyle='--', label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2006)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects\n(95% Confidence Intervals)', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/figure2_eventstudy.png', dpi=150)
plt.close()
print("  Saved figure2_eventstudy.png")

# Figure 3: DiD visualization
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Simple 2x2 DiD plot
x = [0, 1]
treat_y = [pre_treat, post_treat]
ctrl_y = [pre_ctrl, post_ctrl]

ax3.plot(x, treat_y, 'b-o', label='Treatment (26-30)', linewidth=2, markersize=10)
ax3.plot(x, ctrl_y, 'r-s', label='Control (31-35)', linewidth=2, markersize=10)

# Counterfactual
counterfactual_post = pre_treat + (post_ctrl - pre_ctrl)
ax3.plot([0, 1], [pre_treat, counterfactual_post], 'b--', alpha=0.5, label='Treatment Counterfactual')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.55, 0.70)
plt.tight_layout()
plt.savefig('output/figure3_did.png', dpi=150)
plt.close()
print("  Saved figure3_did.png")

# ============================================================
# STEP 8: Summary Statistics Table
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Summary Statistics")
print("=" * 60)

print("\nSummary Statistics by Group and Period:")
for period_label, period_val in [('Pre (2006-2011)', 0), ('Post (2013-2016)', 1)]:
    print(f"\n{period_label}:")
    for group_label, group_val in [('Control (31-35)', 0), ('Treatment (26-30)', 1)]:
        subset = df_analysis[(df_analysis['post'] == period_val) & (df_analysis['treated'] == group_val)]
        print(f"  {group_label}:")
        print(f"    N = {len(subset):,}")
        print(f"    Full-time employment: {subset['fulltime'].mean():.4f}")
        print(f"    Female: {subset['female'].mean():.4f}")
        print(f"    Married: {subset['married'].mean():.4f}")
        print(f"    High school+: {subset['educ_hs'].mean():.4f}")
        print(f"    Mean age: {subset['AGE'].mean():.2f}")

# ============================================================
# STEP 9: Save Results
# ============================================================
print("\n" + "=" * 60)
print("Step 9: Saving Results")
print("=" * 60)

# Save key results to file
results = {
    'Model': ['Basic DiD', 'With Controls', 'Year FE', 'Year+State FE', 'Weighted Year+State FE'],
    'DiD_Estimate': [model1.params['treated_post'], model2.params['treated_post'],
                     model3.params['treated_post'], model4.params['treated_post'],
                     model5.params['treated_post']],
    'Std_Error': [model1.bse['treated_post'], model2.bse['treated_post'],
                  model3.bse['treated_post'], model4.bse['treated_post'],
                  model5.bse['treated_post']],
    'p_value': [model1.pvalues['treated_post'], model2.pvalues['treated_post'],
                model3.pvalues['treated_post'], model4.pvalues['treated_post'],
                model5.pvalues['treated_post']],
    'CI_lower': [model1.conf_int().loc['treated_post', 0], model2.conf_int().loc['treated_post', 0],
                 model3.conf_int().loc['treated_post', 0], model4.conf_int().loc['treated_post', 0],
                 model5.conf_int().loc['treated_post', 0]],
    'CI_upper': [model1.conf_int().loc['treated_post', 1], model2.conf_int().loc['treated_post', 1],
                 model3.conf_int().loc['treated_post', 1], model4.conf_int().loc['treated_post', 1],
                 model5.conf_int().loc['treated_post', 1]],
    'N': [int(model1.nobs), int(model2.nobs), int(model3.nobs), int(model4.nobs), int(model5.nobs)]
}
results_df = pd.DataFrame(results)
results_df.to_csv('output/regression_results.csv', index=False)
print("  Saved regression_results.csv")

# Save sample statistics
sample_stats = {
    'Group': ['Treatment Pre', 'Treatment Post', 'Control Pre', 'Control Post'],
    'N': [
        len(df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==1)]),
        len(df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==1)]),
        len(df_analysis[(df_analysis['post']==0) & (df_analysis['treated']==0)]),
        len(df_analysis[(df_analysis['post']==1) & (df_analysis['treated']==0)])
    ],
    'FT_Employment': [pre_treat, post_treat, pre_ctrl, post_ctrl]
}
stats_df = pd.DataFrame(sample_stats)
stats_df.to_csv('output/sample_statistics.csv', index=False)
print("  Saved sample_statistics.csv")

# Save pre-trend results
trend_df = pd.DataFrame(trend_data)
trend_df.to_csv('output/pretrend_analysis.csv', index=False)
print("  Saved pretrend_analysis.csv")

# Save detailed summary statistics
summary_data = []
for period_label, period_val in [('Pre', 0), ('Post', 1)]:
    for group_label, group_val in [('Control', 0), ('Treatment', 1)]:
        subset = df_analysis[(df_analysis['post'] == period_val) & (df_analysis['treated'] == group_val)]
        summary_data.append({
            'Period': period_label,
            'Group': group_label,
            'N': len(subset),
            'FT_Employment': subset['fulltime'].mean(),
            'Female_Pct': subset['female'].mean(),
            'Married_Pct': subset['married'].mean(),
            'HS_Plus_Pct': subset['educ_hs'].mean(),
            'Mean_Age': subset['AGE'].mean(),
            'Mean_Hours': subset['UHRSWORK'].mean()
        })
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('output/detailed_summary_stats.csv', index=False)
print("  Saved detailed_summary_stats.csv")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY - Preferred Estimate")
print("=" * 60)

# Preferred specification: Model 4 (unweighted with year and state FE)
# Reasoning: Survey weights can be controversial in DiD; unweighted is more robust
preferred_model = model4
preferred_name = "Model 4: Year and State Fixed Effects (unweighted)"

print(f"\nPreferred Specification: {preferred_name}")
print(f"Sample Size: {int(preferred_model.nobs):,}")
print(f"DiD Estimate (treated_post): {preferred_model.params['treated_post']:.4f}")
print(f"Standard Error (robust): {preferred_model.bse['treated_post']:.4f}")
print(f"95% Confidence Interval: [{preferred_model.conf_int().loc['treated_post', 0]:.4f}, {preferred_model.conf_int().loc['treated_post', 1]:.4f}]")
print(f"p-value: {preferred_model.pvalues['treated_post']:.6f}")

interpretation = "increase" if preferred_model.params['treated_post'] > 0 else "decrease"
print(f"\nInterpretation: DACA eligibility is associated with a {abs(preferred_model.params['treated_post'])*100:.2f} percentage point")
print(f"{interpretation} in the probability of full-time employment.")

# Save final summary
with open('output/final_summary.txt', 'w') as f:
    f.write("DACA Replication Study - Final Summary\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Research Question: Effect of DACA eligibility on full-time employment\n")
    f.write(f"Design: Difference-in-Differences\n")
    f.write(f"Treatment: Ages 26-30 at DACA (eligible)\n")
    f.write(f"Control: Ages 31-35 at DACA (just too old)\n\n")
    f.write(f"Preferred Specification: {preferred_name}\n")
    f.write(f"Sample Size: {int(preferred_model.nobs):,}\n")
    f.write(f"DiD Estimate: {preferred_model.params['treated_post']:.4f}\n")
    f.write(f"Standard Error: {preferred_model.bse['treated_post']:.4f}\n")
    f.write(f"95% CI: [{preferred_model.conf_int().loc['treated_post', 0]:.4f}, {preferred_model.conf_int().loc['treated_post', 1]:.4f}]\n")
    f.write(f"p-value: {preferred_model.pvalues['treated_post']:.6f}\n\n")
    f.write(f"Simple DiD Calculation:\n")
    f.write(f"  Pre-treatment mean: {pre_treat:.4f}\n")
    f.write(f"  Post-treatment mean: {post_treat:.4f}\n")
    f.write(f"  Pre-control mean: {pre_ctrl:.4f}\n")
    f.write(f"  Post-control mean: {post_ctrl:.4f}\n")
    f.write(f"  DiD: {simple_did:.4f}\n")

print("\n  Saved final_summary.txt")
print("\nAnalysis complete!")

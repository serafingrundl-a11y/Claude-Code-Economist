"""
DACA Replication Analysis
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States (2013-2016)

This script performs:
1. Data preparation and sample construction
2. Difference-in-differences analysis
3. Robustness checks
4. Generates tables and figures for the replication report
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

# ============================================
# 1. DATA LOADING AND PREPARATION
# ============================================

print("=" * 60)
print("DACA REPLICATION ANALYSIS")
print("=" * 60)
print()

# Load filtered sample (Hispanic-Mexican, Mexican-born)
df = pd.read_pickle('data/mexican_hispanic_sample.pkl')
print(f"Loaded sample: {len(df):,} observations")
print(f"Years: {sorted(df['YEAR'].unique())}")
print()

# ============================================
# 2. VARIABLE CONSTRUCTION
# ============================================

# Age at immigration
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']

# DACA eligibility criteria (fixed as of June 15, 2012)
# 1. Arrived before age 16
crit1 = df['age_at_immig'] < 16

# 2. Under 31 as of June 15, 2012 (born after June 15, 1981)
crit2 = (df['BIRTHYR'] >= 1982) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([1, 2])))

# 3. In US since at least 2007
crit3 = df['YRIMMIG'] <= 2007

# 4. Not a citizen
crit4 = df['CITIZEN'] == 3

# Full DACA eligibility
df['daca_eligible'] = (crit1 & crit2 & crit3 & crit4).astype(int)

# Post-DACA period (2013-2016)
df['post_daca'] = (df['YEAR'] >= 2013).astype(int)

# Interaction term for DiD
df['daca_x_post'] = df['daca_eligible'] * df['post_daca']

# Outcome: Full-time employment (35+ hours per week)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Control variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['age_sq'] = df['AGE'] ** 2

# Education categories
df['educ_less_hs'] = (df['EDUC'] < 6).astype(int)
df['educ_hs'] = (df['EDUC'] == 6).astype(int)
df['educ_some_college'] = (df['EDUC'].isin([7, 8, 9])).astype(int)
df['educ_college_plus'] = (df['EDUC'] >= 10).astype(int)

# ============================================
# 3. SAMPLE RESTRICTIONS
# ============================================

# Restrict to working-age population (18-64)
df_analysis = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
print(f"Working-age sample (18-64): {len(df_analysis):,}")

# Exclude 2012 (DACA announced mid-year)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
print(f"Sample excluding 2012: {len(df_analysis):,}")
print()

# ============================================
# 4. DESCRIPTIVE STATISTICS
# ============================================

print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)
print()

# Sample sizes by group and period
print("Sample sizes:")
print("-" * 40)
pre = df_analysis[df_analysis['post_daca'] == 0]
post = df_analysis[df_analysis['post_daca'] == 1]

print(f"Pre-DACA (2006-2011):")
print(f"  DACA-eligible: {len(pre[pre['daca_eligible']==1]):,}")
print(f"  Control:       {len(pre[pre['daca_eligible']==0]):,}")
print(f"Post-DACA (2013-2016):")
print(f"  DACA-eligible: {len(post[post['daca_eligible']==1]):,}")
print(f"  Control:       {len(post[post['daca_eligible']==0]):,}")
print()

# Mean characteristics by treatment status
print("Mean characteristics by treatment status:")
print("-" * 60)
vars_to_describe = ['AGE', 'female', 'married', 'educ_less_hs', 'educ_hs',
                    'educ_some_college', 'educ_college_plus', 'fulltime']
var_labels = ['Age', 'Female', 'Married', 'Less than HS', 'High School',
              'Some College', 'College+', 'Full-time Employed']

desc_stats = []
for var in vars_to_describe:
    eligible_mean = df_analysis[df_analysis['daca_eligible']==1][var].mean()
    control_mean = df_analysis[df_analysis['daca_eligible']==0][var].mean()
    diff = eligible_mean - control_mean
    desc_stats.append([var, eligible_mean, control_mean, diff])

desc_df = pd.DataFrame(desc_stats, columns=['Variable', 'DACA-Eligible', 'Control', 'Difference'])
desc_df.index = var_labels
print(desc_df[['DACA-Eligible', 'Control', 'Difference']].round(3).to_string())
print()

# Full-time employment rates by group and period
print("Full-time Employment Rates:")
print("-" * 60)
print(f"                        Pre-DACA    Post-DACA    Change")
print(f"DACA-eligible:          {pre[pre['daca_eligible']==1]['fulltime'].mean():.4f}      {post[post['daca_eligible']==1]['fulltime'].mean():.4f}       {post[post['daca_eligible']==1]['fulltime'].mean() - pre[pre['daca_eligible']==1]['fulltime'].mean():.4f}")
print(f"Control:                {pre[pre['daca_eligible']==0]['fulltime'].mean():.4f}      {post[post['daca_eligible']==0]['fulltime'].mean():.4f}       {post[post['daca_eligible']==0]['fulltime'].mean() - pre[pre['daca_eligible']==0]['fulltime'].mean():.4f}")
print()

# Simple DiD
pre_eligible = pre[pre['daca_eligible']==1]['fulltime'].mean()
pre_control = pre[pre['daca_eligible']==0]['fulltime'].mean()
post_eligible = post[post['daca_eligible']==1]['fulltime'].mean()
post_control = post[post['daca_eligible']==0]['fulltime'].mean()
simple_did = (post_eligible - pre_eligible) - (post_control - pre_control)
print(f"Simple DiD estimate: {simple_did:.4f}")
print()

# ============================================
# 5. REGRESSION ANALYSIS
# ============================================

print("=" * 60)
print("DIFFERENCE-IN-DIFFERENCES REGRESSION ANALYSIS")
print("=" * 60)
print()

# Model 1: Basic DiD (no controls)
print("Model 1: Basic DiD (no controls)")
print("-" * 60)
model1 = smf.ols('fulltime ~ daca_eligible + post_daca + daca_x_post', data=df_analysis).fit()
print(f"DiD coefficient (daca_x_post): {model1.params['daca_x_post']:.4f}")
print(f"Standard error: {model1.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model1.conf_int().loc['daca_x_post', 0]:.4f}, {model1.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model1.pvalues['daca_x_post']:.4f}")
print(f"N: {int(model1.nobs):,}")
print(f"R-squared: {model1.rsquared:.4f}")
print()

# Model 2: DiD with demographic controls
print("Model 2: DiD with demographic controls")
print("-" * 60)
formula2 = 'fulltime ~ daca_eligible + post_daca + daca_x_post + AGE + age_sq + female + married'
model2 = smf.ols(formula2, data=df_analysis).fit()
print(f"DiD coefficient (daca_x_post): {model2.params['daca_x_post']:.4f}")
print(f"Standard error: {model2.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model2.conf_int().loc['daca_x_post', 0]:.4f}, {model2.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model2.pvalues['daca_x_post']:.4f}")
print(f"N: {int(model2.nobs):,}")
print(f"R-squared: {model2.rsquared:.4f}")
print()

# Model 3: DiD with demographic and education controls
print("Model 3: DiD with demographic and education controls")
print("-" * 60)
formula3 = 'fulltime ~ daca_eligible + post_daca + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus'
model3 = smf.ols(formula3, data=df_analysis).fit()
print(f"DiD coefficient (daca_x_post): {model3.params['daca_x_post']:.4f}")
print(f"Standard error: {model3.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model3.conf_int().loc['daca_x_post', 0]:.4f}, {model3.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model3.pvalues['daca_x_post']:.4f}")
print(f"N: {int(model3.nobs):,}")
print(f"R-squared: {model3.rsquared:.4f}")
print()

# Model 4: DiD with all controls including state and year fixed effects
print("Model 4: DiD with all controls + state and year FE")
print("-" * 60)
df_analysis['state_fe'] = df_analysis['STATEFIP'].astype(str)
df_analysis['year_fe'] = df_analysis['YEAR'].astype(str)
formula4 = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(state_fe) + C(year_fe)'
model4 = smf.ols(formula4, data=df_analysis).fit()
print(f"DiD coefficient (daca_x_post): {model4.params['daca_x_post']:.4f}")
print(f"Standard error: {model4.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model4.conf_int().loc['daca_x_post', 0]:.4f}, {model4.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model4.pvalues['daca_x_post']:.4f}")
print(f"N: {int(model4.nobs):,}")
print(f"R-squared: {model4.rsquared:.4f}")
print()

# Model 5: Clustered standard errors at state level (PREFERRED SPECIFICATION)
print("Model 5: Preferred specification with clustered SEs at state level")
print("-" * 60)
model5 = smf.ols(formula4, data=df_analysis).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})
print(f"DiD coefficient (daca_x_post): {model5.params['daca_x_post']:.4f}")
print(f"Clustered standard error: {model5.bse['daca_x_post']:.4f}")
print(f"95% CI: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"p-value: {model5.pvalues['daca_x_post']:.4f}")
print(f"N: {int(model5.nobs):,}")
print(f"R-squared: {model5.rsquared:.4f}")
print()

# ============================================
# 6. WEIGHTED ANALYSIS
# ============================================

print("=" * 60)
print("WEIGHTED ANALYSIS (using PERWT)")
print("=" * 60)
print()

# Model 6: Weighted DiD with clustered SEs
print("Model 6: Weighted DiD with full controls and clustered SEs")
print("-" * 60)
import statsmodels.api as sm

# Create design matrices
formula_weighted = 'fulltime ~ daca_eligible + daca_x_post + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(state_fe) + C(year_fe)'
y, X = smf.ols(formula_weighted, data=df_analysis).fit().model.endog, smf.ols(formula_weighted, data=df_analysis).fit().model.exog
model6 = sm.WLS(df_analysis['fulltime'], X, weights=df_analysis['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df_analysis['STATEFIP']})

# Find daca_x_post index
feature_names = smf.ols(formula_weighted, data=df_analysis).fit().model.exog_names
daca_idx = feature_names.index('daca_x_post')

print(f"DiD coefficient (daca_x_post): {model6.params[daca_idx]:.4f}")
print(f"Clustered standard error: {model6.bse[daca_idx]:.4f}")
ci_array = model6.conf_int()
print(f"95% CI: [{ci_array.iloc[daca_idx, 0]:.4f}, {ci_array.iloc[daca_idx, 1]:.4f}]")
print(f"p-value: {model6.pvalues[daca_idx]:.4f}")
print(f"N: {int(model6.nobs):,}")
print()

# ============================================
# 7. ROBUSTNESS CHECKS
# ============================================

print("=" * 60)
print("ROBUSTNESS CHECKS")
print("=" * 60)
print()

# Robustness 1: Alternative age range (16-40)
print("Robustness 1: Alternative age range (16-40)")
print("-" * 60)
df_rob1 = df[(df['AGE'] >= 16) & (df['AGE'] <= 40) & (df['YEAR'] != 2012)].copy()
df_rob1['state_fe'] = df_rob1['STATEFIP'].astype(str)
df_rob1['year_fe'] = df_rob1['YEAR'].astype(str)
model_rob1 = smf.ols(formula4, data=df_rob1).fit(cov_type='cluster', cov_kwds={'groups': df_rob1['STATEFIP']})
print(f"DiD coefficient: {model_rob1.params['daca_x_post']:.4f}")
print(f"Clustered SE: {model_rob1.bse['daca_x_post']:.4f}")
print(f"N: {int(model_rob1.nobs):,}")
print()

# Robustness 2: Males only
print("Robustness 2: Males only")
print("-" * 60)
df_rob2 = df_analysis[df_analysis['female'] == 0].copy()
model_rob2 = smf.ols(formula4, data=df_rob2).fit(cov_type='cluster', cov_kwds={'groups': df_rob2['STATEFIP']})
print(f"DiD coefficient: {model_rob2.params['daca_x_post']:.4f}")
print(f"Clustered SE: {model_rob2.bse['daca_x_post']:.4f}")
print(f"N: {int(model_rob2.nobs):,}")
print()

# Robustness 3: Females only
print("Robustness 3: Females only")
print("-" * 60)
df_rob3 = df_analysis[df_analysis['female'] == 1].copy()
model_rob3 = smf.ols(formula4, data=df_rob3).fit(cov_type='cluster', cov_kwds={'groups': df_rob3['STATEFIP']})
print(f"DiD coefficient: {model_rob3.params['daca_x_post']:.4f}")
print(f"Clustered SE: {model_rob3.bse['daca_x_post']:.4f}")
print(f"N: {int(model_rob3.nobs):,}")
print()

# Robustness 4: Include 2012 in post period
print("Robustness 4: Include 2012 in post period")
print("-" * 60)
df_rob4 = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
df_rob4['post_daca'] = (df_rob4['YEAR'] >= 2012).astype(int)
df_rob4['daca_x_post'] = df_rob4['daca_eligible'] * df_rob4['post_daca']
df_rob4['state_fe'] = df_rob4['STATEFIP'].astype(str)
df_rob4['year_fe'] = df_rob4['YEAR'].astype(str)
model_rob4 = smf.ols(formula4, data=df_rob4).fit(cov_type='cluster', cov_kwds={'groups': df_rob4['STATEFIP']})
print(f"DiD coefficient: {model_rob4.params['daca_x_post']:.4f}")
print(f"Clustered SE: {model_rob4.bse['daca_x_post']:.4f}")
print(f"N: {int(model_rob4.nobs):,}")
print()

# Robustness 5: Placebo test - 2009 as fake treatment year
print("Robustness 5: Placebo test - 2009 as fake treatment year (using 2006-2011 data only)")
print("-" * 60)
df_rob5 = df[(df['AGE'] >= 18) & (df['AGE'] <= 64) & (df['YEAR'] <= 2011)].copy()
df_rob5['post_placebo'] = (df_rob5['YEAR'] >= 2009).astype(int)
df_rob5['daca_x_placebo'] = df_rob5['daca_eligible'] * df_rob5['post_placebo']
df_rob5['state_fe'] = df_rob5['STATEFIP'].astype(str)
df_rob5['year_fe'] = df_rob5['YEAR'].astype(str)
formula_placebo = 'fulltime ~ daca_eligible + daca_x_placebo + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(state_fe) + C(year_fe)'
model_rob5 = smf.ols(formula_placebo, data=df_rob5).fit(cov_type='cluster', cov_kwds={'groups': df_rob5['STATEFIP']})
print(f"Placebo DiD coefficient: {model_rob5.params['daca_x_placebo']:.4f}")
print(f"Clustered SE: {model_rob5.bse['daca_x_placebo']:.4f}")
print(f"p-value: {model_rob5.pvalues['daca_x_placebo']:.4f}")
print(f"N: {int(model_rob5.nobs):,}")
print()

# ============================================
# 8. EVENT STUDY / DYNAMIC EFFECTS
# ============================================

print("=" * 60)
print("EVENT STUDY ANALYSIS")
print("=" * 60)
print()

# Create year interactions (2011 as reference)
df_event = df_analysis.copy()
years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]  # 2011 is reference
for yr in years:
    df_event[f'daca_x_{yr}'] = (df_event['daca_eligible'] * (df_event['YEAR'] == yr)).astype(int)

# Event study regression
year_interactions = ' + '.join([f'daca_x_{yr}' for yr in years])
formula_event = f'fulltime ~ daca_eligible + {year_interactions} + AGE + age_sq + female + married + educ_hs + educ_some_college + educ_college_plus + C(state_fe) + C(year_fe)'
model_event = smf.ols(formula_event, data=df_event).fit(cov_type='cluster', cov_kwds={'groups': df_event['STATEFIP']})

print("Event Study Coefficients (relative to 2011):")
print("-" * 60)
event_results = []
for yr in years:
    coef = model_event.params[f'daca_x_{yr}']
    se = model_event.bse[f'daca_x_{yr}']
    ci_low = model_event.conf_int().loc[f'daca_x_{yr}', 0]
    ci_high = model_event.conf_int().loc[f'daca_x_{yr}', 1]
    print(f"  {yr}: {coef:.4f} (SE: {se:.4f}, 95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    event_results.append([yr, coef, se, ci_low, ci_high])

# Add 2011 as reference
event_results.append([2011, 0, 0, 0, 0])
event_df = pd.DataFrame(event_results, columns=['Year', 'Coefficient', 'SE', 'CI_low', 'CI_high'])
event_df = event_df.sort_values('Year')
print()

# ============================================
# 9. CREATE FIGURE
# ============================================

print("Creating event study figure...")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years_plot = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_low = event_df['CI_low'].values
ci_high = event_df['CI_high'].values

ax.errorbar(years_plot, coefs, yerr=[coefs - ci_low, ci_high - coefs],
            fmt='o-', capsize=4, color='navy', markersize=8)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: event_study.png")
print()

# ============================================
# 10. CREATE TREND FIGURE
# ============================================

print("Creating trends figure...")
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate annual means
trend_data = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()

ax.plot(trend_data.index, trend_data[1], 'o-', color='blue', linewidth=2, markersize=8, label='DACA-Eligible')
ax.plot(trend_data.index, trend_data[0], 's-', color='green', linewidth=2, markersize=8, label='Control')
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.4, 0.7])

plt.tight_layout()
plt.savefig('trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: trends.png")
print()

# ============================================
# 11. SAVE RESULTS FOR REPORT
# ============================================

print("=" * 60)
print("SUMMARY OF MAIN RESULTS")
print("=" * 60)
print()
print(f"PREFERRED SPECIFICATION (Model 5):")
print(f"  Effect size: {model5.params['daca_x_post']:.4f}")
print(f"  Standard error (clustered): {model5.bse['daca_x_post']:.4f}")
print(f"  95% CI: [{model5.conf_int().loc['daca_x_post', 0]:.4f}, {model5.conf_int().loc['daca_x_post', 1]:.4f}]")
print(f"  p-value: {model5.pvalues['daca_x_post']:.4f}")
print(f"  Sample size: {int(model5.nobs):,}")
print()

# Save key results to file
results_dict = {
    'model1_coef': model1.params['daca_x_post'],
    'model1_se': model1.bse['daca_x_post'],
    'model1_n': int(model1.nobs),
    'model2_coef': model2.params['daca_x_post'],
    'model2_se': model2.bse['daca_x_post'],
    'model2_n': int(model2.nobs),
    'model3_coef': model3.params['daca_x_post'],
    'model3_se': model3.bse['daca_x_post'],
    'model3_n': int(model3.nobs),
    'model4_coef': model4.params['daca_x_post'],
    'model4_se': model4.bse['daca_x_post'],
    'model4_n': int(model4.nobs),
    'model5_coef': model5.params['daca_x_post'],
    'model5_se': model5.bse['daca_x_post'],
    'model5_ci_low': model5.conf_int().loc['daca_x_post', 0],
    'model5_ci_high': model5.conf_int().loc['daca_x_post', 1],
    'model5_pval': model5.pvalues['daca_x_post'],
    'model5_n': int(model5.nobs),
    'model5_r2': model5.rsquared,
    'weighted_coef': model6.params[daca_idx],
    'weighted_se': model6.bse[daca_idx],
    'rob1_coef': model_rob1.params['daca_x_post'],
    'rob1_se': model_rob1.bse['daca_x_post'],
    'rob2_coef': model_rob2.params['daca_x_post'],
    'rob2_se': model_rob2.bse['daca_x_post'],
    'rob3_coef': model_rob3.params['daca_x_post'],
    'rob3_se': model_rob3.bse['daca_x_post'],
    'rob4_coef': model_rob4.params['daca_x_post'],
    'rob4_se': model_rob4.bse['daca_x_post'],
    'placebo_coef': model_rob5.params['daca_x_placebo'],
    'placebo_se': model_rob5.bse['daca_x_placebo'],
    'placebo_pval': model_rob5.pvalues['daca_x_placebo'],
    'simple_did': simple_did,
    'pre_eligible_mean': pre_eligible,
    'pre_control_mean': pre_control,
    'post_eligible_mean': post_eligible,
    'post_control_mean': post_control,
    'n_eligible_pre': len(pre[pre['daca_eligible']==1]),
    'n_control_pre': len(pre[pre['daca_eligible']==0]),
    'n_eligible_post': len(post[post['daca_eligible']==1]),
    'n_control_post': len(post[post['daca_eligible']==0]),
}

# Save to pickle for later use
pd.to_pickle(results_dict, 'analysis_results.pkl')
print("Results saved to analysis_results.pkl")

# Save event study results
event_df.to_csv('event_study_results.csv', index=False)
print("Event study results saved to event_study_results.csv")

# Save descriptive stats
desc_df.to_csv('descriptive_stats.csv')
print("Descriptive statistics saved to descriptive_stats.csv")

print()
print("Analysis complete!")

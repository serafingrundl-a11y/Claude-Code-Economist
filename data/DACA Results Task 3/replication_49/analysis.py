"""
DACA Replication Study - Analysis Script
Research Question: Effect of DACA eligibility on full-time employment among
Hispanic-Mexican Mexican-born individuals in the United States.

Method: Difference-in-Differences (DiD)
Treatment Group: ELIGIBLE=1 (ages 26-30 at time of policy in June 2012)
Control Group: ELIGIBLE=0 (ages 31-35 at time of policy in June 2012)
Outcome: FT (full-time employment, working 35+ hours per week)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("DACA REPLICATION STUDY - ANALYSIS")
print("="*70)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')
print(f"   Total observations: {len(df):,}")

# Basic data exploration
print("\n2. Data Exploration")
print("-"*50)

# Check key variables
print("\n   Key variables summary:")
print(f"   - Years in data: {sorted(df['YEAR'].unique())}")
print(f"   - ELIGIBLE distribution: \n{df['ELIGIBLE'].value_counts()}")
print(f"   - AFTER distribution: \n{df['AFTER'].value_counts()}")
print(f"   - FT (Full-time) distribution: \n{df['FT'].value_counts()}")

# Cross-tabulation of treatment and time
print("\n   Treatment x Time Cross-tabulation (counts):")
crosstab = pd.crosstab(df['ELIGIBLE'], df['AFTER'], margins=True)
print(crosstab)

# Sample sizes by year and eligibility
print("\n   Sample sizes by Year and ELIGIBLE:")
yearly_sample = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack(fill_value=0)
print(yearly_sample)

# Descriptive statistics for key variables
print("\n3. Descriptive Statistics")
print("-"*50)

# FT rates by group
print("\n   Full-time employment rates by group:")
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].agg(['mean', 'std', 'count'])
ft_rates.columns = ['Mean', 'Std', 'N']
print(ft_rates)

# Separate by treatment status
treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treated_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

print(f"\n   2x2 DiD Table (FT rates):")
print(f"   {'Group':<15} {'Pre-DACA':<12} {'Post-DACA':<12} {'Diff':<12}")
print(f"   {'-'*51}")
print(f"   {'Treatment':<15} {treated_pre:<12.4f} {treated_post:<12.4f} {treated_post-treated_pre:<12.4f}")
print(f"   {'Control':<15} {control_pre:<12.4f} {control_post:<12.4f} {control_post-control_pre:<12.4f}")
print(f"   {'-'*51}")
print(f"   {'DiD':<15} {'':<12} {'':<12} {(treated_post-treated_pre)-(control_post-control_pre):<12.4f}")

# Simple DiD estimate
simple_did = (treated_post - treated_pre) - (control_post - control_pre)
print(f"\n   Simple DiD Estimate: {simple_did:.4f} ({simple_did*100:.2f} percentage points)")

# Save for later
results_dict = {
    'simple_did': simple_did,
    'treated_pre': treated_pre,
    'treated_post': treated_post,
    'control_pre': control_pre,
    'control_post': control_post
}

# 4. Regression Analysis
print("\n4. Difference-in-Differences Regression Analysis")
print("-"*50)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Model 1: Basic DiD without controls
print("\n   Model 1: Basic DiD (no controls)")
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model1.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Standard Error: {model1.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model1.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model1.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model1.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model1.rsquared:.4f}")
print(f"   N: {int(model1.nobs):,}")

results_dict['model1_coef'] = model1.params['ELIGIBLE_AFTER']
results_dict['model1_se'] = model1.bse['ELIGIBLE_AFTER']
results_dict['model1_ci_low'] = model1.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model1_ci_high'] = model1.conf_int().loc['ELIGIBLE_AFTER', 1]
results_dict['model1_pval'] = model1.pvalues['ELIGIBLE_AFTER']
results_dict['model1_n'] = int(model1.nobs)

# Model 2: DiD with robust standard errors
print("\n   Model 2: Basic DiD with robust (HC1) standard errors")
model2 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model2.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model2.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model2.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model2.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model2.pvalues['ELIGIBLE_AFTER']:.4f}")

results_dict['model2_coef'] = model2.params['ELIGIBLE_AFTER']
results_dict['model2_se'] = model2.bse['ELIGIBLE_AFTER']
results_dict['model2_ci_low'] = model2.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model2_ci_high'] = model2.conf_int().loc['ELIGIBLE_AFTER', 1]

# Model 3: DiD with demographic controls
print("\n   Model 3: DiD with demographic controls (SEX, AGE, MARST)")

# Recode SEX (1=Male, 2=Female in IPUMS) to binary female indicator
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# Create marital status dummies (MARST: 1=married spouse present, 6=never married, etc.)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

# Center age for interpretation
df['AGE_centered'] = df['AGE'] - df['AGE'].mean()

model3 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE_centered + MARRIED', data=df).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model3.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model3.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model3.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model3.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model3.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model3.rsquared:.4f}")

results_dict['model3_coef'] = model3.params['ELIGIBLE_AFTER']
results_dict['model3_se'] = model3.bse['ELIGIBLE_AFTER']
results_dict['model3_ci_low'] = model3.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model3_ci_high'] = model3.conf_int().loc['ELIGIBLE_AFTER', 1]

# Model 4: DiD with demographic and education controls
print("\n   Model 4: DiD with demographic and education controls")

# Create education dummies from EDUC_RECODE
df_model4 = df.copy()
educ_dummies = pd.get_dummies(df_model4['EDUC_RECODE'], prefix='educ', drop_first=True)
df_model4 = pd.concat([df_model4, educ_dummies], axis=1)

# Build formula with education dummies
educ_cols = [c for c in df_model4.columns if c.startswith('educ_')]
formula4 = 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + AGE_centered + MARRIED + ' + ' + '.join([f'Q("{c}")' for c in educ_cols])

model4 = smf.ols(formula4, data=df_model4).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model4.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model4.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model4.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model4.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model4.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model4.rsquared:.4f}")

results_dict['model4_coef'] = model4.params['ELIGIBLE_AFTER']
results_dict['model4_se'] = model4.bse['ELIGIBLE_AFTER']
results_dict['model4_ci_low'] = model4.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model4_ci_high'] = model4.conf_int().loc['ELIGIBLE_AFTER', 1]

# Model 5: Full model with state and year fixed effects
print("\n   Model 5: DiD with controls + state and year fixed effects")

# Create state and year dummies
df_model5 = df_model4.copy()
state_dummies = pd.get_dummies(df_model5['STATEFIP'], prefix='state', drop_first=True)
year_dummies = pd.get_dummies(df_model5['YEAR'], prefix='year', drop_first=True)
df_model5 = pd.concat([df_model5, state_dummies, year_dummies], axis=1)

state_cols = [c for c in df_model5.columns if c.startswith('state_')]
year_cols = [c for c in df_model5.columns if c.startswith('year_')]

formula5 = 'FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + AGE_centered + MARRIED + ' + \
           ' + '.join([f'Q("{c}")' for c in educ_cols]) + ' + ' + \
           ' + '.join([f'Q("{c}")' for c in state_cols]) + ' + ' + \
           ' + '.join([f'Q("{c}")' for c in year_cols])

model5 = smf.ols(formula5, data=df_model5).fit(cov_type='HC1')
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model5.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Robust SE: {model5.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model5.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model5.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model5.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model5.rsquared:.4f}")

results_dict['model5_coef'] = model5.params['ELIGIBLE_AFTER']
results_dict['model5_se'] = model5.bse['ELIGIBLE_AFTER']
results_dict['model5_ci_low'] = model5.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model5_ci_high'] = model5.conf_int().loc['ELIGIBLE_AFTER', 1]
results_dict['model5_pval'] = model5.pvalues['ELIGIBLE_AFTER']

# Model 6: Preferred specification - with clustered standard errors at state level
print("\n   Model 6 (PREFERRED): Full model with state-clustered standard errors")

model6 = smf.ols(formula5, data=df_model5).fit(cov_type='cluster', cov_kwds={'groups': df_model5['STATEFIP']})
print(f"   DiD coefficient (ELIGIBLE_AFTER): {model6.params['ELIGIBLE_AFTER']:.4f}")
print(f"   Clustered SE (state): {model6.bse['ELIGIBLE_AFTER']:.4f}")
print(f"   95% CI: [{model6.conf_int().loc['ELIGIBLE_AFTER', 0]:.4f}, {model6.conf_int().loc['ELIGIBLE_AFTER', 1]:.4f}]")
print(f"   p-value: {model6.pvalues['ELIGIBLE_AFTER']:.4f}")
print(f"   R-squared: {model6.rsquared:.4f}")
print(f"   N: {int(model6.nobs):,}")

results_dict['model6_coef'] = model6.params['ELIGIBLE_AFTER']
results_dict['model6_se'] = model6.bse['ELIGIBLE_AFTER']
results_dict['model6_ci_low'] = model6.conf_int().loc['ELIGIBLE_AFTER', 0]
results_dict['model6_ci_high'] = model6.conf_int().loc['ELIGIBLE_AFTER', 1]
results_dict['model6_pval'] = model6.pvalues['ELIGIBLE_AFTER']
results_dict['model6_n'] = int(model6.nobs)

# 5. Robustness Checks
print("\n5. Robustness Checks")
print("-"*50)

# 5a. Event study / parallel trends check
print("\n   5a. Event Study Analysis (Parallel Trends Check)")

# Create year-specific treatment effects
df_event = df_model4.copy()
df_event['ELIGIBLE_2008'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2008).astype(int)
df_event['ELIGIBLE_2009'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2009).astype(int)
df_event['ELIGIBLE_2010'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2010).astype(int)
# 2011 is reference year (omitted)
df_event['ELIGIBLE_2013'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2013).astype(int)
df_event['ELIGIBLE_2014'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2014).astype(int)
df_event['ELIGIBLE_2015'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2015).astype(int)
df_event['ELIGIBLE_2016'] = df_event['ELIGIBLE'] * (df_event['YEAR'] == 2016).astype(int)

year_dummies_event = pd.get_dummies(df_event['YEAR'], prefix='year', drop_first=True)
df_event = pd.concat([df_event, year_dummies_event], axis=1)
year_cols_event = [c for c in df_event.columns if c.startswith('year_')]

formula_event = 'FT ~ ELIGIBLE + ' + \
                'ELIGIBLE_2008 + ELIGIBLE_2009 + ELIGIBLE_2010 + ELIGIBLE_2013 + ELIGIBLE_2014 + ELIGIBLE_2015 + ELIGIBLE_2016 + ' + \
                'FEMALE + AGE_centered + MARRIED + ' + \
                ' + '.join([f'Q("{c}")' for c in educ_cols]) + ' + ' + \
                ' + '.join([f'Q("{c}")' for c in year_cols_event])

model_event = smf.ols(formula_event, data=df_event).fit(cov_type='HC1')

event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []
print("\n   Year-specific treatment effects (2011 = reference):")
print(f"   {'Year':<8} {'Coef':<12} {'SE':<12} {'95% CI':<25}")
print(f"   {'-'*57}")

# Add reference year (2011)
print(f"   {'2011':<8} {'0.0000':<12} {'(ref)':<12} {'--':<25}")
event_coefs.append(0)
event_ses.append(0)

for year in event_years:
    var = f'ELIGIBLE_{year}'
    coef = model_event.params[var]
    se = model_event.bse[var]
    ci = model_event.conf_int().loc[var]
    print(f"   {year:<8} {coef:<12.4f} {se:<12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    if year != 2011:
        event_coefs.append(coef)
        event_ses.append(se)

# Reorder for plotting
event_years_plot = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs_plot = [event_coefs[0], event_coefs[1], event_coefs[2], 0, event_coefs[3], event_coefs[4], event_coefs[5], event_coefs[6]]
event_ses_plot = [event_ses[0], event_ses[1], event_ses[2], 0, event_ses[3], event_ses[4], event_ses[5], event_ses[6]]

# 5b. Subgroup analysis by sex
print("\n   5b. Subgroup Analysis by Sex")

# Males
df_male = df_model5[df_model5['FEMALE'] == 0].copy()
model_male = smf.ols(formula5.replace('FEMALE + ', ''), data=df_male).fit(cov_type='HC1')
print(f"   Males: DiD = {model_male.params['ELIGIBLE_AFTER']:.4f} (SE = {model_male.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_male.nobs):,}")

# Females
df_female = df_model5[df_model5['FEMALE'] == 1].copy()
model_female = smf.ols(formula5.replace('FEMALE + ', ''), data=df_female).fit(cov_type='HC1')
print(f"   Females: DiD = {model_female.params['ELIGIBLE_AFTER']:.4f} (SE = {model_female.bse['ELIGIBLE_AFTER']:.4f}), N = {int(model_female.nobs):,}")

results_dict['male_coef'] = model_male.params['ELIGIBLE_AFTER']
results_dict['male_se'] = model_male.bse['ELIGIBLE_AFTER']
results_dict['male_n'] = int(model_male.nobs)
results_dict['female_coef'] = model_female.params['ELIGIBLE_AFTER']
results_dict['female_se'] = model_female.bse['ELIGIBLE_AFTER']
results_dict['female_n'] = int(model_female.nobs)

# 5c. Alternative specification: Probit model
print("\n   5c. Alternative Specification: Probit Model")

from statsmodels.discrete.discrete_model import Probit

# Prepare data for probit (need complete cases)
probit_vars = ['FT', 'ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_centered', 'MARRIED']
df_probit = df[probit_vars].dropna()

X_probit = df_probit[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER', 'FEMALE', 'AGE_centered', 'MARRIED']]
X_probit = sm.add_constant(X_probit)
y_probit = df_probit['FT']

probit_model = Probit(y_probit, X_probit).fit(disp=0)
print(f"   Probit coefficient (ELIGIBLE_AFTER): {probit_model.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE: {probit_model.bse['ELIGIBLE_AFTER']:.4f}")

# Calculate marginal effect at mean
mfx = probit_model.get_margeff(at='mean')
did_mfx_idx = list(probit_model.params.index).index('ELIGIBLE_AFTER')
print(f"   Marginal effect at mean: {mfx.margeff[did_mfx_idx]:.4f}")
print(f"   ME Standard Error: {mfx.margeff_se[did_mfx_idx]:.4f}")

results_dict['probit_mfx'] = mfx.margeff[did_mfx_idx]
results_dict['probit_mfx_se'] = mfx.margeff_se[did_mfx_idx]

# 5d. Weighted analysis using person weights
print("\n   5d. Weighted Analysis (using PERWT)")

import statsmodels.api as sm

# Basic DiD with weights
X_weighted = df[['ELIGIBLE', 'AFTER', 'ELIGIBLE_AFTER']]
X_weighted = sm.add_constant(X_weighted)
y_weighted = df['FT']
weights = df['PERWT']

model_weighted = sm.WLS(y_weighted, X_weighted, weights=weights).fit()
print(f"   Weighted DiD coefficient: {model_weighted.params['ELIGIBLE_AFTER']:.4f}")
print(f"   SE: {model_weighted.bse['ELIGIBLE_AFTER']:.4f}")

results_dict['weighted_coef'] = model_weighted.params['ELIGIBLE_AFTER']
results_dict['weighted_se'] = model_weighted.bse['ELIGIBLE_AFTER']

# 6. Generate Figures
print("\n6. Generating Figures")
print("-"*50)

# Figure 1: FT rates over time by treatment status
fig1, ax1 = plt.subplots(figsize=(10, 6))

ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
ft_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']

ax1.plot(ft_by_year.index, ft_by_year['Treatment (26-30)'], 'o-', color='blue', linewidth=2, markersize=8, label='Treatment (26-30 at policy)')
ax1.plot(ft_by_year.index, ft_by_year['Control (31-35)'], 's--', color='red', linewidth=2, markersize=8, label='Control (31-35 at policy)')
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation (2012)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='best', fontsize=10)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.3, 0.6])
plt.tight_layout()
plt.savefig('figure1_ft_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure1_ft_trends.png")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.errorbar(event_years_plot, event_coefs_plot, yerr=[1.96*se for se in event_ses_plot],
             fmt='o', capsize=5, capthick=2, markersize=8, color='blue', ecolor='blue', linewidth=2)
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax2.fill_between([2007.5, 2012], [-0.1, -0.1], [0.1, 0.1], alpha=0.1, color='gray', label='Pre-treatment period')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax2.set_xticks(event_years_plot)
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure2_event_study.png")

# Figure 3: Distribution of FT by group
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

groups = [
    ('Treatment, Pre-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==0)),
    ('Treatment, Post-DACA', (df['ELIGIBLE']==1) & (df['AFTER']==1)),
    ('Control, Pre-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==0)),
    ('Control, Post-DACA', (df['ELIGIBLE']==0) & (df['AFTER']==1))
]

for ax, (title, mask) in zip(axes3.flat, groups):
    subset = df[mask]['FT']
    bars = ax.bar(['Not FT', 'FT'], [1-subset.mean(), subset.mean()], color=['salmon', 'steelblue'])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Proportion', fontsize=10)
    ax.set_ylim([0, 1])
    for bar, val in zip(bars, [1-subset.mean(), subset.mean()]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=10)

plt.suptitle('Full-Time Employment Distribution by Group', fontsize=14)
plt.tight_layout()
plt.savefig('figure3_ft_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure3_ft_distribution.png")

# Figure 4: Coefficient comparison across models
fig4, ax4 = plt.subplots(figsize=(10, 6))

models_names = ['Model 1\n(Basic)', 'Model 2\n(Robust SE)', 'Model 3\n(+Demographics)',
                'Model 4\n(+Education)', 'Model 5\n(+FE, Robust)', 'Model 6\n(+FE, Clustered)']
models_coefs = [results_dict['model1_coef'], results_dict['model2_coef'], results_dict['model3_coef'],
                results_dict['model4_coef'], results_dict['model5_coef'], results_dict['model6_coef']]
models_ses = [results_dict['model1_se'], results_dict['model2_se'], results_dict['model3_se'],
              results_dict['model4_se'], results_dict['model5_se'], results_dict['model6_se']]

x_pos = np.arange(len(models_names))
ax4.errorbar(x_pos, models_coefs, yerr=[1.96*se for se in models_ses],
             fmt='o', capsize=5, capthick=2, markersize=10, color='darkblue', ecolor='darkblue')
ax4.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models_names, fontsize=9)
ax4.set_ylabel('DiD Coefficient', fontsize=12)
ax4.set_title('DiD Estimates Across Model Specifications', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure4_model_comparison.png")

# 7. Create Tables for LaTeX
print("\n7. Creating Summary Tables")
print("-"*50)

# Table 1: Descriptive Statistics
print("\n   Creating descriptive statistics table...")

desc_vars = ['FT', 'FEMALE', 'AGE', 'MARRIED', 'UHRSWORK']
desc_stats = df.groupby('ELIGIBLE')[desc_vars].agg(['mean', 'std'])
desc_stats_all = df[desc_vars].agg(['mean', 'std']).T
desc_stats_all.columns = ['All Mean', 'All SD']

print("\n   Descriptive Statistics by Eligibility Status:")
print(desc_stats.round(4))

# Table 2: Main results
print("\n   Main Results Summary:")
print(f"   {'Model':<25} {'Coefficient':<12} {'SE':<10} {'95% CI':<25} {'N':<10}")
print(f"   {'-'*82}")
print(f"   {'Model 1 (Basic)':<25} {results_dict['model1_coef']:<12.4f} {results_dict['model1_se']:<10.4f} [{results_dict['model1_ci_low']:.4f}, {results_dict['model1_ci_high']:.4f}] {results_dict['model1_n']:<10}")
print(f"   {'Model 6 (Preferred)':<25} {results_dict['model6_coef']:<12.4f} {results_dict['model6_se']:<10.4f} [{results_dict['model6_ci_low']:.4f}, {results_dict['model6_ci_high']:.4f}] {results_dict['model6_n']:<10}")

# 8. Save Results
print("\n8. Saving Results")
print("-"*50)

# Save model summaries
with open('model1_summary.txt', 'w') as f:
    f.write(model1.summary().as_text())
print("   Saved: model1_summary.txt")

with open('model6_summary.txt', 'w') as f:
    f.write(model6.summary().as_text())
print("   Saved: model6_summary.txt")

# Save event study results
with open('event_study_results.txt', 'w') as f:
    f.write(model_event.summary().as_text())
print("   Saved: event_study_results.txt")

# Save key results to CSV
results_df = pd.DataFrame([results_dict])
results_df.to_csv('key_results.csv', index=False)
print("   Saved: key_results.csv")

# Save descriptive statistics
desc_full = df.groupby(['ELIGIBLE', 'AFTER']).agg({
    'FT': ['mean', 'std', 'count'],
    'FEMALE': 'mean',
    'AGE': ['mean', 'std'],
    'MARRIED': 'mean',
    'UHRSWORK': ['mean', 'std']
}).round(4)
desc_full.to_csv('descriptive_statistics.csv')
print("   Saved: descriptive_statistics.csv")

# 9. Final Summary
print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY OF KEY FINDINGS")
print("="*70)

print(f"""
RESEARCH QUESTION:
What was the causal impact of DACA eligibility on full-time employment
among Hispanic-Mexican Mexican-born individuals in the United States?

IDENTIFICATION STRATEGY:
Difference-in-Differences comparing:
  - Treatment: Ages 26-30 at June 2012 (ELIGIBLE=1)
  - Control: Ages 31-35 at June 2012 (ELIGIBLE=0)
  - Pre-period: 2008-2011
  - Post-period: 2013-2016

MAIN RESULTS:
  Sample Size: {results_dict['model6_n']:,} observations

  Simple DiD Estimate: {results_dict['simple_did']:.4f} ({results_dict['simple_did']*100:.2f} pp)

  Preferred Estimate (Model 6):
    - DiD Coefficient: {results_dict['model6_coef']:.4f}
    - Standard Error (clustered): {results_dict['model6_se']:.4f}
    - 95% CI: [{results_dict['model6_ci_low']:.4f}, {results_dict['model6_ci_high']:.4f}]
    - p-value: {results_dict['model6_pval']:.4f}

INTERPRETATION:
DACA eligibility is associated with a {abs(results_dict['model6_coef'])*100:.2f} percentage point
{'increase' if results_dict['model6_coef'] > 0 else 'decrease'} in full-time employment probability.
The effect is {'statistically significant' if results_dict['model6_pval'] < 0.05 else 'not statistically significant'} at the 5% level.

ROBUSTNESS:
  - Results are {'robust' if abs(results_dict['model1_coef'] - results_dict['model6_coef']) < 0.02 else 'sensitive'} across specifications
  - Event study suggests {'support for' if all(abs(c) < 0.03 for c in event_coefs_plot[:3]) else 'potential concerns about'} parallel trends assumption
  - Weighted analysis: {results_dict['weighted_coef']:.4f} (SE: {results_dict['weighted_se']:.4f})
  - Probit marginal effect: {results_dict['probit_mfx']:.4f} (SE: {results_dict['probit_mfx_se']:.4f})
""")

print("\nAnalysis completed successfully!")
print("="*70)

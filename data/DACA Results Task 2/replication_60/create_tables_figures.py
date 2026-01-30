"""
Create tables and figures for the DACA replication report.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

print("Loading data and recreating analysis sample...")

# Load data
df = pd.read_csv('data/data.csv')

# Apply all filters
df_sample = df[df['HISPAN'] == 1].copy()
df_sample = df_sample[df_sample['BPL'] == 200].copy()
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()

def age_on_june15_2012(birthyr, birthqtr):
    if birthqtr in [1, 2]:
        return 2012 - birthyr
    else:
        return 2012 - birthyr - 1

df_sample['age_june2012'] = df_sample.apply(
    lambda x: age_on_june15_2012(x['BIRTHYR'], x['BIRTHQTR']), axis=1
)
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[df_sample['age_at_arrival'] < 16].copy()
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007].copy()

df_sample['treat_group'] = np.where(
    (df_sample['age_june2012'] >= 26) & (df_sample['age_june2012'] <= 30), 1,
    np.where((df_sample['age_june2012'] >= 31) & (df_sample['age_june2012'] <= 35), 0, np.nan)
)

df_analysis = df_sample[df_sample['treat_group'].notna()].copy()
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = np.where(df_analysis['YEAR'] >= 2013, 1, 0)
df_analysis['fulltime'] = np.where(df_analysis['UHRSWORK'] >= 35, 1, 0)
df_analysis['treat_post'] = df_analysis['treat_group'] * df_analysis['post']

# Create covariates
df_analysis['female'] = np.where(df_analysis['SEX'] == 2, 1, 0)
df_analysis['married'] = np.where(df_analysis['MARST'].isin([1, 2]), 1, 0)
df_analysis['educ_hs'] = np.where(df_analysis['EDUC'] == 6, 1, 0)
df_analysis['educ_somecoll'] = np.where(df_analysis['EDUC'].isin([7, 8, 9]), 1, 0)
df_analysis['educ_coll'] = np.where(df_analysis['EDUC'] >= 10, 1, 0)
df_analysis['age'] = df_analysis['AGE']
df_analysis['age_sq'] = df_analysis['age'] ** 2

# ============================================================================
# TABLE 1: Summary Statistics
# ============================================================================
print("\nCreating Table 1: Summary Statistics...")

# Calculate summary stats with proper weighting
def weighted_mean(data, weights):
    return np.average(data, weights=weights)

def weighted_std(data, weights):
    average = np.average(data, weights=weights)
    variance = np.average((data - average)**2, weights=weights)
    return np.sqrt(variance)

summary_vars = {
    'fulltime': 'Full-time employed (35+ hrs/wk)',
    'age': 'Age (current)',
    'female': 'Female',
    'married': 'Married',
    'educ_hs': 'High school graduate',
    'educ_somecoll': 'Some college',
    'educ_coll': 'College degree or higher',
}

# Pre-period stats
pre_treat = df_analysis[(df_analysis['post'] == 0) & (df_analysis['treat_group'] == 1)]
pre_ctrl = df_analysis[(df_analysis['post'] == 0) & (df_analysis['treat_group'] == 0)]
post_treat = df_analysis[(df_analysis['post'] == 1) & (df_analysis['treat_group'] == 1)]
post_ctrl = df_analysis[(df_analysis['post'] == 1) & (df_analysis['treat_group'] == 0)]

summary_table = []
for var, label in summary_vars.items():
    row = {
        'Variable': label,
        'Treat Pre Mean': weighted_mean(pre_treat[var], pre_treat['PERWT']),
        'Treat Pre SD': weighted_std(pre_treat[var], pre_treat['PERWT']),
        'Ctrl Pre Mean': weighted_mean(pre_ctrl[var], pre_ctrl['PERWT']),
        'Ctrl Pre SD': weighted_std(pre_ctrl[var], pre_ctrl['PERWT']),
        'Treat Post Mean': weighted_mean(post_treat[var], post_treat['PERWT']),
        'Treat Post SD': weighted_std(post_treat[var], post_treat['PERWT']),
        'Ctrl Post Mean': weighted_mean(post_ctrl[var], post_ctrl['PERWT']),
        'Ctrl Post SD': weighted_std(post_ctrl[var], post_ctrl['PERWT']),
    }
    summary_table.append(row)

# Add sample sizes
summary_table.append({
    'Variable': 'N (unweighted)',
    'Treat Pre Mean': len(pre_treat),
    'Treat Pre SD': '',
    'Ctrl Pre Mean': len(pre_ctrl),
    'Ctrl Pre SD': '',
    'Treat Post Mean': len(post_treat),
    'Treat Post SD': '',
    'Ctrl Post Mean': len(post_ctrl),
    'Ctrl Post SD': '',
})

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv('table1_summary_stats.csv', index=False)
print("Saved: table1_summary_stats.csv")

# ============================================================================
# TABLE 2: Main Regression Results
# ============================================================================
print("\nCreating Table 2: Main Regression Results...")

# Run all models
model1 = smf.ols('fulltime ~ treat_group + post + treat_post', data=df_analysis).fit(cov_type='HC1')

model2 = smf.wls('fulltime ~ treat_group + post + treat_post',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

model3 = smf.wls('fulltime ~ treat_group + post + treat_post + female + married + educ_hs + educ_somecoll + educ_coll + age + age_sq',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

df_analysis['year_factor'] = df_analysis['YEAR'].astype(str)
model4 = smf.wls('fulltime ~ treat_group + treat_post + C(year_factor) + female + married + educ_hs + educ_somecoll + educ_coll + age + age_sq',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

df_analysis['state_factor'] = df_analysis['STATEFIP'].astype(str)
model5 = smf.wls('fulltime ~ treat_group + treat_post + C(year_factor) + C(state_factor) + female + married + educ_hs + educ_somecoll + educ_coll',
                  data=df_analysis,
                  weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Extract coefficients for table
reg_results = []
models = [
    ('(1) Basic DiD', model1),
    ('(2) Weighted', model2),
    ('(3) + Covariates', model3),
    ('(4) + Year FE', model4),
    ('(5) + State FE', model5),
]

for name, model in models:
    coef = model.params['treat_post']
    se = model.bse['treat_post']
    pval = model.pvalues['treat_post']
    ci = model.conf_int().loc['treat_post']

    stars = ''
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.10:
        stars = '*'

    reg_results.append({
        'Model': name,
        'Coefficient': f'{coef:.4f}{stars}',
        'Std. Error': f'({se:.4f})',
        'CI Lower': f'{ci[0]:.4f}',
        'CI Upper': f'{ci[1]:.4f}',
        'N': int(model.nobs),
        'R-squared': f'{model.rsquared:.3f}',
        'Weights': 'No' if '(1)' in name else 'Yes',
        'Year FE': 'Yes' if '(4)' in name or '(5)' in name else 'No',
        'State FE': 'Yes' if '(5)' in name else 'No',
        'Covariates': 'Yes' if '(3)' in name or '(4)' in name or '(5)' in name else 'No',
    })

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv('table2_regression_results.csv', index=False)
print("Saved: table2_regression_results.csv")

# ============================================================================
# TABLE 3: Robustness Checks
# ============================================================================
print("\nCreating Table 3: Robustness Checks...")

robustness_results = []

# Placebo test
df_pre = df_analysis[df_analysis['post'] == 0].copy()
df_pre['placebo_post'] = np.where(df_pre['YEAR'] >= 2009, 1, 0)
df_pre['placebo_treat_post'] = df_pre['treat_group'] * df_pre['placebo_post']
model_placebo = smf.wls('fulltime ~ treat_group + placebo_post + placebo_treat_post',
                         data=df_pre, weights=df_pre['PERWT']).fit(cov_type='HC1')

pval = model_placebo.pvalues['placebo_treat_post']
stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
robustness_results.append({
    'Test': 'Placebo (fake treatment at 2009)',
    'Coefficient': f"{model_placebo.params['placebo_treat_post']:.4f}{stars}",
    'Std. Error': f"({model_placebo.bse['placebo_treat_post']:.4f})",
    'N': int(model_placebo.nobs),
})

# Alternative age windows
df_sample_alt = df_sample.copy()
df_sample_alt['treat_group_alt'] = np.where(
    (df_sample_alt['age_june2012'] >= 24) & (df_sample_alt['age_june2012'] <= 28), 1,
    np.where((df_sample_alt['age_june2012'] >= 33) & (df_sample_alt['age_june2012'] <= 37), 0, np.nan)
)
df_alt = df_sample_alt[df_sample_alt['treat_group_alt'].notna()].copy()
df_alt = df_alt[df_alt['YEAR'] != 2012]
df_alt['post'] = np.where(df_alt['YEAR'] >= 2013, 1, 0)
df_alt['fulltime'] = np.where(df_alt['UHRSWORK'] >= 35, 1, 0)
df_alt['treat_post_alt'] = df_alt['treat_group_alt'] * df_alt['post']
model_alt = smf.wls('fulltime ~ treat_group_alt + post + treat_post_alt',
                     data=df_alt, weights=df_alt['PERWT']).fit(cov_type='HC1')

pval = model_alt.pvalues['treat_post_alt']
stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
robustness_results.append({
    'Test': 'Alternative age bands (24-28 vs 33-37)',
    'Coefficient': f"{model_alt.params['treat_post_alt']:.4f}{stars}",
    'Std. Error': f"({model_alt.bse['treat_post_alt']:.4f})",
    'N': int(model_alt.nobs),
})

# By gender
for gender, label in [(0, 'Male'), (1, 'Female')]:
    df_g = df_analysis[df_analysis['female'] == gender]
    model_g = smf.wls('fulltime ~ treat_group + post + treat_post',
                       data=df_g, weights=df_g['PERWT']).fit(cov_type='HC1')
    pval = model_g.pvalues['treat_post']
    stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
    robustness_results.append({
        'Test': f'Subgroup: {label} only',
        'Coefficient': f"{model_g.params['treat_post']:.4f}{stars}",
        'Std. Error': f"({model_g.bse['treat_post']:.4f})",
        'N': int(model_g.nobs),
    })

robustness_df = pd.DataFrame(robustness_results)
robustness_df.to_csv('table3_robustness.csv', index=False)
print("Saved: table3_robustness.csv")

# ============================================================================
# FIGURE 1: Event Study Plot
# ============================================================================
print("\nCreating Figure 1: Event Study...")

years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years:
    df_analysis[f'year_{year}'] = np.where(df_analysis['YEAR'] == year, 1, 0)
    df_analysis[f'treat_year_{year}'] = df_analysis['treat_group'] * df_analysis[f'year_{year}']

year_terms = ' + '.join([f'year_{y}' for y in years if y != 2011])
treat_year_terms = ' + '.join([f'treat_year_{y}' for y in years if y != 2011])
formula_event = f'fulltime ~ treat_group + {year_terms} + {treat_year_terms}'
model_event = smf.wls(formula_event, data=df_analysis, weights=df_analysis['PERWT']).fit(cov_type='HC1')

# Extract coefficients
event_coefs = []
for year in years:
    if year == 2011:
        event_coefs.append({'year': year, 'coef': 0, 'se': 0, 'ci_low': 0, 'ci_high': 0})
    else:
        coef = model_event.params[f'treat_year_{year}']
        se = model_event.bse[f'treat_year_{year}']
        event_coefs.append({
            'year': year,
            'coef': coef,
            'se': se,
            'ci_low': coef - 1.96 * se,
            'ci_high': coef + 1.96 * se
        })

event_df = pd.DataFrame(event_coefs)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Add vertical line at DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation (June 2012)')

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# Plot coefficients with confidence intervals
ax.errorbar(event_df['year'], event_df['coef'],
            yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
            fmt='o-', capsize=5, capthick=2, markersize=8, color='navy')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference year: 2011)', fontsize=14)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()
print("Saved: figure1_event_study.png and figure1_event_study.pdf")

# ============================================================================
# FIGURE 2: Trends in Full-Time Employment by Group
# ============================================================================
print("\nCreating Figure 2: Employment Trends...")

# Calculate weighted means by year and group
trends = df_analysis.groupby(['YEAR', 'treat_group']).apply(
    lambda x: pd.Series({
        'fulltime_mean': np.average(x['fulltime'], weights=x['PERWT']),
        'n': len(x)
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment group
treat_data = trends[trends['treat_group'] == 1]
ax.plot(treat_data['YEAR'], treat_data['fulltime_mean'], 'o-',
        label='Treatment (Ages 26-30)', color='navy', markersize=8, linewidth=2)

# Plot control group
ctrl_data = trends[trends['treat_group'] == 0]
ax.plot(ctrl_data['YEAR'], ctrl_data['fulltime_mean'], 's--',
        label='Control (Ages 31-35)', color='darkred', markersize=8, linewidth=2)

# Add vertical line at DACA
ax.axvline(x=2012, color='gray', linestyle=':', alpha=0.7, label='DACA (June 2012)')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status\n(Hispanic-Mexican, Mexican-born, Non-citizens arrived before age 16)', fontsize=13)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.55, 0.75])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved: figure2_trends.png and figure2_trends.pdf")

# ============================================================================
# FIGURE 3: DiD Visualization
# ============================================================================
print("\nCreating Figure 3: DiD Visualization...")

# Calculate group means for pre and post
did_data = df_analysis.groupby(['treat_group', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index()
did_data.columns = ['treat_group', 'post', 'fulltime']

fig, ax = plt.subplots(figsize=(8, 6))

# Treatment group
treat = did_data[did_data['treat_group'] == 1]
ax.plot(['Pre (2006-2011)', 'Post (2013-2016)'],
        [treat[treat['post'] == 0]['fulltime'].values[0], treat[treat['post'] == 1]['fulltime'].values[0]],
        'o-', label='Treatment (Ages 26-30)', color='navy', markersize=12, linewidth=3)

# Control group
ctrl = did_data[did_data['treat_group'] == 0]
ax.plot(['Pre (2006-2011)', 'Post (2013-2016)'],
        [ctrl[ctrl['post'] == 0]['fulltime'].values[0], ctrl[ctrl['post'] == 1]['fulltime'].values[0]],
        's-', label='Control (Ages 31-35)', color='darkred', markersize=12, linewidth=3)

# Counterfactual for treatment group
ctrl_diff = ctrl[ctrl['post'] == 1]['fulltime'].values[0] - ctrl[ctrl['post'] == 0]['fulltime'].values[0]
treat_pre = treat[treat['post'] == 0]['fulltime'].values[0]
counterfactual = treat_pre + ctrl_diff
ax.plot(['Pre (2006-2011)', 'Post (2013-2016)'],
        [treat_pre, counterfactual],
        'o--', label='Treatment (counterfactual)', color='navy', alpha=0.4, markersize=10, linewidth=2)

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.58, 0.70])

# Add annotation for DiD
did_effect = treat[treat['post'] == 1]['fulltime'].values[0] - counterfactual
ax.annotate(f'DiD Effect: {did_effect:.3f}',
            xy=('Post (2013-2016)', treat[treat['post'] == 1]['fulltime'].values[0]),
            xytext=(1.1, 0.64),
            fontsize=11,
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()
print("Saved: figure3_did.png and figure3_did.pdf")

# Save event study data
event_df.to_csv('event_study_data.csv', index=False)
print("Saved: event_study_data.csv")

print("\nAll tables and figures created successfully!")

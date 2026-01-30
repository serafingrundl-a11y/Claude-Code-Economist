"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Create derived variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)

# 1. Parallel Trends Figure
print("Creating Figure 1: Parallel Trends...")
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate annual means by treatment status
annual_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
annual_means.columns = ['Control (31-35)', 'Treatment (26-30)']

years = annual_means.index.tolist()
x_positions = list(range(len(years)))

ax.plot(x_positions, annual_means['Treatment (26-30)'], 'b-o', linewidth=2, markersize=8, label='Treatment (26-30)')
ax.plot(x_positions, annual_means['Control (31-35)'], 'r--s', linewidth=2, markersize=8, label='Control (31-35)')

# Add vertical line at DACA implementation
# Find position between 2011 and 2013 (2012 is excluded)
daca_position = x_positions[years.index(2011)] + 0.5
ax.axvline(x=daca_position, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(daca_position + 0.1, ax.get_ylim()[1] - 0.02, 'DACA\n(June 2012)', fontsize=10, ha='left')

ax.set_xticks(x_positions)
ax.set_xticklabels(years)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.55, 0.80)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png")

# 2. DID Visualization
print("Creating Figure 2: DID Visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

# Group means for before/after
pre_years = [2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

treat_pre = df[(df['ELIGIBLE']==1) & (df['YEAR'].isin(pre_years))]['FT'].mean()
treat_post = df[(df['ELIGIBLE']==1) & (df['YEAR'].isin(post_years))]['FT'].mean()
ctrl_pre = df[(df['ELIGIBLE']==0) & (df['YEAR'].isin(pre_years))]['FT'].mean()
ctrl_post = df[(df['ELIGIBLE']==0) & (df['YEAR'].isin(post_years))]['FT'].mean()

# Plot
periods = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
x = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x - width/2, [treat_pre, treat_post], width, label='Treatment (26-30)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, [ctrl_pre, ctrl_post], width, label='Control (31-35)', color='indianred', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 0.85)
ax.grid(True, alpha=0.3, axis='y')

# Add DID calculation annotation
did_estimate = (treat_post - treat_pre) - (ctrl_post - ctrl_pre)
ax.annotate(f'DID Estimate: {did_estimate:.4f}',
            xy=(0.5, 0.05), xycoords='axes fraction',
            fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure2_did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_did_visualization.png")

# 3. FT Employment Distribution by Group
print("Creating Figure 3: FT Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-period
pre_data = df[df['AFTER'] == 0]
treat_pre_ft = pre_data[pre_data['ELIGIBLE']==1]['FT'].value_counts(normalize=True).sort_index()
ctrl_pre_ft = pre_data[pre_data['ELIGIBLE']==0]['FT'].value_counts(normalize=True).sort_index()

x = np.array([0, 1])
width = 0.35
axes[0].bar(x - width/2, [treat_pre_ft.get(0, 0), treat_pre_ft.get(1, 0)], width,
            label='Treatment', color='steelblue', alpha=0.8)
axes[0].bar(x + width/2, [ctrl_pre_ft.get(0, 0), ctrl_pre_ft.get(1, 0)], width,
            label='Control', color='indianred', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Not Full-Time', 'Full-Time'])
axes[0].set_ylabel('Proportion')
axes[0].set_title('Pre-DACA Period (2008-2011)')
axes[0].legend()
axes[0].set_ylim(0, 0.8)

# Post-period
post_data = df[df['AFTER'] == 1]
treat_post_ft = post_data[post_data['ELIGIBLE']==1]['FT'].value_counts(normalize=True).sort_index()
ctrl_post_ft = post_data[post_data['ELIGIBLE']==0]['FT'].value_counts(normalize=True).sort_index()

axes[1].bar(x - width/2, [treat_post_ft.get(0, 0), treat_post_ft.get(1, 0)], width,
            label='Treatment', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, [ctrl_post_ft.get(0, 0), ctrl_post_ft.get(1, 0)], width,
            label='Control', color='indianred', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Not Full-Time', 'Full-Time'])
axes[1].set_ylabel('Proportion')
axes[1].set_title('Post-DACA Period (2013-2016)')
axes[1].legend()
axes[1].set_ylim(0, 0.8)

plt.suptitle('Full-Time Employment Distribution by Group and Period', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figure3_ft_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_ft_distribution.png")

# 4. Event Study Plot
print("Creating Figure 4: Event Study...")
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate difference (treatment - control) by year
annual_diff = annual_means['Treatment (26-30)'] - annual_means['Control (31-35)']

# Bootstrap confidence intervals
def bootstrap_diff(group_data, n_bootstrap=1000):
    diffs = []
    for _ in range(n_bootstrap):
        sample = group_data.sample(n=len(group_data), replace=True)
        treat_mean = sample[sample['ELIGIBLE']==1]['FT'].mean()
        ctrl_mean = sample[sample['ELIGIBLE']==0]['FT'].mean()
        diffs.append(treat_mean - ctrl_mean)
    return np.percentile(diffs, [2.5, 97.5])

ci_low = []
ci_high = []
for year in years:
    year_data = df[df['YEAR'] == year]
    low, high = bootstrap_diff(year_data)
    ci_low.append(low)
    ci_high.append(high)

# Plot
ax.errorbar(x_positions, annual_diff.values,
            yerr=[annual_diff.values - ci_low, ci_high - annual_diff.values],
            fmt='o-', color='navy', linewidth=2, markersize=8, capsize=5, capthick=2,
            label='Difference (Treatment - Control)')

# Add horizontal line at 0
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Add vertical line at DACA implementation
ax.axvline(x=daca_position, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(daca_position + 0.1, ax.get_ylim()[1] - 0.01, 'DACA', fontsize=10, color='red', ha='left')

ax.set_xticks(x_positions)
ax.set_xticklabels(years)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference in FT Rate (Treatment - Control)', fontsize=12)
ax.set_title('Event Study: Treatment-Control Difference Over Time', fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure4_event_study.png")

# 5. Coefficient Plot
print("Creating Figure 5: Coefficient Plot...")
import statsmodels.formula.api as smf

# Run all models and collect estimates
df['ELIGIBLE_X_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Create education dummies
df['EDUC_HS'] = (df['EDUC_RECODE'] == 'High School Degree').astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC_RECODE'] == 'Some College').astype(int)
df['EDUC_TWOYEAR'] = (df['EDUC_RECODE'] == 'Two-Year Degree').astype(int)
df['EDUC_BAPLUS'] = (df['EDUC_RECODE'] == 'BA+').astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

models_info = [
    ("Basic DID", 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER'),
    ("+ Demographics", 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE'),
    ("+ Education", 'FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER + FEMALE + MARRIED + NCHILD + AGE + EDUC_HS + EDUC_SOMECOLL + EDUC_TWOYEAR + EDUC_BAPLUS'),
]

estimates = []
cis_low = []
cis_high = []
names = []

for name, formula in models_info:
    model = smf.ols(formula, data=df).fit(cov_type='HC1')
    estimates.append(model.params['ELIGIBLE_X_AFTER'])
    cis_low.append(model.conf_int().loc['ELIGIBLE_X_AFTER', 0])
    cis_high.append(model.conf_int().loc['ELIGIBLE_X_AFTER', 1])
    names.append(name)

fig, ax = plt.subplots(figsize=(8, 5))

y_pos = np.arange(len(names))
errors = [[e - l for e, l in zip(estimates, cis_low)],
          [h - e for e, h in zip(estimates, cis_high)]]

ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', color='navy',
            markersize=10, capsize=5, capthick=2, linewidth=2)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('DID Estimate (Effect on FT Employment)', fontsize=12)
ax.set_title('Sensitivity of DID Estimate to Model Specification', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add estimate values
for i, (est, low, high) in enumerate(zip(estimates, cis_low, cis_high)):
    ax.annotate(f'{est:.4f}', xy=(est, i), xytext=(5, 0),
                textcoords='offset points', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('figure5_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure5_coefficient_plot.png")

# 6. Heterogeneity by Sex
print("Creating Figure 6: Heterogeneity by Sex...")
fig, ax = plt.subplots(figsize=(8, 5))

# Run models by sex
df_male = df[df['SEX'] == 1]
df_female = df[df['SEX'] == 2]

model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_male).fit(cov_type='HC1')
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df_female).fit(cov_type='HC1')
model_all = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_X_AFTER', data=df).fit(cov_type='HC1')

names = ['All', 'Male', 'Female']
estimates = [model_all.params['ELIGIBLE_X_AFTER'],
             model_male.params['ELIGIBLE_X_AFTER'],
             model_female.params['ELIGIBLE_X_AFTER']]
cis_low = [model_all.conf_int().loc['ELIGIBLE_X_AFTER', 0],
           model_male.conf_int().loc['ELIGIBLE_X_AFTER', 0],
           model_female.conf_int().loc['ELIGIBLE_X_AFTER', 0]]
cis_high = [model_all.conf_int().loc['ELIGIBLE_X_AFTER', 1],
            model_male.conf_int().loc['ELIGIBLE_X_AFTER', 1],
            model_female.conf_int().loc['ELIGIBLE_X_AFTER', 1]]

y_pos = np.arange(len(names))
errors = [[e - l for e, l in zip(estimates, cis_low)],
          [h - e for e, h in zip(estimates, cis_high)]]

ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', color='navy',
            markersize=10, capsize=5, capthick=2, linewidth=2)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel('DID Estimate (Effect on FT Employment)', fontsize=12)
ax.set_title('Heterogeneity Analysis: Effect by Sex', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

for i, (est, low, high) in enumerate(zip(estimates, cis_low, cis_high)):
    ax.annotate(f'{est:.4f}', xy=(est, i), xytext=(5, 0),
                textcoords='offset points', fontsize=9, va='center')

plt.tight_layout()
plt.savefig('figure6_heterogeneity_sex.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure6_heterogeneity_sex.png")

# 7. Sample Size by Year and Group
print("Creating Figure 7: Sample Sizes...")
fig, ax = plt.subplots(figsize=(10, 6))

sample_sizes = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_sizes.columns = ['Control (31-35)', 'Treatment (26-30)']

x = np.arange(len(years))
width = 0.35

bars1 = ax.bar(x - width/2, sample_sizes['Treatment (26-30)'], width,
               label='Treatment (26-30)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, sample_sizes['Control (31-35)'], width,
               label='Control (31-35)', color='indianred', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Observations', fontsize=12)
ax.set_title('Sample Size by Year and Treatment Status', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add vertical line at DACA
ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)

plt.tight_layout()
plt.savefig('figure7_sample_sizes.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure7_sample_sizes.png")

print("\nAll figures created successfully!")

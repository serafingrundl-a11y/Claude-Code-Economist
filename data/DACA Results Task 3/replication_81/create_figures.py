"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_labelled_version.csv')

# Create interaction term and other variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 'Female').astype(int)
df['MARRIED'] = df['MARST'].isin(['Married, spouse present', 'Married, spouse absent']).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Helper function for weighted mean
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

# Figure 1: Parallel Trends Plot
print("Creating Figure 1: Parallel Trends...")
fig, ax = plt.subplots(figsize=(10, 6))

years = sorted(df['YEAR'].unique())
treat_means = []
ctrl_means = []

for year in years:
    treat_data = df[(df['YEAR'] == year) & (df['ELIGIBLE'] == 1)]
    ctrl_data = df[(df['YEAR'] == year) & (df['ELIGIBLE'] == 0)]
    treat_means.append(weighted_mean(treat_data, 'FT', 'PERWT'))
    ctrl_means.append(weighted_mean(ctrl_data, 'FT', 'PERWT'))

ax.plot(years, treat_means, 'b-o', label='Treatment (Ages 26-30 in June 2012)', linewidth=2, markersize=8)
ax.plot(years, ctrl_means, 'r-s', label='Control (Ages 31-35 in June 2012)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.55, 0.80)
ax.grid(True, alpha=0.3)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure1_parallel_trends.png")

# Figure 2: DiD Visualization
print("Creating Figure 2: DiD Visualization...")
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate pre and post means
pre_treat = weighted_mean(df[(df['AFTER'] == 0) & (df['ELIGIBLE'] == 1)], 'FT', 'PERWT')
post_treat = weighted_mean(df[(df['AFTER'] == 1) & (df['ELIGIBLE'] == 1)], 'FT', 'PERWT')
pre_ctrl = weighted_mean(df[(df['AFTER'] == 0) & (df['ELIGIBLE'] == 0)], 'FT', 'PERWT')
post_ctrl = weighted_mean(df[(df['AFTER'] == 1) & (df['ELIGIBLE'] == 0)], 'FT', 'PERWT')

# Counterfactual
counterfactual = pre_treat + (post_ctrl - pre_ctrl)

x = [0, 1]
ax.plot(x, [pre_treat, post_treat], 'b-o', label='Treatment (Actual)', linewidth=3, markersize=12)
ax.plot(x, [pre_ctrl, post_ctrl], 'r-s', label='Control', linewidth=3, markersize=12)
ax.plot(x, [pre_treat, counterfactual], 'b--', label='Treatment (Counterfactual)', linewidth=2, alpha=0.7)

# DiD arrow
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.05, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
        fontsize=12, color='green', fontweight='bold')

ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)'])
ax.legend(loc='upper left', fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2_did_visualization.png")

# Figure 3: Sample Distribution by Year
print("Creating Figure 3: Sample Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sample sizes
sample_sizes = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_sizes.columns = ['Control', 'Treatment']

sample_sizes.plot(kind='bar', ax=axes[0], color=['red', 'blue'], alpha=0.7)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Number of Observations', fontsize=12)
axes[0].set_title('Sample Size by Year and Group', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].tick_params(axis='x', rotation=45)

# Weighted population
weighted_pop = df.groupby(['YEAR', 'ELIGIBLE'])['PERWT'].sum().unstack() / 1000
weighted_pop.columns = ['Control', 'Treatment']

weighted_pop.plot(kind='bar', ax=axes[1], color=['red', 'blue'], alpha=0.7)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Weighted Population (thousands)', fontsize=12)
axes[1].set_title('Weighted Population by Year and Group', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figure3_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure3_sample_distribution.png")

# Figure 4: Heterogeneity by Sex
print("Creating Figure 4: Heterogeneity by Sex...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, sex in enumerate(['Male', 'Female']):
    subset = df[df['SEX'] == sex]

    treat_means_sex = []
    ctrl_means_sex = []

    for year in years:
        treat_data = subset[(subset['YEAR'] == year) & (subset['ELIGIBLE'] == 1)]
        ctrl_data = subset[(subset['YEAR'] == year) & (subset['ELIGIBLE'] == 0)]
        if len(treat_data) > 0:
            treat_means_sex.append(weighted_mean(treat_data, 'FT', 'PERWT'))
        else:
            treat_means_sex.append(np.nan)
        if len(ctrl_data) > 0:
            ctrl_means_sex.append(weighted_mean(ctrl_data, 'FT', 'PERWT'))
        else:
            ctrl_means_sex.append(np.nan)

    axes[idx].plot(years, treat_means_sex, 'b-o', label='Treatment', linewidth=2, markersize=8)
    axes[idx].plot(years, ctrl_means_sex, 'r-s', label='Control', linewidth=2, markersize=8)
    axes[idx].axvline(x=2012, color='gray', linestyle='--', linewidth=2)
    axes[idx].set_xlabel('Year', fontsize=12)
    axes[idx].set_ylabel('Full-Time Employment Rate', fontsize=12)
    axes[idx].set_title(f'{sex}', fontsize=14)
    axes[idx].legend(loc='lower right', fontsize=10)
    axes[idx].set_ylim(0.35, 0.95)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xticks(years)

plt.suptitle('Full-Time Employment Trends by Sex', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figure4_heterogeneity_sex.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure4_heterogeneity_sex.png")

# Figure 5: Coefficient Plot
print("Creating Figure 5: Coefficient Plot...")
fig, ax = plt.subplots(figsize=(10, 6))

# Model results
models = ['(1) Basic\nOLS', '(2) Weighted\nOLS', '(3) Clustered\nSE',
          '(4) + Demo\nControls', '(5) + State\nFE', '(6) + Year\nFE']
estimates = [0.0643, 0.0748, 0.0748, 0.0616, 0.0611, 0.0583]
ses = [0.0153, 0.0152, 0.0203, 0.0213, 0.0219, 0.0212]

y_pos = np.arange(len(models))
colors = ['gray', 'gray', 'steelblue', 'steelblue', 'steelblue', 'darkblue']

for i in range(len(models)):
    ax.errorbar(estimates[i], y_pos[i], xerr=1.96*ses[i], fmt='o',
                color=colors[i], capsize=5, capthick=2, markersize=10)

ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Null Effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Effect on FT Employment Probability)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim(-0.02, 0.14)

plt.tight_layout()
plt.savefig('figure5_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure5_coefficient_plot.png")

# Figure 6: Education Distribution
print("Creating Figure 6: Education Distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_dist = df.groupby(['ELIGIBLE', 'EDUC_RECODE']).size().unstack().fillna(0)
educ_dist = educ_dist[educ_order]
educ_dist_pct = educ_dist.div(educ_dist.sum(axis=1), axis=0) * 100

x = np.arange(len(educ_order))
width = 0.35

bars1 = ax.bar(x - width/2, educ_dist_pct.loc[0], width, label='Control', color='red', alpha=0.7)
bars2 = ax.bar(x + width/2, educ_dist_pct.loc[1], width, label='Treatment', color='blue', alpha=0.7)

ax.set_xlabel('Education Level', fontsize=12)
ax.set_ylabel('Percentage of Group', fontsize=12)
ax.set_title('Education Distribution by DACA Eligibility', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(educ_order, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure6_education_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure6_education_distribution.png")

print("\nAll figures created successfully!")

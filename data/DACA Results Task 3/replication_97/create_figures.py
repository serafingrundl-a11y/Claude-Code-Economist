"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_97\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

def weighted_mean(x, weights):
    return np.average(x, weights=weights)

# Create output directory path
output_dir = r"C:\Users\seraf\DACA Results Task 3\replication_97"

# ============================================================================
# Figure 1: Event Study Plot
# ============================================================================
print("Creating Figure 1: Event Study Plot...")

years = sorted(df['YEAR'].unique())
year_effects = []

for year in years:
    subset = df[df['YEAR'] == year]
    treat = subset[subset['ELIGIBLE'] == 1]
    ctrl = subset[subset['ELIGIBLE'] == 0]

    treat_mean = weighted_mean(treat['FT'], treat['PERWT'])
    ctrl_mean = weighted_mean(ctrl['FT'], ctrl['PERWT'])
    diff = treat_mean - ctrl_mean
    year_effects.append({'year': year, 'diff': diff, 'treat': treat_mean, 'ctrl': ctrl_mean})

year_df = pd.DataFrame(year_effects)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot the differences
colors = ['#1f77b4' if y < 2012 else '#d62728' for y in year_df['year']]
bars = ax.bar(year_df['year'], year_df['diff'], color=colors, edgecolor='black', linewidth=0.5)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add vertical line at 2012 (treatment)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')

# Add mean lines for pre and post
pre_mean = year_df[year_df['year'] < 2012]['diff'].mean()
post_mean = year_df[year_df['year'] > 2012]['diff'].mean()

ax.axhline(y=pre_mean, xmin=0, xmax=0.45, color='#1f77b4', linestyle='--', linewidth=2, alpha=0.7)
ax.axhline(y=post_mean, xmin=0.55, xmax=1, color='#d62728', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference in FT Employment Rate\n(Treatment - Control)', fontsize=12)
ax.set_title('Event Study: Difference in Full-Time Employment\nBetween Treatment and Control Groups', fontsize=14)
ax.set_xticks(years)
ax.legend(loc='upper left')

# Add text annotations
ax.text(2009.5, pre_mean + 0.01, f'Pre-period avg: {pre_mean:.3f}', fontsize=10, color='#1f77b4')
ax.text(2014.5, post_mean + 0.01, f'Post-period avg: {post_mean:.3f}', fontsize=10, color='#d62728')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure1_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure1_event_study.png/pdf")

# ============================================================================
# Figure 2: Parallel Trends Plot
# ============================================================================
print("Creating Figure 2: Parallel Trends Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(year_df['year'], year_df['treat'], 'o-', color='#d62728', linewidth=2,
        markersize=8, label='Treatment (Ages 26-30)', markeredgecolor='black')
ax.plot(year_df['year'], year_df['ctrl'], 's-', color='#1f77b4', linewidth=2,
        markersize=8, label='Control (Ages 31-35)', markeredgecolor='black')

# Add vertical line at 2012
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Full-Time Employment Trends by Group', fontsize=14)
ax.set_xticks(years)
ax.legend(loc='lower right')
ax.set_ylim(0.55, 0.75)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure2_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure2_parallel_trends.png/pdf")

# ============================================================================
# Figure 3: DiD Visualization (2x2)
# ============================================================================
print("Creating Figure 3: DiD 2x2 Visualization...")

# Calculate weighted means for 2x2
wt_means = {}
for eligible in [0, 1]:
    for after in [0, 1]:
        mask = (df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)
        subset = df[mask]
        wt_means[(eligible, after)] = weighted_mean(subset['FT'], subset['PERWT'])

fig, ax = plt.subplots(figsize=(8, 6))

# X positions
x_pre = 0
x_post = 1

# Plot lines
ax.plot([x_pre, x_post], [wt_means[(1, 0)], wt_means[(1, 1)]], 'o-', color='#d62728',
        linewidth=2.5, markersize=12, label='Treatment (Ages 26-30)', markeredgecolor='black')
ax.plot([x_pre, x_post], [wt_means[(0, 0)], wt_means[(0, 1)]], 's-', color='#1f77b4',
        linewidth=2.5, markersize=12, label='Control (Ages 31-35)', markeredgecolor='black')

# Add counterfactual line for treatment
counterfactual = wt_means[(1, 0)] + (wt_means[(0, 1)] - wt_means[(0, 0)])
ax.plot([x_pre, x_post], [wt_means[(1, 0)], counterfactual], '--', color='#d62728',
        linewidth=1.5, alpha=0.5, label='Treatment Counterfactual')

# Add arrow showing DiD effect
did_effect = wt_means[(1, 1)] - counterfactual
ax.annotate('', xy=(x_post + 0.05, wt_means[(1, 1)]), xytext=(x_post + 0.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(x_post + 0.1, (wt_means[(1, 1)] + counterfactual) / 2,
        f'DiD = {did_effect:.3f}', fontsize=12, color='green', fontweight='bold', va='center')

# Formatting
ax.set_xticks([x_pre, x_post])
ax.set_xticklabels(['Pre-Period\n(2008-2011)', 'Post-Period\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Estimate', fontsize=14)
ax.legend(loc='upper left')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)

# Add data labels
for (eligible, after), val in wt_means.items():
    x = x_pre if after == 0 else x_post
    offset = 0.015 if eligible == 1 else -0.02
    ax.text(x, val + offset, f'{val:.3f}', fontsize=10, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure3_did_visualization.png/pdf")

# ============================================================================
# Figure 4: Coefficient Comparison Across Models
# ============================================================================
print("Creating Figure 4: Coefficient Comparison...")

# Model results (from analysis.py output)
models_data = [
    ('(1) OLS\nunweighted', 0.0643, 0.0153),
    ('(2) WLS\nweighted', 0.0748, 0.0152),
    ('(3) OLS\nrobust SE', 0.0643, 0.0153),
    ('(4) WLS\nrobust SE', 0.0748, 0.0181),
    ('(5) WLS +\ndemographics', 0.0668, 0.0168),
    ('(6) WLS +\neducation', 0.0640, 0.0167),
    ('(7) WLS +\nyear FE', 0.0613, 0.0167),
    ('(8) WLS +\nstate/year FE', 0.0607, 0.0166),
]

fig, ax = plt.subplots(figsize=(12, 6))

names = [m[0] for m in models_data]
estimates = [m[1] for m in models_data]
ses = [m[2] for m in models_data]

x_pos = np.arange(len(names))
ci_lower = [e - 1.96*s for e, s in zip(estimates, ses)]
ci_upper = [e + 1.96*s for e, s in zip(estimates, ses)]

# Plot points with error bars
colors = ['#1f77b4' if i != 3 else '#d62728' for i in range(len(names))]
ax.errorbar(x_pos, estimates, yerr=[np.array(estimates)-np.array(ci_lower),
                                     np.array(ci_upper)-np.array(estimates)],
            fmt='o', markersize=10, capsize=5, capthick=2, color='#1f77b4',
            markeredgecolor='black', ecolor='gray')

# Highlight preferred model
ax.scatter([3], [estimates[3]], s=200, color='#d62728', zorder=5,
           marker='o', edgecolors='black', linewidth=2, label='Preferred Model')

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel('DiD Estimate (ELIGIBLE × AFTER)', fontsize=12)
ax.set_title('Comparison of DiD Estimates Across Model Specifications\n(with 95% Confidence Intervals)', fontsize=14)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure4_coefficient_comparison.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure4_coefficient_comparison.png/pdf")

# ============================================================================
# Figure 5: Subgroup Analysis
# ============================================================================
print("Creating Figure 5: Subgroup Analysis...")

subgroup_data = [
    ('Full Sample', 0.0748, 0.0181),
    ('Males', 0.0716, 0.0199),
    ('Females', 0.0527, 0.0281),
]

fig, ax = plt.subplots(figsize=(8, 5))

names = [s[0] for s in subgroup_data]
estimates = [s[1] for s in subgroup_data]
ses = [s[2] for s in subgroup_data]

x_pos = np.arange(len(names))
ci_lower = [e - 1.96*s for e, s in zip(estimates, ses)]
ci_upper = [e + 1.96*s for e, s in zip(estimates, ses)]

colors = ['#2ca02c', '#1f77b4', '#d62728']
ax.barh(x_pos, estimates, xerr=[np.array(estimates)-np.array(ci_lower),
                                  np.array(ci_upper)-np.array(estimates)],
        color=colors, edgecolor='black', capsize=5, height=0.6)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for i, (est, ci_u) in enumerate(zip(estimates, ci_upper)):
    ax.text(ci_u + 0.005, i, f'{est:.3f}', va='center', fontsize=11, fontweight='bold')

ax.set_yticks(x_pos)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('DiD Estimate (Effect on FT Employment)', fontsize=12)
ax.set_title('Subgroup Analysis by Sex\n(with 95% Confidence Intervals)', fontsize=14)
ax.set_xlim(-0.05, 0.15)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_subgroup_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure5_subgroup_analysis.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure5_subgroup_analysis.png/pdf")

print("\nAll figures created successfully!")

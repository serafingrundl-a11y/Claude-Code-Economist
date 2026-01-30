"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set up output directory
output_dir = r'C:\Users\seraf\DACA Results Task 3\replication_09'

# Load data
df = pd.read_csv(f'{output_dir}/data/prepared_data_numeric_version.csv', low_memory=False)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# =============================================================================
# Figure 1: Parallel Trends Plot
# =============================================================================
print("Creating Figure 1: Parallel Trends...")

yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_means.columns = ['Control (Ages 31-35)', 'Treated (Ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(yearly_means.index, yearly_means['Control (Ages 31-35)'],
        'o-', color='#2ecc71', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')
ax.plot(yearly_means.index, yearly_means['Treated (Ages 26-30)'],
        's-', color='#e74c3c', linewidth=2, markersize=8, label='Treated (Ages 26-30 in 2012)')

# Add vertical line for treatment
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.72), fontsize=10, ha='center',
            color='gray')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-treatment period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='orange', label='Post-treatment period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Group Over Time')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.55, 0.75)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure1_parallel_trends.png/pdf")

# =============================================================================
# Figure 2: DiD Illustration
# =============================================================================
print("Creating Figure 2: DiD Illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate means
control_pre = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_post = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treated_post = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

# Counterfactual
counterfactual = treated_pre + (control_post - control_pre)

# Plot
x_pre, x_post = 0, 1
width = 0.15

# Control group
ax.plot([x_pre, x_post], [control_pre, control_post], 'o-', color='#2ecc71',
        linewidth=2.5, markersize=12, label='Control Group')

# Treated group
ax.plot([x_pre, x_post], [treated_pre, treated_post], 's-', color='#e74c3c',
        linewidth=2.5, markersize=12, label='Treated Group')

# Counterfactual
ax.plot([x_pre, x_post], [treated_pre, counterfactual], 's--', color='#e74c3c',
        linewidth=1.5, markersize=8, alpha=0.5, label='Counterfactual (Treated)')

# Treatment effect arrow
did = treated_post - counterfactual
ax.annotate('', xy=(x_post+0.02, treated_post), xytext=(x_post+0.02, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(x_post+0.08, (treated_post + counterfactual)/2, f'DiD Effect\n= {did:.3f}',
        fontsize=11, va='center', fontweight='bold')

ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.58, 0.72)
ax.set_xticks([x_pre, x_post])
ax.set_xticklabels(['Pre-Treatment\n(2008-2011)', 'Post-Treatment\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Estimation')
ax.legend(loc='lower right')

# Add annotations for values
ax.annotate(f'{control_pre:.3f}', xy=(x_pre, control_pre), xytext=(-0.15, control_pre),
            fontsize=10, color='#2ecc71')
ax.annotate(f'{control_post:.3f}', xy=(x_post, control_post), xytext=(x_post-0.12, control_post-0.015),
            fontsize=10, color='#2ecc71')
ax.annotate(f'{treated_pre:.3f}', xy=(x_pre, treated_pre), xytext=(-0.15, treated_pre),
            fontsize=10, color='#e74c3c')
ax.annotate(f'{treated_post:.3f}', xy=(x_post, treated_post), xytext=(x_post-0.12, treated_post+0.01),
            fontsize=10, color='#e74c3c')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure2_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure2_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure2_did_illustration.png/pdf")

# =============================================================================
# Figure 3: Event Study
# =============================================================================
print("Creating Figure 3: Event Study...")

event_df = pd.read_csv(f'{output_dir}/event_study.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Reference year (2011) is 0
years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ses = event_df['SE'].values

# Plot coefficients with confidence intervals
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o', color='#3498db',
            markersize=10, capsize=5, capthick=2, linewidth=2, elinewidth=2)

# Connect with line
ax.plot(years, coefs, '-', color='#3498db', linewidth=1, alpha=0.5)

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Treatment line
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA', xy=(2012, 0.08), fontsize=10, ha='center', color='gray')

# Shade periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='orange')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Treatment Effects by Year')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add note for reference year
ax.annotate('Reference\nYear', xy=(2011, 0), xytext=(2011, -0.05),
            fontsize=9, ha='center', va='top',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))

plt.tight_layout()
plt.savefig(f'{output_dir}/figure3_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure3_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure3_event_study.png/pdf")

# =============================================================================
# Figure 4: Coefficient Plot
# =============================================================================
print("Creating Figure 4: Coefficient Plot...")

results_df = pd.read_csv(f'{output_dir}/results_table.csv')

fig, ax = plt.subplots(figsize=(10, 7))

models = results_df['Model'].values
coefs = results_df['Coefficient'].values
ci_lower = results_df['95% CI Lower'].values
ci_upper = results_df['95% CI Upper'].values

y_pos = np.arange(len(models))[::-1]

# Plot
ax.errorbar(coefs, y_pos, xerr=[coefs-ci_lower, ci_upper-coefs],
            fmt='o', color='#e74c3c', markersize=10, capsize=5,
            capthick=2, linewidth=2, elinewidth=2)

# Reference line at 0
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)')
ax.set_title('DACA Effect Estimates Across Model Specifications')

# Add coefficient values
for i, (c, lower, upper) in enumerate(zip(coefs, ci_lower, ci_upper)):
    ax.annotate(f'{c:.3f}\n[{lower:.3f}, {upper:.3f}]',
                xy=(c, y_pos[i]), xytext=(c+0.02, y_pos[i]),
                fontsize=9, va='center')

ax.set_xlim(-0.02, 0.15)

plt.tight_layout()
plt.savefig(f'{output_dir}/figure4_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure4_coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure4_coefficient_plot.png/pdf")

# =============================================================================
# Figure 5: Distribution of FT by Group
# =============================================================================
print("Creating Figure 5: FT Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

groups = [
    ((df['ELIGIBLE']==0) & (df['AFTER']==0), 'Control - Pre (2008-2011)'),
    ((df['ELIGIBLE']==0) & (df['AFTER']==1), 'Control - Post (2013-2016)'),
    ((df['ELIGIBLE']==1) & (df['AFTER']==0), 'Treated - Pre (2008-2011)'),
    ((df['ELIGIBLE']==1) & (df['AFTER']==1), 'Treated - Post (2013-2016)'),
]

colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']

for ax, (mask, title), color in zip(axes.flat, groups, colors):
    data = df[mask]['FT']
    counts = data.value_counts().sort_index()
    bars = ax.bar([0, 1], [counts.get(0, 0), counts.get(1, 0)], color=color, alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Not Full-Time', 'Full-Time'])
    ax.set_ylabel('Count')
    ax.set_title(f'{title}\n(Mean FT = {data.mean():.3f}, n = {len(data):,})')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.suptitle('Distribution of Full-Time Employment by Group and Period', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/figure5_ft_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure5_ft_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure5_ft_distribution.png/pdf")

# =============================================================================
# Figure 6: Sample Composition by Year
# =============================================================================
print("Creating Figure 6: Sample Composition...")

fig, ax = plt.subplots(figsize=(10, 6))

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control (31-35)', 'Treated (26-30)']

sample_by_year.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Number of Observations')
ax.set_title('Sample Size by Year and Treatment Group')
ax.legend(title='Group')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# Add vertical separator
ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('Pre-Treatment', xy=(1.5, ax.get_ylim()[1]*0.95), fontsize=10, ha='center')
ax.annotate('Post-Treatment', xy=(5.5, ax.get_ylim()[1]*0.95), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure6_sample_composition.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{output_dir}/figure6_sample_composition.pdf', bbox_inches='tight')
plt.close()
print("  Saved figure6_sample_composition.png/pdf")

print("\nAll figures created successfully!")

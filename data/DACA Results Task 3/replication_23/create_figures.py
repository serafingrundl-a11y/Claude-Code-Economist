"""
DACA Replication Study - Create Figures
========================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 6)

print("Creating figures for replication report...")

# ==============================================================================
# Load Data
# ==============================================================================
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Define weighted mean function
def weighted_mean(x, w):
    return np.average(x, weights=w)

# ==============================================================================
# Figure 1: Full-Time Employment Trends by Year (Parallel Trends)
# ==============================================================================
print("Creating Figure 1: Parallel Trends...")

# Calculate weighted means by year and group
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda g: weighted_mean(g['FT'], g['PERWT'])
).unstack()
trends.columns = ['Control (31-35)', 'Treatment (26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

years = trends.index.values
ax.plot(years, trends['Control (31-35)'], 'o-', color='#1f77b4',
        linewidth=2, markersize=8, label='Control (31-35)')
ax.plot(years, trends['Treatment (26-30)'], 's-', color='#d62728',
        linewidth=2, markersize=8, label='Treatment (26-30)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA (2012)')

# Add shaded region for post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (Weighted)')
ax.set_title('Full-Time Employment Trends by Treatment Status')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim([0.55, 0.75])
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# ==============================================================================
# Figure 2: DiD Visualization (2x2 Plot)
# ==============================================================================
print("Creating Figure 2: DiD Visualization...")

# Calculate group means by period
did_data = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda g: weighted_mean(g['FT'], g['PERWT'])
).unstack()
did_data.index = ['Control', 'Treatment']
did_data.columns = ['Pre-DACA', 'Post-DACA']

fig, ax = plt.subplots(figsize=(8, 6))

x = np.array([0, 1])  # Pre, Post
width = 0.35

control_bars = ax.bar(x - width/2, did_data.loc['Control'], width,
                       label='Control (31-35)', color='#1f77b4', alpha=0.8)
treatment_bars = ax.bar(x + width/2, did_data.loc['Treatment'], width,
                         label='Treatment (26-30)', color='#d62728', alpha=0.8)

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Full-Time Employment')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend()
ax.set_ylim([0.5, 0.8])

# Add value labels on bars
for bars in [control_bars, treatment_bars]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_bars.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_did_bars.png/pdf")

# ==============================================================================
# Figure 3: Coefficient Plot
# ==============================================================================
print("Creating Figure 3: Coefficient Plot...")

# Read model results
results = pd.read_csv('model_results_summary.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results))
colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#d62728', '#2ca02c']

ax.errorbar(results['DiD Estimate'], y_pos,
            xerr=1.96*results['SE'],
            fmt='o', capsize=5, capthick=2, markersize=8,
            color='black', ecolor='gray', linewidth=2)

# Add vertical line at 0
ax.axvline(x=0, color='red', linestyle='--', linewidth=1, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(results['Model'])
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment)')
ax.set_title('DACA Effect Estimates Across Specifications')

# Invert y-axis so first model is on top
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figure3_coef_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coef_plot.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_coef_plot.png/pdf")

# ==============================================================================
# Figure 4: Sample Distribution by Year and Group
# ==============================================================================
print("Creating Figure 4: Sample Distribution...")

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(sample_by_year))
width = 0.35

ax.bar(x - width/2, sample_by_year['Control (31-35)'], width,
       label='Control (31-35)', color='#1f77b4', alpha=0.8)
ax.bar(x + width/2, sample_by_year['Treatment (26-30)'], width,
       label='Treatment (26-30)', color='#d62728', alpha=0.8)

ax.set_ylabel('Number of Observations')
ax.set_xlabel('Year')
ax.set_title('Sample Size by Year and Treatment Group')
ax.set_xticks(x)
ax.set_xticklabels(sample_by_year.index)
ax.legend()

# Add vertical line for DACA
ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=2, label='DACA')

plt.tight_layout()
plt.savefig('figure4_sample_dist.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_dist.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_sample_dist.png/pdf")

# ==============================================================================
# Figure 5: Event Study Style Plot
# ==============================================================================
print("Creating Figure 5: Event Study Plot...")

# Calculate difference (Treatment - Control) by year
diff_by_year = trends['Treatment (26-30)'] - trends['Control (31-35)']

# Calculate SE for difference (approximate)
se_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].agg(['std', 'count']).unstack()
se_diff = np.sqrt(
    (se_by_year[('std', 0)]**2 / se_by_year[('count', 0)]) +
    (se_by_year[('std', 1)]**2 / se_by_year[('count', 1)])
)

fig, ax = plt.subplots(figsize=(10, 6))

# Normalize to 2011 (last pre-treatment year)
diff_normalized = diff_by_year - diff_by_year[2011]

ax.errorbar(diff_by_year.index, diff_normalized, yerr=1.96*se_diff.values,
            fmt='o-', capsize=5, capthick=2, markersize=8,
            color='#1f77b4', linewidth=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)

# Shade post-treatment
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Difference (Treatment - Control)\nNormalized to 2011')
ax.set_title('Event Study: Treatment-Control Gap Over Time')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure5_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_event_study.png/pdf")

# ==============================================================================
# Figure 6: Demographics Comparison
# ==============================================================================
print("Creating Figure 6: Demographics Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sex distribution
sex_data = df.groupby(['ELIGIBLE', 'SEX']).apply(
    lambda g: weighted_mean(np.ones(len(g)), g['PERWT'])
)
sex_weighted = df.groupby('ELIGIBLE').apply(
    lambda g: pd.Series({
        'Male': weighted_mean((g['SEX']==1).astype(int), g['PERWT']),
        'Female': weighted_mean((g['SEX']==2).astype(int), g['PERWT'])
    })
)

x = np.arange(2)
width = 0.35

axes[0].bar(x - width/2, sex_weighted.loc[0], width, label='Control', color='#1f77b4')
axes[0].bar(x + width/2, sex_weighted.loc[1], width, label='Treatment', color='#d62728')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Male', 'Female'])
axes[0].set_ylabel('Proportion')
axes[0].set_title('Sex Distribution by Group')
axes[0].legend()

# Education distribution
educ_weighted = df.groupby(['ELIGIBLE', 'EDUC_RECODE']).apply(
    lambda g: g['PERWT'].sum()
).unstack(fill_value=0)
educ_weighted = educ_weighted.div(educ_weighted.sum(axis=1), axis=0)

educ_cats = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_weighted = educ_weighted[[c for c in educ_cats if c in educ_weighted.columns]]

x = np.arange(len(educ_weighted.columns))
width = 0.35

axes[1].bar(x - width/2, educ_weighted.loc[0], width, label='Control', color='#1f77b4')
axes[1].bar(x + width/2, educ_weighted.loc[1], width, label='Treatment', color='#d62728')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['<HS', 'HS', 'Some Coll.', '2-Year', 'BA+'], rotation=0)
axes[1].set_ylabel('Proportion')
axes[1].set_title('Education Distribution by Group')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure6_demographics.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_demographics.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure6_demographics.png/pdf")

print("\nAll figures created successfully!")

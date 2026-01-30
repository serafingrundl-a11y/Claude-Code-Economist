"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
trends = pd.read_csv('trends_data.csv')
event_study = pd.read_csv('event_study_results.csv')

# =============================================================================
# Figure 1: Parallel Trends
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Pivot data for plotting
trends_pivot = trends.pivot(index='YEAR', columns='daca_eligible', values='fulltime_rate')

# Plot treatment and control groups
ax.plot(trends_pivot.index, trends_pivot[1], 'o-', color='#2166AC', linewidth=2,
        markersize=8, label='DACA Eligible (Treatment)')
ax.plot(trends_pivot.index, trends_pivot[0], 's--', color='#B2182B', linewidth=2,
        markersize=8, label='Age-Ineligible (Control)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.72, 'DACA Implemented\n(June 2012)', fontsize=10, color='gray')

# Add shading for pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray', label='Pre-DACA Period')
ax.axvspan(2012.5, 2016, alpha=0.1, color='blue', label='Post-DACA Period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Figure 1: Trends in Full-Time Employment by DACA Eligibility Status')
ax.legend(loc='lower right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.35, 0.80)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_parallel_trends.png/pdf")

# =============================================================================
# Figure 2: Event Study
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Sort by year
event_study = event_study.sort_values('year')

# Plot coefficients with confidence intervals
years = event_study['year'].values
coefs = event_study['coefficient'].values
ci_lower = event_study['ci_lower'].values
ci_upper = event_study['ci_upper'].values

# Calculate error bars
errors = np.array([coefs - ci_lower, ci_upper - coefs])

# Plot
ax.errorbar(years, coefs, yerr=errors, fmt='o', color='#2166AC',
            markersize=8, capsize=4, capthick=2, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='gray')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Effect on Full-Time Employment)')
ax.set_title('Figure 2: Event Study - Year-by-Year Treatment Effects\n(Reference Year: 2011)')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
ax.legend()

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_event_study.png/pdf")

# =============================================================================
# Figure 3: DiD Illustration
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Pre and post means
summary = pd.read_csv('analysis_summary.csv')
pre_treat = summary['Pre_Treat_FT_Rate'].values[0]
post_treat = summary['Post_Treat_FT_Rate'].values[0]
pre_control = summary['Pre_Control_FT_Rate'].values[0]
post_control = summary['Post_Control_FT_Rate'].values[0]

# Plot points
ax.scatter([0], [pre_treat], s=150, color='#2166AC', zorder=5, label='Treatment')
ax.scatter([1], [post_treat], s=150, color='#2166AC', zorder=5)
ax.scatter([0], [pre_control], s=150, color='#B2182B', marker='s', zorder=5, label='Control')
ax.scatter([1], [post_control], s=150, color='#B2182B', marker='s', zorder=5)

# Draw lines
ax.plot([0, 1], [pre_treat, post_treat], '-', color='#2166AC', linewidth=2)
ax.plot([0, 1], [pre_control, post_control], '--', color='#B2182B', linewidth=2)

# Counterfactual line (what would have happened to treatment without DACA)
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], ':', color='#2166AC', linewidth=2, alpha=0.5)
ax.scatter([1], [counterfactual], s=100, color='#2166AC', marker='o', alpha=0.5, facecolors='none', linewidths=2)

# Add labels
ax.annotate('Treatment\n(Observed)', (1.05, post_treat), fontsize=10, color='#2166AC')
ax.annotate('Treatment\n(Counterfactual)', (1.05, counterfactual), fontsize=10, color='#2166AC', alpha=0.7)
ax.annotate('Control', (1.05, post_control), fontsize=10, color='#B2182B')

# DiD effect arrow
mid_y = (post_treat + counterfactual) / 2
ax.annotate('', xy=(0.9, post_treat), xytext=(0.9, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(0.85, mid_y, f'DiD\n={post_treat-counterfactual:.3f}', fontsize=10, color='green', ha='right')

ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.35, 0.75)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Figure 3: Difference-in-Differences Illustration')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_did_illustration.png/pdf")

# =============================================================================
# Figure 4: Sample Size by Year
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

sample_by_year = trends.groupby('YEAR')['n'].sum()

# Create bar chart with different colors for pre/post
colors = ['#B2182B' if y < 2012 else ('#808080' if y == 2012 else '#2166AC')
          for y in sample_by_year.index]

bars = ax.bar(sample_by_year.index, sample_by_year.values, color=colors, edgecolor='black')

ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Figure 4: Analysis Sample Size by Year')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#B2182B', edgecolor='black', label='Pre-DACA (2006-2011)'),
                   Patch(facecolor='#2166AC', edgecolor='black', label='Post-DACA (2013-2016)')]
ax.legend(handles=legend_elements, loc='upper right')

# Add sample size labels on bars
for bar, val in zip(bars, sample_by_year.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_size.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved: figure4_sample_size.png/pdf")

print("\nAll figures created successfully!")

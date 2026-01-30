"""
DACA Replication Study - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# =============================================================================
# Figure 1: Parallel Trends
# =============================================================================
print("Creating Figure 1: Parallel Trends...")

trends = pd.read_csv('trends_by_year.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment group
treat = trends[trends['treated'] == 1]
control = trends[trends['treated'] == 0]

ax.plot(treat['year'], treat['fulltime_rate'], 'o-', color='#2166ac',
        linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(control['year'], control['fulltime_rate'], 's--', color='#b2182b',
        linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.6, 0.545, 'DACA\nImplemented', fontsize=10, color='gray', va='bottom')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Treatment Status')
ax.legend(loc='lower right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.50, 0.65)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# =============================================================================
# Figure 2: Event Study
# =============================================================================
print("Creating Figure 2: Event Study...")

event = pd.read_csv('event_study_results.csv')
event = event.sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(event['year'], event['coef'],
            yerr=[event['coef'] - event['ci_lower'], event['ci_upper'] - event['coef']],
            fmt='o', capsize=4, capthick=2, color='#2166ac',
            markersize=8, linewidth=2, elinewidth=2)

# Connect with line
ax.plot(event['year'], event['coef'], '-', color='#2166ac', linewidth=1, alpha=0.5)

# Add horizontal line at 0
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.08, 'DACA\n(June 2012)', fontsize=10, color='red', va='bottom')

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (relative to 2011)')
ax.set_title('Event Study: Year-by-Year Treatment Effects on Full-Time Employment')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# =============================================================================
# Figure 3: DiD Visualization (2x2 means)
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(8, 6))

# Data for 2x2 DiD
pre_treat = 0.5413
post_treat = 0.5892
pre_control = 0.5805
post_control = 0.5745

# Plot
x = [0, 1]
ax.plot(x, [pre_treat, post_treat], 'o-', color='#2166ac', linewidth=3,
        markersize=12, label='Treatment (Ages 26-30)')
ax.plot(x, [pre_control, post_control], 's--', color='#b2182b', linewidth=3,
        markersize=12, label='Control (Ages 31-35)')

# Add counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], ':', color='#2166ac', linewidth=2,
        alpha=0.5, label='Counterfactual')

# Add annotation for treatment effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=10, va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-Time Employment')
ax.legend(loc='lower right')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.50, 0.62)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

print("\nAll figures created successfully!")

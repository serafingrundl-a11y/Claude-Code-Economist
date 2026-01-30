"""
Create Figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load data
year_group_means = pd.read_csv('year_group_means.csv')
event_study = pd.read_csv('event_study_results.csv')
regression_results = pd.read_csv('regression_results.csv')

# =============================================================================
# Figure 1: Trends in Full-Time Employment
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Separate treatment and control
treat_data = year_group_means[year_group_means['treatment_group'] == 1].sort_values('YEAR')
control_data = year_group_means[year_group_means['treatment_group'] == 0].sort_values('YEAR')

# Plot trends
ax.plot(treat_data['YEAR'], treat_data['fulltime_rate'], 'o-',
        label='Treatment (Ages 26-30)', color='#2166AC', linewidth=2, markersize=8)
ax.plot(control_data['YEAR'], control_data['fulltime_rate'], 's--',
        label='Control (Ages 31-35)', color='#B2182B', linewidth=2, markersize=8)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.98, 'DACA\nImplemented', fontsize=9,
        verticalalignment='top', color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Trends in Full-Time Employment by Treatment Status\n(DACA-Eligible Hispanic-Mexican Immigrants)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim([0.55, 0.75])

plt.tight_layout()
plt.savefig('figure1_trends.png', bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: Trends in Full-Time Employment")

# =============================================================================
# Figure 2: Event Study
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

event_study_sorted = event_study.sort_values('year')

# Plot coefficients with confidence intervals
ax.errorbar(event_study_sorted['year'], event_study_sorted['coef'],
            yerr=[event_study_sorted['coef'] - event_study_sorted['ci_low'],
                  event_study_sorted['ci_high'] - event_study_sorted['coef']],
            fmt='o', capsize=5, capthick=2, color='#2166AC', linewidth=2,
            markersize=10, markerfacecolor='white', markeredgewidth=2)

# Connect points with line
ax.plot(event_study_sorted['year'], event_study_sorted['coef'], '-',
        color='#2166AC', linewidth=1, alpha=0.5)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.9, 'DACA\nImplemented', fontsize=9,
        verticalalignment='top', color='gray')

# Shade post-treatment period
ax.axvspan(2012, 2017, alpha=0.1, color='green')
ax.axvspan(2005, 2012, alpha=0.1, color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure2_event_study.png', bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: Event Study")

# =============================================================================
# Figure 3: Coefficient Comparison Across Specifications
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Exclude placebo test for main comparison
main_specs = regression_results[regression_results['Specification'] != 'Placebo Test']

y_pos = np.arange(len(main_specs))
coeffs = main_specs['Coefficient'].values
errors = main_specs['Std_Error'].values * 1.96  # 95% CI

# Create horizontal bar chart with error bars
bars = ax.barh(y_pos, coeffs, xerr=errors, capsize=5, color='#2166AC',
               alpha=0.7, edgecolor='black', linewidth=1)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(main_specs['Specification'].values)
ax.set_xlabel('DiD Coefficient (Change in Full-Time Employment Probability)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications\n(with 95% Confidence Intervals)', fontsize=14)

# Add annotation for preferred estimate
ax.annotate('Preferred', xy=(0.042, 2), xytext=(0.08, 2.5),
            fontsize=10, color='red',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()
plt.savefig('figure3_coefficient_comparison.png', bbox_inches='tight')
plt.savefig('figure3_coefficient_comparison.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: Coefficient Comparison")

# =============================================================================
# Figure 4: DiD Visual Representation
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Get means for plotting
treat_pre = year_group_means[(year_group_means['treatment_group']==1) &
                              (year_group_means['YEAR'] <= 2011)]['fulltime_rate'].mean()
treat_post = year_group_means[(year_group_means['treatment_group']==1) &
                               (year_group_means['YEAR'] >= 2013)]['fulltime_rate'].mean()
control_pre = year_group_means[(year_group_means['treatment_group']==0) &
                                (year_group_means['YEAR'] <= 2011)]['fulltime_rate'].mean()
control_post = year_group_means[(year_group_means['treatment_group']==0) &
                                 (year_group_means['YEAR'] >= 2013)]['fulltime_rate'].mean()

# Plot the 2x2 DiD
x = [0, 1]
ax.plot(x, [treat_pre, treat_post], 'o-', color='#2166AC', linewidth=3,
        markersize=12, label='Treatment (Ages 26-30)')
ax.plot(x, [control_pre, control_post], 's--', color='#B2182B', linewidth=3,
        markersize=12, label='Control (Ages 31-35)')

# Add counterfactual line
counterfactual = treat_pre + (control_post - control_pre)
ax.plot([0, 1], [treat_pre, counterfactual], ':', color='#2166AC', linewidth=2,
        alpha=0.5, label='Counterfactual (Treatment)')

# Add arrow showing treatment effect
ax.annotate('', xy=(1.05, treat_post), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (treat_post + counterfactual)/2, f'DiD Effect:\n{treat_post-counterfactual:.3f}',
        fontsize=10, verticalalignment='center', color='green')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Visual Representation\nof DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim([-0.2, 1.5])

plt.tight_layout()
plt.savefig('figure4_did_visual.png', bbox_inches='tight')
plt.savefig('figure4_did_visual.pdf', bbox_inches='tight')
plt.close()

print("Figure 4 saved: DiD Visual Representation")

print("\nAll figures created successfully!")

"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# =============================================================================
# FIGURE 1: Parallel Trends / Event Study
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_study = results['event_study']

# Extract coefficients and SEs
coefs = []
ses = []
for yr in years:
    if str(yr) in event_study:
        coefs.append(event_study[str(yr)]['coef'])
        ses.append(event_study[str(yr)]['se'])
    else:  # 2011 is reference
        coefs.append(0)
        ses.append(0)

# Create error bars (95% CI = 1.96 * SE)
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

# Plot
ax.errorbar(years, coefs, yerr=[np.array(coefs)-np.array(ci_lower),
                                 np.array(ci_upper)-np.array(coefs)],
            fmt='o-', color='navy', capsize=4, markersize=8, linewidth=2)

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')

# Fill confidence interval
ax.fill_between(years, ci_lower, ci_upper, alpha=0.2, color='navy')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)')
ax.set_xticks(years)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: Event Study")

# =============================================================================
# FIGURE 2: Full-Time Employment Trends by Group
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

year_rates = results['year_rates']
years_full = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

eligible_rates = [year_rates['Eligible'][str(yr)] for yr in years_full]
not_eligible_rates = [year_rates['Not Eligible'][str(yr)] for yr in years_full]

ax.plot(years_full, eligible_rates, 'o-', color='darkgreen', markersize=8,
        linewidth=2, label='DACA Eligible')
ax.plot(years_full, not_eligible_rates, 's-', color='darkblue', markersize=8,
        linewidth=2, label='Not DACA Eligible')

# Add vertical line for DACA
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5,
           label='DACA Implementation (June 2012)')

# Add recession shading (Dec 2007 - June 2009)
ax.axvspan(2008, 2009.5, alpha=0.15, color='gray', label='Great Recession')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.set_xticks(years_full)
ax.set_ylim(0.35, 0.65)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: Trends by Group")

# =============================================================================
# FIGURE 3: DiD Visual
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Pre and post means
ft_rates = results['ft_rates']
pre_eligible = ft_rates['Pre-DACA']['Eligible']
post_eligible = ft_rates['Post-DACA']['Eligible']
pre_not_eligible = ft_rates['Pre-DACA']['Not Eligible']
post_not_eligible = ft_rates['Post-DACA']['Not Eligible']

# Plot bars
x = np.array([0, 1, 3, 4])
heights = [pre_not_eligible, post_not_eligible, pre_eligible, post_eligible]
colors = ['lightblue', 'darkblue', 'lightgreen', 'darkgreen']
labels = ['Control Pre', 'Control Post', 'Treated Pre', 'Treated Post']

bars = ax.bar(x, heights, color=colors, edgecolor='black', linewidth=1.2)

# Add labels
ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Not DACA Eligible\n(Control)', 'DACA Eligible\n(Treatment)'])

# Add value labels on bars
for bar, val in zip(bars, heights):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='lightblue', edgecolor='black', label='Pre-DACA (2006-2011)'),
                   Patch(facecolor='darkblue', edgecolor='black', label='Post-DACA (2013-2016)')]
ax.legend(handles=legend_elements, loc='upper right')

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Full-Time Employment Rates')
ax.set_ylim(0, 0.7)

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: DiD Visualization")

# =============================================================================
# FIGURE 4: Coefficient Comparison
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Extract main results
main_results = results['main_results']
models = ['Model 1\n(Basic)', 'Model 2\n(+Controls)', 'Model 3\n(+Year FE)',
          'Model 4\n(+State FE)', 'Weighted']
model_keys = ['model1', 'model2', 'model3', 'model4', 'weighted']

coefs = [main_results[k]['coef'] for k in model_keys]
ses = [main_results[k]['se'] for k in model_keys]

ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

x_pos = np.arange(len(models))

# Plot
ax.errorbar(x_pos, coefs, yerr=[np.array(coefs)-np.array(ci_lower),
                                 np.array(ci_upper)-np.array(coefs)],
            fmt='o', color='navy', capsize=6, markersize=10, capthick=2)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Add coefficient values
for i, (c, ci_l, ci_u) in enumerate(zip(coefs, ci_lower, ci_upper)):
    ax.text(i, ci_u + 0.003, f'{c:.4f}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('DiD Coefficient')
ax.set_title('Difference-in-Differences Estimates Across Specifications\n(with 95% Confidence Intervals)')
ax.set_ylim(-0.01, 0.08)

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficients.pdf', bbox_inches='tight')
plt.close()

print("Figure 4 saved: Coefficient Comparison")

print("\nAll figures created successfully!")

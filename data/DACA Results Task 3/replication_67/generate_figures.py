"""
Generate figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (8, 5)

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Load yearly rates
yearly_rates = pd.read_csv('yearly_ft_rates.csv', index_col=0)

# =============================================================================
# FIGURE 1: Parallel Trends
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treatment_rates = yearly_rates['Treatment (26-30)'].values
control_rates = yearly_rates['Control (31-35)'].values

ax.plot(years, treatment_rates, 'o-', color='#2E86AB', linewidth=2, markersize=8,
        label='Treatment (Ages 26-30)')
ax.plot(years, control_rates, 's--', color='#A23B72', linewidth=2, markersize=8,
        label='Control (Ages 31-35)')

# Add vertical line at treatment
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.73), fontsize=10, ha='center', color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status')
ax.legend(loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.80)

# Add shading for pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA')

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_parallel_trends.png/pdf")

# =============================================================================
# FIGURE 2: Event Study
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Event study coefficients (relative to 2011)
event_years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
event_coefs = [results[f'event_{y}_coef'] for y in event_years]
event_ses = [results[f'event_{y}_se'] for y in event_years]

# Add 2011 as reference (coefficient = 0)
all_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
all_coefs = event_coefs[:3] + [0] + event_coefs[3:]
all_ses = event_ses[:3] + [0] + event_ses[3:]

# Calculate confidence intervals
ci_low = [c - 1.96*s for c, s in zip(all_coefs, all_ses)]
ci_high = [c + 1.96*s for c, s in zip(all_coefs, all_ses)]

# Plot
ax.errorbar(all_years, all_coefs, yerr=[np.array(all_coefs)-np.array(ci_low),
                                          np.array(ci_high)-np.array(all_coefs)],
            fmt='o', color='#2E86AB', capsize=5, capthick=2, linewidth=2, markersize=8)

# Connect points with lines
ax.plot(all_years, all_coefs, '-', color='#2E86AB', alpha=0.5)

# Reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Annotations
ax.annotate('DACA\nImplemented', xy=(2012, 0.08), fontsize=10, ha='center', color='gray')
ax.annotate('Reference Year', xy=(2011, 0.005), fontsize=9, ha='center',
            xytext=(2011, 0.025), arrowprops=dict(arrowstyle='->', color='gray'))

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Treatment Effect on Full-Time Employment')
ax.set_xticks(all_years)
ax.set_ylim(-0.15, 0.15)

# Shading
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_event_study.png/pdf")

# =============================================================================
# FIGURE 3: DiD Visual
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Data for 2x2 DiD
periods = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
treatment_means = [results['ft_rate_treat_pre'], results['ft_rate_treat_post']]
control_means = [results['ft_rate_ctrl_pre'], results['ft_rate_ctrl_post']]

x = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x - width/2, treatment_means, width, label='Treatment (Ages 26-30)',
               color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, control_means, width, label='Control (Ages 31-35)',
               color='#A23B72', alpha=0.8)

# Add value labels on bars
for bar, val in zip(bars1, treatment_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, control_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Full-Time Employment')
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend(loc='upper left')
ax.set_ylim(0, 0.85)

# Add DiD annotation
ax.annotate(f'DiD Estimate: {results["model1_coef"]:.3f}***\n(SE: {results["model1_se"]:.3f})',
            xy=(0.95, 0.15), xycoords='axes fraction', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            ha='right')

plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visual.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_did_visual.png/pdf")

# =============================================================================
# FIGURE 4: Coefficient Comparison Across Models
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Model 1:\nBasic DiD', 'Model 2:\n+ Demographics', 'Model 3:\n+ State/Year FE']
coefs = [results['model1_coef'], results['model2_coef'], results['model3_coef']]
ses = [results['model1_se'], results['model2_se'], results['model3_se']]
ci_lows = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_highs = [c + 1.96*s for c, s in zip(coefs, ses)]

x = np.arange(len(models))

ax.errorbar(x, coefs, yerr=[np.array(coefs)-np.array(ci_lows),
                             np.array(ci_highs)-np.array(coefs)],
            fmt='o', color='#2E86AB', capsize=8, capthick=2, linewidth=2, markersize=12)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient')
ax.set_title('DACA Effect on Full-Time Employment Across Specifications')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(-0.02, 0.14)

# Add coefficient labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.annotate(f'{c:.3f}\n({s:.3f})', xy=(i, c+0.02), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficient_comparison.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_coefficient_comparison.png/pdf")

print("\nAll figures generated successfully!")

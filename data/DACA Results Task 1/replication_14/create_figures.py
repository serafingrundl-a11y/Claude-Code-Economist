"""
Create figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# =============================================================================
# Figure 1: Trends in Full-Time Employment
# =============================================================================
print("Creating Figure 1: Employment Trends...")
trends = pd.read_csv('trends_data.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Eligible group
eligible = trends[trends['eligible'] == 1]
control = trends[trends['eligible'] == 0]

ax.plot(eligible['YEAR'], eligible['fulltime_rate'], 'o-', color='#2E86AB',
        linewidth=2, markersize=8, label='DACA-Eligible')
ax.plot(control['YEAR'], control['fulltime_rate'], 's--', color='#E94F37',
        linewidth=2, markersize=8, label='Non-Eligible')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.95, 'DACA\nImplemented', fontsize=10,
        ha='left', va='top', color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.3, 0.7)

plt.tight_layout()
plt.savefig('figure1_trends.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.pdf")

# =============================================================================
# Figure 2: Event Study
# =============================================================================
print("Creating Figure 2: Event Study...")
event = pd.read_csv('event_study_data.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(event['year'], event['coef'],
            yerr=[event['coef'] - event['ci_low'], event['ci_high'] - event['coef']],
            fmt='o-', color='#2E86AB', linewidth=2, markersize=8, capsize=4, capthick=2)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.9, 'DACA\nImplemented', fontsize=10,
        ha='left', va='top', color='gray')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure2_event_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.pdf")

# =============================================================================
# Figure 3: Coefficient Comparison Across Models
# =============================================================================
print("Creating Figure 3: Model Comparison...")
results = pd.read_csv('results_summary.csv')

fig, ax = plt.subplots(figsize=(8, 5))

models = results['Model']
coefs = results['Coefficient']
ses = results['SE']

colors = ['#E94F37', '#2E86AB', '#4CAF50', '#9C27B0']
y_pos = np.arange(len(models))

# Plot bars
bars = ax.barh(y_pos, coefs, xerr=1.96*ses, color=colors, alpha=0.8, capsize=5)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect on Full-Time Employment Across Model Specifications', fontsize=14)

# Add value labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.text(c + 1.96*s + 0.005, i, f'{c:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_models.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure3_models.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_models.pdf")

# =============================================================================
# Figure 4: Heterogeneity Analysis
# =============================================================================
print("Creating Figure 4: Heterogeneity Analysis...")
het = pd.read_csv('heterogeneity_results.csv')

fig, ax = plt.subplots(figsize=(8, 5))

subgroups = het['Subgroup']
coefs = het['Coefficient']
ses = het['SE']

y_pos = np.arange(len(subgroups))
colors = ['#2E86AB', '#E94F37', '#4CAF50', '#9C27B0']

# Plot with error bars
ax.errorbar(coefs, y_pos, xerr=1.96*ses, fmt='o', color='#2E86AB',
            markersize=10, capsize=5, capthick=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Coefficient', fontsize=12)
ax.set_title('Heterogeneous Effects of DACA on Full-Time Employment', fontsize=14)

plt.tight_layout()
plt.savefig('figure4_heterogeneity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure4_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure4_heterogeneity.pdf")

print("\nAll figures created successfully!")

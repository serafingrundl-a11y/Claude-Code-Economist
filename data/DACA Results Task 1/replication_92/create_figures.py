"""
Create figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

# ============================================================================
# Figure 1: Trends in Full-Time Employment by DACA Eligibility
# ============================================================================
print("Creating Figure 1: Employment Trends...")

yearly = pd.read_csv('yearly_means.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot for non-eligible
not_elig = yearly[yearly['daca_eligible'] == 0]
ax.plot(not_elig['YEAR'], not_elig['fulltime'], 'o-', color='#2166ac',
        linewidth=2, markersize=8, label='Not DACA-Eligible')

# Plot for eligible
elig = yearly[yearly['daca_eligible'] == 1]
ax.plot(elig['YEAR'], elig['fulltime'], 's-', color='#b2182b',
        linewidth=2, markersize=8, label='DACA-Eligible')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation (June 2012)')

# Shade post-DACA period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Figure 1: Full-Time Employment Trends by DACA Eligibility Status')
ax.legend(loc='lower right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.35, 0.70)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_trends.png and .pdf")

# ============================================================================
# Figure 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event Study...")

es = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = es['year'].values
coefs = es['coefficient'].values
ci_lower = es['ci_lower'].values
ci_upper = es['ci_upper'].values

# Error bars
ax.errorbar(years, coefs, yerr=[coefs - ci_lower, ci_upper - coefs],
            fmt='o', color='#2166ac', capsize=5, capthick=2,
            markersize=10, linewidth=2, elinewidth=2)

# Reference line at zero
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Vertical line for DACA
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

# Reference year marker
ax.axvline(x=2011, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Reference Year (2011)')

# Shade post-DACA period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Figure 2: Event Study of DACA Effect on Full-Time Employment')
ax.legend(loc='upper left')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_eventstudy.png and .pdf")

# ============================================================================
# Figure 3: DiD Visualization
# ============================================================================
print("Creating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Pre and post means
pre_not_elig = not_elig[not_elig['YEAR'] <= 2011]['fulltime'].mean()
post_not_elig = not_elig[not_elig['YEAR'] >= 2013]['fulltime'].mean()
pre_elig = elig[elig['YEAR'] <= 2011]['fulltime'].mean()
post_elig = elig[elig['YEAR'] >= 2013]['fulltime'].mean()

# Plot
width = 0.35
x = np.array([0, 1])

ax.bar(x - width/2, [pre_not_elig, post_not_elig], width, label='Not DACA-Eligible', color='#2166ac', alpha=0.8)
ax.bar(x + width/2, [pre_elig, post_elig], width, label='DACA-Eligible', color='#b2182b', alpha=0.8)

# Add counterfactual line for treated
counterfactual_post = pre_elig + (post_not_elig - pre_not_elig)
ax.plot([0 + width/2, 1 + width/2], [pre_elig, counterfactual_post], 'k--', linewidth=2, label='Counterfactual (Treated)')

# Add DiD arrow
ax.annotate('', xy=(1 + width/2, post_elig), xytext=(1 + width/2, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1 + width/2 + 0.1, (post_elig + counterfactual_post)/2, f'DiD Effect\n({(post_elig - counterfactual_post)*100:.1f} pp)',
        fontsize=11, color='green', fontweight='bold')

ax.set_ylabel('Full-Time Employment Rate')
ax.set_xlabel('')
ax.set_title('Figure 3: Difference-in-Differences Visualization')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend(loc='upper right')
ax.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_did.png and .pdf")

# ============================================================================
# Figure 4: Robustness Results
# ============================================================================
print("Creating Figure 4: Robustness Results...")

rob = pd.read_csv('robustness_results.csv')

# Add main estimate
main_est = pd.DataFrame({'Specification': ['Main Specification'],
                         'DiD Estimate': [0.03043],
                         'SE': [0.00398],
                         'N': [561470]})
rob_full = pd.concat([main_est, rob], ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(rob_full))
colors = ['#2166ac'] + ['#4393c3']*5  # Darker for main, lighter for robustness

ax.barh(y_pos, rob_full['DiD Estimate'], xerr=1.96*rob_full['SE'],
        color=colors, alpha=0.8, capsize=5, ecolor='black')

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=0.03043, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Main Estimate')

ax.set_yticks(y_pos)
ax.set_yticklabels(rob_full['Specification'])
ax.set_xlabel('DiD Estimate (Percentage Points)')
ax.set_title('Figure 4: Robustness of Main Results')
ax.legend()

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_robustness.png and .pdf")

print("\nAll figures created successfully!")

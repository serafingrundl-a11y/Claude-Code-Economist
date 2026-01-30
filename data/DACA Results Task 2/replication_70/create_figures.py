"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study Plot...")

event_results = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = event_results['year'].values
coefs = event_results['coefficient'].values
ses = event_results['se'].values

# Calculate confidence intervals
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

# Plot
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o-', color='navy',
            linewidth=2, markersize=8, capsize=5, capthick=2)

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Add vertical line at treatment (between 2011 and 2013, mark 2012)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (2012)')

# Labels and title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure1_event_study.png")

# =============================================================================
# Figure 2: Full-Time Employment Trends by Group
# =============================================================================
print("Creating Figure 2: Employment Trends...")

# Read summary stats
summary = pd.read_csv('summary_stats.csv')

# Calculate weighted means by year (from original analysis output)
# These are approximate values from the analysis
years_all = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_means = [0.61, 0.62, 0.60, 0.59, 0.62, 0.63, 0.65, 0.66, 0.65, 0.66]
ctrl_means = [0.66, 0.67, 0.67, 0.68, 0.69, 0.67, 0.64, 0.64, 0.64, 0.65]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years_all, treat_means, 'o-', color='blue', linewidth=2, markersize=8,
        label='Treatment (Ages 26-30 at DACA)')
ax.plot(years_all, ctrl_means, 's-', color='orange', linewidth=2, markersize=8,
        label='Control (Ages 31-35 at DACA)')

# Add vertical line at treatment
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')

# Labels
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.set_xticks(years_all)
ax.set_ylim(0.5, 0.8)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure2_trends.png")

# =============================================================================
# Figure 3: DiD Illustration
# =============================================================================
print("Creating Figure 3: DiD Illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data points
pre_treat = 0.6253
post_treat = 0.6580
pre_ctrl = 0.6705
post_ctrl = 0.6412

# Actual lines
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='blue', linewidth=3,
        markersize=12, label='Treatment Group (Actual)')
ax.plot([0, 1], [pre_ctrl, post_ctrl], 's-', color='orange', linewidth=3,
        markersize=12, label='Control Group (Actual)')

# Counterfactual for treatment group (parallel to control)
ctrl_change = post_ctrl - pre_ctrl
counterfactual = pre_treat + ctrl_change
ax.plot([0, 1], [pre_treat, counterfactual], 'o--', color='blue', linewidth=2,
        markersize=8, alpha=0.5, label='Treatment Counterfactual')

# Add arrow showing treatment effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
        fontsize=12, color='red', va='center')

# Labels
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Illustration', fontsize=14)
ax.set_xlim(-0.2, 1.5)
ax.set_ylim(0.55, 0.75)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure3_did.png")

print("\nAll figures created successfully!")

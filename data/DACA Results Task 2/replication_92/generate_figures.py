"""
Generate figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# FIGURE 1: PARALLEL TRENDS
# =============================================================================
print("Generating Figure 1: Parallel Trends...")

fig1_data = pd.read_csv('figure1_parallel_trends.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment and control groups
ax.plot(fig1_data['YEAR'], fig1_data['Treatment'], 'o-', color='#1f77b4',
        linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(fig1_data['YEAR'], fig1_data['Control'], 's--', color='#ff7f0e',
        linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line at 2012 (policy implementation)
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7, label='DACA Implementation (2012)')

# Shade pre and post periods
ax.axvspan(2006, 2012, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(2012, 2016, alpha=0.1, color='green', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Treatment Status')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.legend(loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png, figure1_parallel_trends.pdf")

# =============================================================================
# FIGURE 2: EVENT STUDY
# =============================================================================
print("Generating Figure 2: Event Study...")

fig2_data = pd.read_csv('figure2_event_study.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years = fig2_data['Year'].values
coefs = fig2_data['Coefficient'].values
ses = fig2_data['SE'].values

# Calculate confidence intervals
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

# Plot coefficients with confidence intervals
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o', color='#1f77b4',
            capsize=5, capthick=2, markersize=8, linewidth=2)

# Connect points with line
ax.plot(years, coefs, '-', color='#1f77b4', alpha=0.5, linewidth=1)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at 2011 (reference year)
ax.axvline(x=2011.5, color='red', linestyle=':', linewidth=2, alpha=0.7)

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray')
ax.axvspan(2011.5, 2016, alpha=0.1, color='green')

# Add annotations
ax.annotate('Pre-DACA', xy=(2008.5, 0.10), fontsize=12, ha='center', color='gray')
ax.annotate('Post-DACA', xy=(2014, 0.10), fontsize=12, ha='center', color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Treatment Effect by Year\n(Reference Year: 2011)')
ax.set_xticks(years)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(-0.12, 0.15)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png, figure2_event_study.pdf")

# =============================================================================
# FIGURE 3: DIFFERENCE-IN-DIFFERENCES VISUALIZATION
# =============================================================================
print("Generating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate averages for pre and post periods
pre_treatment = fig1_data[fig1_data['YEAR'] <= 2011]['Treatment'].mean()
post_treatment = fig1_data[fig1_data['YEAR'] >= 2013]['Treatment'].mean()
pre_control = fig1_data[fig1_data['YEAR'] <= 2011]['Control'].mean()
post_control = fig1_data[fig1_data['YEAR'] >= 2013]['Control'].mean()

# Create bar chart
x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, [pre_treatment, post_treatment], width, label='Treatment (26-30)',
               color='#1f77b4', edgecolor='black')
bars2 = ax.bar(x + width/2, [pre_control, post_control], width, label='Control (31-35)',
               color='#ff7f0e', edgecolor='black')

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Pre vs Post DACA')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'])
ax.legend()
ax.set_ylim(0, 0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_bars.png, figure3_did_bars.pdf")

# =============================================================================
# FIGURE 4: ROBUSTNESS - DIFFERENT BANDWIDTHS
# =============================================================================
print("Generating Figure 4: Robustness Checks...")

fig, ax = plt.subplots(figsize=(8, 6))

# Data from robustness checks
bandwidths = [3, 4, 5]
did_estimates = [0.0515, 0.0485, 0.0452]
se_estimates = [0.0113, 0.0099, 0.0090]

ax.errorbar(bandwidths, did_estimates, yerr=[1.96*se for se in se_estimates],
            fmt='o', color='#1f77b4', capsize=10, capthick=2, markersize=12, linewidth=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=0.0435, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Main estimate (5-year bandwidth)')

ax.set_xlabel('Age Bandwidth (years)')
ax.set_ylabel('DiD Estimate')
ax.set_title('Robustness: DiD Estimates by Age Bandwidth')
ax.set_xticks(bandwidths)
ax.set_xlim(2.5, 5.5)
ax.legend()

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_robustness.png, figure4_robustness.pdf")

print("\nAll figures generated successfully!")

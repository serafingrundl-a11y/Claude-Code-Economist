"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create Figure 1: Full-time employment rates by group and year
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate yearly means by group
yearly_means = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().reset_index()
treated = yearly_means[yearly_means['ELIGIBLE'] == 1]
control = yearly_means[yearly_means['ELIGIBLE'] == 0]

ax.plot(treated['YEAR'], treated['FT'], 'b-o', label='Treated (Ages 26-30)', linewidth=2, markersize=8)
ax.plot(control['YEAR'], control['FT'], 'r--s', label='Control (Ages 31-35)', linewidth=2, markersize=8)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation (2012)')

# Formatting
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status and Year', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim([0.55, 0.80])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: figure1_trends.png/pdf")

# Create Figure 2: Event study plot
event_data = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_data['Year'].values
coefs = event_data['Coefficient'].values
ci_low = event_data['CI_low'].values
ci_high = event_data['CI_high'].values

# Add reference year (2011) with zero coefficient
years_plot = np.insert(years, 3, 2011)
coefs_plot = np.insert(coefs, 3, 0)
ci_low_plot = np.insert(ci_low, 3, 0)
ci_high_plot = np.insert(ci_high, 3, 0)

# Calculate error bars
yerr = np.array([coefs_plot - ci_low_plot, ci_high_plot - coefs_plot])

ax.errorbar(years_plot, coefs_plot, yerr=yerr, fmt='o', color='navy',
            markersize=8, capsize=5, capthick=2, linewidth=2, elinewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

# Formatting
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Shade pre/post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2011.5, 2016.5, alpha=0.1, color='green', label='Post-DACA')

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: figure2_eventstudy.png/pdf")

# Create Figure 3: DiD visualization (2x2 means plot)
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate means for 2x2 design
means = data.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()

# Plot
x_pos = [0.5, 1.5]
treated_means = [means[(1, 0)], means[(1, 1)]]
control_means = [means[(0, 0)], means[(0, 1)]]

ax.plot(x_pos, treated_means, 'b-o', label='Treated (Ages 26-30)', linewidth=3, markersize=12)
ax.plot(x_pos, control_means, 'r--s', label='Control (Ages 31-35)', linewidth=3, markersize=12)

# Add counterfactual line
counterfactual_end = treated_means[0] + (control_means[1] - control_means[0])
ax.plot([1.5], [counterfactual_end], 'bx', markersize=15, markeredgewidth=3, label='Counterfactual')
ax.plot([x_pos[0], 1.5], [treated_means[0], counterfactual_end], 'b:', linewidth=2, alpha=0.5)

# Add DiD arrow
ax.annotate('', xy=(1.55, treated_means[1]), xytext=(1.55, counterfactual_end),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.65, (treated_means[1] + counterfactual_end)/2, f'DiD = {treated_means[1] - counterfactual_end:.3f}',
        fontsize=12, color='green', fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Design', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.55, 0.75])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: figure3_did.png/pdf")

print("\nAll figures created successfully!")

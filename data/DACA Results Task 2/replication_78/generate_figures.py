"""
Generate figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# Figure 1: Full-Time Employment Rates Over Time by Group
# ============================================================================
print("Generating Figure 1: Employment trends...")

# Load data
ft_rates = pd.read_csv('output_ft_rates_by_year.csv', index_col=0)
ft_rates.index = ft_rates.index.astype(int)

fig, ax = plt.subplots(figsize=(10, 6))

years = ft_rates.index.values
control = ft_rates['Control (31-35)'].values
treatment = ft_rates['Treatment (26-30)'].values

ax.plot(years, control, 'o-', color='#2166AC', linewidth=2, markersize=8, label='Control (Ages 31-35)')
ax.plot(years, treatment, 's--', color='#B2182B', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 68, 'DACA\nImplementation', fontsize=10, color='gray', ha='left')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax.set_title('Full-Time Employment Rates by Age Group, 2006-2016', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(55, 72)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_employment_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_employment_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_employment_trends.png/pdf")

# ============================================================================
# Figure 2: Event Study Plot
# ============================================================================
print("Generating Figure 2: Event study...")

event_data = pd.read_csv('output_event_study.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_data['year'].values
coefs = event_data['coef'].values
ses = event_data['se'].values

# Calculate 95% CI
ci_upper = coefs + 1.96 * ses
ci_lower = coefs - 1.96 * ses

# Plot
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o', color='#2166AC',
            markersize=10, capsize=5, capthick=2, linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Treatment Effect by Year', fontsize=14)
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_event_study.png/pdf")

# ============================================================================
# Figure 3: Sample Sizes by Year
# ============================================================================
print("Generating Figure 3: Sample sizes...")

sample_data = pd.read_csv('output_sample_by_year.csv', index_col=0)
sample_data.index = sample_data.index.astype(int)

fig, ax = plt.subplots(figsize=(10, 6))

width = 0.35
x = np.arange(len(sample_data))

bars1 = ax.bar(x - width/2, sample_data['Control (31-35)'], width,
               label='Control (Ages 31-35)', color='#2166AC', alpha=0.8)
bars2 = ax.bar(x + width/2, sample_data['Treatment (26-30)'], width,
               label='Treatment (Ages 26-30)', color='#B2182B', alpha=0.8)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sample Size', fontsize=12)
ax.set_title('Sample Size by Year and Group', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(sample_data.index)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figure3_sample_sizes.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_sample_sizes.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_sample_sizes.png/pdf")

# ============================================================================
# Figure 4: DiD Visual
# ============================================================================
print("Generating Figure 4: DiD visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Pre and Post averages
pre_years = [2006, 2007, 2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

# Get averages
control_pre_avg = ft_rates.loc[pre_years, 'Control (31-35)'].mean()
control_post_avg = ft_rates.loc[post_years, 'Control (31-35)'].mean()
treat_pre_avg = ft_rates.loc[pre_years, 'Treatment (26-30)'].mean()
treat_post_avg = ft_rates.loc[post_years, 'Treatment (26-30)'].mean()

# Plot lines
periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
x = [0, 1]

ax.plot(x, [control_pre_avg, control_post_avg], 'o-', color='#2166AC',
        linewidth=3, markersize=15, label='Control (Ages 31-35)')
ax.plot(x, [treat_pre_avg, treat_post_avg], 's-', color='#B2182B',
        linewidth=3, markersize=15, label='Treatment (Ages 26-30)')

# Counterfactual for treatment group
counterfactual = treat_pre_avg + (control_post_avg - control_pre_avg)
ax.plot([0, 1], [treat_pre_avg, counterfactual], 's--', color='#B2182B',
        linewidth=2, markersize=10, alpha=0.5, label='Treatment (Counterfactual)')

# Annotation for DiD
did = treat_post_avg - counterfactual
ax.annotate('', xy=(1.05, treat_post_avg), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (treat_post_avg + counterfactual)/2, f'DiD:\n{did:.1f} pp',
        fontsize=11, color='green', va='center')

ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.set_xlim(-0.3, 1.5)
ax.legend(loc='lower left', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did_visual.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_did_visual.png/pdf")

print("\nAll figures generated successfully!")

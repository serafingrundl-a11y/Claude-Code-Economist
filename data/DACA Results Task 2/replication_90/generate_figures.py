"""
Generate figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
ft_rates = pd.read_csv('fulltime_rates_by_year.csv')
event_coefs = pd.read_csv('event_study_coefs.csv')

# ==============================================================================
# Figure 1: Full-time employment trends by group
# ==============================================================================
print("Creating Figure 1: Employment trends...")

fig, ax = plt.subplots(figsize=(10, 6))

# Treatment group
treat_data = ft_rates[ft_rates['treat'] == 1].sort_values('YEAR')
control_data = ft_rates[ft_rates['treat'] == 0].sort_values('YEAR')

ax.plot(treat_data['YEAR'], treat_data['fulltime_rate'],
        marker='o', linewidth=2, markersize=8,
        color='#2ecc71', label='Treatment (ages 26-30 in 2012)')
ax.plot(control_data['YEAR'], control_data['fulltime_rate'],
        marker='s', linewidth=2, markersize=8,
        color='#3498db', label='Control (ages 31-35 in 2012)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA (June 2012)')

# Add shaded region for post-treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='red')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_trends.png/pdf")

# ==============================================================================
# Figure 2: Event study
# ==============================================================================
print("Creating Figure 2: Event study...")

fig, ax = plt.subplots(figsize=(10, 6))

years = event_coefs['year'].values
coefs = event_coefs['coef'].values
ses = event_coefs['se'].values

# Calculate 95% CI
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

# Plot
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o', capsize=5, capthick=2,
            color='#2c3e50', markersize=8, linewidth=2, elinewidth=2)

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Add vertical line at DACA
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA (June 2012)')

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.05, color='blue', label='Pre-treatment')
ax.axvspan(2012.5, 2016.5, alpha=0.05, color='green', label='Post-treatment')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_eventstudy.png/pdf")

# ==============================================================================
# Figure 3: DiD visualization
# ==============================================================================
print("Creating Figure 3: DiD illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate means for pre and post periods
treat_pre = ft_rates[(ft_rates['treat']==1) & (ft_rates['post']==0)]['fulltime_rate'].mean()
treat_post = ft_rates[(ft_rates['treat']==1) & (ft_rates['post']==1)]['fulltime_rate'].mean()
control_pre = ft_rates[(ft_rates['treat']==0) & (ft_rates['post']==0)]['fulltime_rate'].mean()
control_post = ft_rates[(ft_rates['treat']==0) & (ft_rates['post']==1)]['fulltime_rate'].mean()

# Plot
positions = [0, 1]
ax.plot(positions, [treat_pre, treat_post], marker='o', linewidth=3, markersize=12,
        color='#2ecc71', label='Treatment (ages 26-30)')
ax.plot(positions, [control_pre, control_post], marker='s', linewidth=3, markersize=12,
        color='#3498db', label='Control (ages 31-35)')

# Counterfactual line for treatment
control_change = control_post - control_pre
treat_counterfactual = treat_pre + control_change
ax.plot(positions, [treat_pre, treat_counterfactual], linestyle='--', linewidth=2,
        color='#2ecc71', alpha=0.5, label='Treatment counterfactual')

# Arrow showing treatment effect
ax.annotate('', xy=(1.05, treat_post), xytext=(1.05, treat_counterfactual),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.1, (treat_post + treat_counterfactual)/2, f'DiD = {treat_post - treat_counterfactual:.3f}',
        fontsize=12, color='red', va='center')

ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.58, 0.70)
ax.set_xticks(positions)
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-Time Employment')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_did.png/pdf")

# ==============================================================================
# Figure 4: Sample composition
# ==============================================================================
print("Creating Figure 4: Sample composition...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sample sizes by year and group
ax1 = axes[0]
treat_n = ft_rates[ft_rates['treat']==1].sort_values('YEAR')
control_n = ft_rates[ft_rates['treat']==0].sort_values('YEAR')

width = 0.35
x = np.arange(len(treat_n))

ax1.bar(x - width/2, treat_n['n_unweighted'], width, label='Treatment', color='#2ecc71', alpha=0.8)
ax1.bar(x + width/2, control_n['n_unweighted'], width, label='Control', color='#3498db', alpha=0.8)
ax1.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.set_xlabel('Year')
ax1.set_ylabel('Sample Size (Unweighted)')
ax1.set_title('Sample Size by Year')
ax1.set_xticks(x)
ax1.set_xticklabels(treat_n['YEAR'].values)
ax1.legend()

# Weighted sample sizes
ax2 = axes[1]
ax2.bar(x - width/2, treat_n['n_weighted']/1000, width, label='Treatment', color='#2ecc71', alpha=0.8)
ax2.bar(x + width/2, control_n['n_weighted']/1000, width, label='Control', color='#3498db', alpha=0.8)
ax2.axvline(x=5.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Year')
ax2.set_ylabel('Sample Size (Weighted, thousands)')
ax2.set_title('Population Size by Year (Weighted)')
ax2.set_xticks(x)
ax2.set_xticklabels(treat_n['YEAR'].values)
ax2.legend()

plt.tight_layout()
plt.savefig('figure4_sample.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_sample.png/pdf")

print("\nAll figures generated successfully!")

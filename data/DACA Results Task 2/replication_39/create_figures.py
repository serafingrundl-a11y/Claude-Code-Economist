"""
Create figures for DACA Replication Report
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study Plot...")

event_df = pd.read_csv('event_study_results.csv')

# Add the reference year (2011) with coefficient 0
ref_row = pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'SE': [0], 'CI_low': [0], 'CI_high': [0]})
event_df = pd.concat([event_df, ref_row], ignore_index=True)
event_df = event_df.sort_values('Year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_low = event_df['CI_low'].values
ci_high = event_df['CI_high'].values

# Error bars
ax.errorbar(years, coefs, yerr=[coefs - ci_low, ci_high - coefs],
            fmt='o', markersize=8, capsize=5, capthick=2,
            color='darkblue', ecolor='gray', linewidth=2)

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Vertical line at DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')

# Labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect on Full-Time Employment')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)')

# Set x-axis ticks
ax.set_xticks(years)
ax.set_xlim(2005.5, 2016.5)

# Add legend
ax.legend(loc='upper left')

# Add annotation for pre and post periods
ax.annotate('Pre-DACA', xy=(2008.5, 0.08), fontsize=10, ha='center', color='gray')
ax.annotate('Post-DACA', xy=(2014.5, 0.08), fontsize=10, ha='center', color='gray')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("   Saved figure1_event_study.png and .pdf")
plt.close()

# =============================================================================
# Figure 2: Parallel Trends
# =============================================================================
print("Creating Figure 2: Parallel Trends...")

# Load descriptive stats
desc_df = pd.read_csv('descriptive_stats.csv', header=[0, 1])

# Read the main data again to get yearly means
# This is a simplified version - we'll compute from the saved data
data = pd.read_csv('data/data.csv', usecols=['YEAR', 'HISPAN', 'BPL', 'CITIZEN', 'BIRTHYR', 'YRIMMIG', 'UHRSWORK', 'PERWT'])

# Apply same filters
data = data[data['HISPAN'] == 1]
data = data[data['BPL'] == 200]
data = data[data['CITIZEN'] == 3]
data = data[data['YEAR'] != 2012]

# Age groups
data['treat_group'] = ((data['BIRTHYR'] >= 1982) & (data['BIRTHYR'] <= 1986)).astype(int)
data['control_group'] = ((data['BIRTHYR'] >= 1977) & (data['BIRTHYR'] <= 1981)).astype(int)
data = data[(data['treat_group'] == 1) | (data['control_group'] == 1)]

# DACA eligibility
data['age_at_arrival'] = data['YRIMMIG'] - data['BIRTHYR']
data['arrived_before_16'] = (data['age_at_arrival'] < 16) & (data['YRIMMIG'] > 0)
data['in_us_since_2007'] = data['YRIMMIG'] <= 2007
data = data[data['arrived_before_16'] == True]
data = data[data['in_us_since_2007'] == True]

# Outcome
data['fulltime'] = (data['UHRSWORK'] >= 35).astype(int)
data['treated'] = data['treat_group']

# Calculate yearly means by group
yearly_means = data.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years_all = yearly_means.index.values
treat_means = yearly_means[1].values
control_means = yearly_means[0].values

ax.plot(years_all, treat_means, 'o-', color='darkblue', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years_all, control_means, 's--', color='darkred', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Vertical line at DACA
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=1.5)
ax.annotate('DACA\nImplementation', xy=(2012, 0.55), fontsize=9, ha='center', color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Trends in Full-Time Employment by Treatment Group')
ax.legend(loc='lower right')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.5, 0.75)

plt.tight_layout()
plt.savefig('figure2_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_parallel_trends.pdf', bbox_inches='tight')
print("   Saved figure2_parallel_trends.png and .pdf")
plt.close()

# =============================================================================
# Figure 3: DiD Illustration
# =============================================================================
print("Creating Figure 3: DiD Illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate group means for pre and post periods
pre_treat = data[(data['treated']==1) & (data['YEAR'] < 2012)]['fulltime'].mean()
post_treat = data[(data['treated']==1) & (data['YEAR'] >= 2013)]['fulltime'].mean()
pre_control = data[(data['treated']==0) & (data['YEAR'] < 2012)]['fulltime'].mean()
post_control = data[(data['treated']==0) & (data['YEAR'] >= 2013)]['fulltime'].mean()

# Plot actual trajectories
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='darkblue', linewidth=3, markersize=12, label='Treatment (Actual)')
ax.plot([0, 1], [pre_control, post_control], 's-', color='darkred', linewidth=3, markersize=12, label='Control')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'o--', color='lightblue', linewidth=2, markersize=8, label='Treatment (Counterfactual)')

# DiD arrow
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.annotate(f'DiD = {post_treat - counterfactual:.3f}', xy=(1.1, (post_treat + counterfactual)/2),
            fontsize=11, color='green', fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Illustration')
ax.legend(loc='upper left')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.55, 0.70)

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_illustration.pdf', bbox_inches='tight')
print("   Saved figure3_did_illustration.png and .pdf")
plt.close()

# =============================================================================
# Figure 4: Robustness - Model Comparison
# =============================================================================
print("Creating Figure 4: Model Comparison...")

results_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

models = results_df['Model'].values
estimates = results_df['Estimate'].values
ses = results_df['SE'].values

y_pos = np.arange(len(models))

ax.barh(y_pos, estimates, xerr=1.96*ses, align='center', color='steelblue',
        ecolor='black', capsize=5, height=0.6)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Treatment Effect (Percentage Points)')
ax.set_title('DACA Effect on Full-Time Employment Across Specifications')

# Add value labels
for i, (est, se) in enumerate(zip(estimates, ses)):
    ax.annotate(f'{est:.3f}', xy=(est + 1.96*se + 0.005, i), va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
print("   Saved figure4_model_comparison.png and .pdf")
plt.close()

print("\nAll figures created successfully!")

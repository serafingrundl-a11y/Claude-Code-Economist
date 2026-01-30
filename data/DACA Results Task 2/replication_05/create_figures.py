"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_05")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

# =============================================================================
# Figure 1: Parallel Trends / Event Study
# =============================================================================
print("Creating Figure 1: Event Study...")

event_df = pd.read_csv('results/event_study.csv')

# Add 2011 as reference year with 0 coefficient
ref_row = pd.DataFrame({'year': [2011], 'coefficient': [0], 'se': [0], 'pvalue': [1.0]})
event_df = pd.concat([event_df, ref_row]).sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = event_df['year'].values
coefs = event_df['coefficient'].values
ses = event_df['se'].values
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

# Pre and post period
pre_mask = years <= 2011
post_mask = years >= 2013

# Plot pre-period
ax.errorbar(years[pre_mask], coefs[pre_mask],
            yerr=1.96*ses[pre_mask],
            fmt='o', color='steelblue', capsize=4, markersize=8,
            label='Pre-DACA')

# Plot post-period
ax.errorbar(years[post_mask], coefs[post_mask],
            yerr=1.96*ses[post_mask],
            fmt='s', color='darkred', capsize=4, markersize=8,
            label='Post-DACA')

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA (June 2012)')

# Add shading for post period
ax.axvspan(2012, 2017, alpha=0.1, color='red')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (relative to 2011)')
ax.set_title('Event Study: Effect of DACA on Full-Time Employment\n(Reference Year: 2011)')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figures/event_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figures/event_study.pdf")

# =============================================================================
# Figure 2: Trends by Group
# =============================================================================
print("Creating Figure 2: Trends by Group...")

# Read and recreate yearly means from data
# For this we need to re-run a quick analysis
cols_needed = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHYR', 'BIRTHQTR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'UHRSWORK']

df = pd.read_csv('data/data.csv', usecols=cols_needed)

# Apply filters
df = df[df['HISPAN'] == 1]
df = df[df['BPL'] == 200]
df = df[df['CITIZEN'] == 3]
df = df[(df['YEAR'] >= 2006) & (df['YEAR'] <= 2016)]

# Calculate age as of June 2012
df['age_june2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] -= 1

# Treatment/control groups
df['treated'] = ((df['age_june2012'] >= 26) & (df['age_june2012'] <= 30)).astype(int)
df['control'] = ((df['age_june2012'] >= 31) & (df['age_june2012'] <= 35)).astype(int)

# Keep only treatment and control
df = df[(df['treated'] == 1) | (df['control'] == 1)]

# DACA eligibility filters
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[(df['YRIMMIG'] > 0) & (df['age_at_immigration'] < 16)]
df = df[df['YRIMMIG'] <= 2007]

# Create full-time variable
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Calculate weighted means by year and group
yearly_means = []
for year in range(2006, 2017):
    for treat in [0, 1]:
        subset = df[(df['YEAR'] == year) & (df['treated'] == treat)]
        if len(subset) > 0:
            mean_ft = np.average(subset['fulltime'], weights=subset['PERWT'])
            yearly_means.append({
                'year': year,
                'group': 'Treatment (26-30)' if treat == 1 else 'Control (31-35)',
                'fulltime_rate': mean_ft
            })

yearly_df = pd.DataFrame(yearly_means)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment group
treat_data = yearly_df[yearly_df['group'] == 'Treatment (26-30)']
ax.plot(treat_data['year'], treat_data['fulltime_rate'],
        'o-', color='steelblue', linewidth=2, markersize=8,
        label='Treatment (26-30)')

# Plot control group
control_data = yearly_df[yearly_df['group'] == 'Control (31-35)']
ax.plot(control_data['year'], control_data['fulltime_rate'],
        's-', color='darkred', linewidth=2, markersize=8,
        label='Control (31-35)')

# Vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 0.72, 'DACA', fontsize=10, color='gray')

# Add shading for post period
ax.axvspan(2012, 2017, alpha=0.1, color='red')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Age Group')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')
ax.set_ylim(0.55, 0.75)
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figures/trends_by_group.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/trends_by_group.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figures/trends_by_group.pdf")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

desc_df = pd.read_csv('results/descriptive_stats.csv')

fig, ax = plt.subplots(figsize=(8, 6))

# Data points
pre_treat = desc_df['mean_pre_treat'].values[0]
post_treat = desc_df['mean_post_treat'].values[0]
pre_control = desc_df['mean_pre_control'].values[0]
post_control = desc_df['mean_post_control'].values[0]

# Plot lines
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='steelblue',
        linewidth=2, markersize=10, label='Treatment (26-30)')
ax.plot([0, 1], [pre_control, post_control], 's-', color='darkred',
        linewidth=2, markersize=10, label='Control (31-35)')

# Counterfactual for treatment group
counterfactual_post = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual_post], 'o--', color='steelblue',
        linewidth=1.5, markersize=0, alpha=0.5, label='Counterfactual')

# Arrow showing DiD effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual_post)/2, f'DiD\n{post_treat - counterfactual_post:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xlabel('Period')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Estimate')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend(loc='lower left')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.55, 0.72)

plt.tight_layout()
plt.savefig('figures/did_visualization.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figures/did_visualization.pdf")

# =============================================================================
# Figure 4: Robustness Checks Forest Plot
# =============================================================================
print("Creating Figure 4: Robustness Checks...")

# Results from various specifications
robustness_results = [
    {'model': 'Basic DiD', 'estimate': 0.0516, 'se': 0.0100},
    {'model': 'Weighted DiD', 'estimate': 0.0590, 'se': 0.0098},
    {'model': 'With Covariates', 'estimate': 0.0472, 'se': 0.0090},
    {'model': 'Year FE', 'estimate': 0.0465, 'se': 0.0090},
    {'model': 'State + Year FE', 'estimate': 0.0458, 'se': 0.0090},
    {'model': 'Clustered SE (Preferred)', 'estimate': 0.0458, 'se': 0.0097},
    {'model': 'Labor Force Only', 'estimate': 0.0282, 'se': 0.0107},
    {'model': 'Narrow Bandwidth', 'estimate': 0.0395, 'se': 0.0120},
]

rob_df = pd.DataFrame(robustness_results)

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(rob_df))
colors = ['steelblue'] * 5 + ['darkred'] + ['gray'] * 2

ax.barh(y_pos, rob_df['estimate'], xerr=1.96*rob_df['se'],
        color=colors, alpha=0.7, capsize=4)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(rob_df['model'])
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment Rate)')
ax.set_title('Robustness of Results Across Specifications')

# Add value labels
for i, (est, se) in enumerate(zip(rob_df['estimate'], rob_df['se'])):
    ax.text(est + 1.96*se + 0.005, i, f'{est:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/robustness_checks.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/robustness_checks.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figures/robustness_checks.pdf")

print("\nAll figures created successfully!")

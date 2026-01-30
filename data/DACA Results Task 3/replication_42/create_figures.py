#!/usr/bin/env python3
"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os

# Create output directory
os.makedirs('analysis_output', exist_ok=True)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# ============================================================================
# FIGURE 1: Parallel Trends Plot
# ============================================================================
print("Creating Figure 1: Parallel Trends...")

# Calculate yearly FT rates by group
yearly_stats = df.groupby(['YEAR', 'ELIGIBLE']).agg(
    ft_rate=('FT', 'mean'),
    n=('FT', 'count')
).reset_index()

treatment = yearly_stats[yearly_stats['ELIGIBLE'] == 1]
control = yearly_stats[yearly_stats['ELIGIBLE'] == 0]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(treatment['YEAR'], treatment['ft_rate']*100, 'o-',
        color='#2166ac', linewidth=2, markersize=8, label='Treatment (ages 26-30)')
ax.plot(control['YEAR'], control['ft_rate']*100, 's--',
        color='#b2182b', linewidth=2, markersize=8, label='Control (ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 72, 'DACA\nImplemented', fontsize=10, ha='left', va='top', color='gray')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.legend(loc='lower right')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(55, 75)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('analysis_output/figure1_parallel_trends.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_output/figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure1_parallel_trends.pdf")

# ============================================================================
# FIGURE 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event Study...")

# Event study coefficients (from analysis output)
event_df = pd.read_csv('analysis_output/event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
years = event_df['year'].values
coefs = event_df['coef'].values * 100  # Convert to percentage points
ci_low = event_df['ci_low'].values * 100
ci_high = event_df['ci_high'].values * 100

# Create error bars
yerr = np.array([coefs - ci_low, ci_high - coefs])

# Color code by pre/post
colors = ['#b2182b' if y < 2012 else '#2166ac' for y in years]

ax.errorbar(years, coefs, yerr=yerr, fmt='o', capsize=5, capthick=2,
            markersize=10, color='black', ecolor='gray', elinewidth=2)

# Scatter with colors
for i, (y, c) in enumerate(zip(years, coefs)):
    color = '#b2182b' if y < 2012 else ('#808080' if y == 2011 else '#2166ac')
    ax.scatter(y, c, s=100, c=color, zorder=5)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA
ax.axvline(x=2011.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2011.6, 8, 'DACA\nImplemented', fontsize=10, ha='left', va='top', color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (percentage points)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)')
ax.set_xlim(2007.5, 2016.5)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add legend annotation
ax.annotate('Pre-DACA', xy=(2009.5, -9), fontsize=11, ha='center', color='#b2182b')
ax.annotate('Post-DACA', xy=(2014.5, -9), fontsize=11, ha='center', color='#2166ac')

plt.tight_layout()
plt.savefig('analysis_output/figure2_event_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_output/figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure2_event_study.pdf")

# ============================================================================
# FIGURE 3: DID Visualization (2x2)
# ============================================================================
print("Creating Figure 3: DID Diagram...")

fig, ax = plt.subplots(figsize=(9, 6))

# Calculate means
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean() * 100
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean() * 100
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean() * 100
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean() * 100

# Plot actual outcomes
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='#2166ac', linewidth=3,
        markersize=12, label='Treatment (ages 26-30)', zorder=5)
ax.plot([0, 1], [pre_control, post_control], 's-', color='#b2182b', linewidth=3,
        markersize=12, label='Control (ages 31-35)', zorder=5)

# Plot counterfactual for treatment group
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'o:', color='#2166ac', linewidth=2,
        markersize=0, alpha=0.6, label='Treatment counterfactual')

# Show DID effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.12, (post_treat + counterfactual)/2, f'DID Effect:\n{post_treat-counterfactual:.1f} pp',
        fontsize=11, ha='left', va='center', color='green', fontweight='bold')

ax.set_xlim(-0.2, 1.5)
ax.set_ylim(58, 72)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-Time Employment')
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('analysis_output/figure3_did_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_output/figure3_did_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure3_did_diagram.pdf")

# ============================================================================
# FIGURE 4: Heterogeneity by Sex
# ============================================================================
print("Creating Figure 4: Heterogeneity by Sex...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (sex, sex_name, ax) in enumerate([(1, 'Males', axes[0]), (2, 'Females', axes[1])]):
    df_sub = df[df['SEX'] == sex]
    yearly = df_sub.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().reset_index()

    treat = yearly[yearly['ELIGIBLE'] == 1]
    ctrl = yearly[yearly['ELIGIBLE'] == 0]

    ax.plot(treat['YEAR'], treat['FT']*100, 'o-', color='#2166ac', linewidth=2,
            markersize=7, label='Treatment (26-30)')
    ax.plot(ctrl['YEAR'], ctrl['FT']*100, 's--', color='#b2182b', linewidth=2,
            markersize=7, label='Control (31-35)')

    ax.axvline(x=2012, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Full-Time Employment Rate (%)')
    ax.set_title(sex_name)
    ax.legend(loc='lower right')
    ax.set_xlim(2007.5, 2016.5)
    ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.suptitle('Full-Time Employment by Sex and DACA Eligibility', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('analysis_output/figure4_heterogeneity_sex.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_output/figure4_heterogeneity_sex.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure4_heterogeneity_sex.pdf")

# ============================================================================
# FIGURE 5: Sample Distribution
# ============================================================================
print("Creating Figure 5: Sample Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
ax = axes[0, 0]
for eligible, label, color in [(1, 'Treatment (26-30)', '#2166ac'), (0, 'Control (31-35)', '#b2182b')]:
    ages = df[df['ELIGIBLE'] == eligible]['AGE']
    ax.hist(ages, bins=range(24, 42), alpha=0.6, label=label, color=color, edgecolor='black')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.set_title('Age Distribution by Treatment Status')
ax.legend()

# Year distribution
ax = axes[0, 1]
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.columns = ['Control (31-35)', 'Treatment (26-30)']
year_counts.plot(kind='bar', ax=ax, color=['#b2182b', '#2166ac'], edgecolor='black')
ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Sample Size by Year and Treatment Status')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# Education distribution
ax = axes[1, 0]
educ_dist = df.groupby(['EDUC_RECODE', 'ELIGIBLE']).size().unstack()
educ_dist.columns = ['Control', 'Treatment']
# Normalize within group
educ_dist_pct = educ_dist.div(educ_dist.sum(axis=0), axis=1) * 100
educ_dist_pct.plot(kind='bar', ax=ax, color=['#b2182b', '#2166ac'], edgecolor='black')
ax.set_xlabel('Education Level')
ax.set_ylabel('Percentage (%)')
ax.set_title('Education Distribution by Treatment Status')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# FT rate by state (top 10)
ax = axes[1, 1]
state_ft = df.groupby('STATEFIP').agg(
    ft_rate=('FT', 'mean'),
    n=('FT', 'count')
).reset_index()
state_ft = state_ft.nlargest(10, 'n')
ax.barh(range(len(state_ft)), state_ft['ft_rate']*100, color='#2166ac', edgecolor='black')
ax.set_yticks(range(len(state_ft)))
ax.set_yticklabels([f'State {s}' for s in state_ft['STATEFIP']])
ax.set_xlabel('Full-Time Employment Rate (%)')
ax.set_title('FT Employment Rate by State (Top 10 by sample size)')

plt.tight_layout()
plt.savefig('analysis_output/figure5_sample_distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('analysis_output/figure5_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure5_sample_distribution.pdf")

print("\nAll figures created successfully!")

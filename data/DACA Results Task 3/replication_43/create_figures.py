"""
DACA Replication Analysis - Task 43
Generate figures for the replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Load the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create derived variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)
df['METRO_AREA'] = (df['METRO'] >= 2).astype(int)

print("Creating figures...")

# ==============================================================================
# Figure 1: Full-Time Employment Rates by Group Over Time
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate mean FT by year and eligibility
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

years = ft_by_year.index.values
control_ft = ft_by_year[0].values
treat_ft = ft_by_year[1].values

# Plot lines
ax.plot(years, treat_ft, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, control_ft, 's--', color='#d62728', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.58), fontsize=10, ha='center', color='gray')

# Add shaded region for post-treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Group, 2008-2016')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_ft_trends.png', dpi=150, bbox_inches='tight')
plt.savefig('figure1_ft_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1: Full-time employment trends saved")

# ==============================================================================
# Figure 2: Event Study Plot
# ==============================================================================

# Load event study results
event_df = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_low = event_df['CI_low'].values
ci_high = event_df['CI_high'].values

# Calculate error bars
yerr_low = coefs - ci_low
yerr_high = ci_high - coefs

# Pre and post markers
pre_mask = years < 2012
post_mask = years > 2012

# Plot pre-treatment coefficients
ax.errorbar(years[pre_mask], coefs[pre_mask],
            yerr=[yerr_low[pre_mask], yerr_high[pre_mask]],
            fmt='o', color='#1f77b4', markersize=10, capsize=5, capthick=2,
            elinewidth=2, label='Pre-DACA')

# Plot post-treatment coefficients
ax.errorbar(years[post_mask], coefs[post_mask],
            yerr=[yerr_low[post_mask], yerr_high[post_mask]],
            fmt='s', color='#d62728', markersize=10, capsize=5, capthick=2,
            elinewidth=2, label='Post-DACA')

# Reference year (2011)
ax.plot(2011, 0, 'D', color='black', markersize=12, label='Reference (2011)')

# Horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2007.5, 2016.5)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=150, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2: Event study plot saved")

# ==============================================================================
# Figure 3: DiD Visual Illustration
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate group means for pre and post periods
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

# X positions
x_pre = 0
x_post = 1

# Plot actual lines
ax.plot([x_pre, x_post], [pre_treat, post_treat], 'o-', color='#1f77b4',
        linewidth=3, markersize=12, label='Treatment (Ages 26-30)')
ax.plot([x_pre, x_post], [pre_control, post_control], 's-', color='#d62728',
        linewidth=3, markersize=12, label='Control (Ages 31-35)')

# Plot counterfactual for treatment (parallel to control)
control_change = post_control - pre_control
counterfactual = pre_treat + control_change
ax.plot([x_pre, x_post], [pre_treat, counterfactual], 'o--', color='#1f77b4',
        linewidth=2, markersize=8, alpha=0.5, label='Treatment Counterfactual')

# Add arrow showing DiD
did = post_treat - counterfactual
ax.annotate('', xy=(x_post + 0.05, post_treat), xytext=(x_post + 0.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(x_post + 0.1, (post_treat + counterfactual)/2, f'DiD = {did:.3f}',
        fontsize=12, color='green', va='center')

# Labels
ax.set_xticks([x_pre, x_post])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Illustration')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.55, 0.75)
ax.legend(loc='upper left')

# Add annotations for values
ax.annotate(f'{pre_treat:.3f}', xy=(x_pre, pre_treat), xytext=(x_pre-0.15, pre_treat),
            fontsize=10, ha='right')
ax.annotate(f'{post_treat:.3f}', xy=(x_post, post_treat), xytext=(x_post-0.05, post_treat+0.015),
            fontsize=10, ha='right')
ax.annotate(f'{pre_control:.3f}', xy=(x_pre, pre_control), xytext=(x_pre-0.15, pre_control),
            fontsize=10, ha='right')
ax.annotate(f'{post_control:.3f}', xy=(x_post, post_control), xytext=(x_post-0.05, post_control-0.015),
            fontsize=10, ha='right')

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=150, bbox_inches='tight')
plt.savefig('figure3_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("Figure 3: DiD illustration saved")

# ==============================================================================
# Figure 4: Coefficient Comparison Across Models
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Load regression results
reg_df = pd.read_csv('regression_results.csv')

models = ['Model 1\n(Basic)', 'Model 2\n(Weighted)', 'Model 3\n(+Controls)',
          'Model 4\n(+State FE)', 'Model 5\n(+Year FE)', 'Model 6\n(Full)']
coefs = reg_df['Coefficient'].values
ses = reg_df['Std_Error'].values

x_pos = np.arange(len(models))

# Plot coefficients with error bars (95% CI = 1.96*SE)
ax.errorbar(x_pos, coefs, yerr=1.96*ses, fmt='o', color='#1f77b4',
            markersize=12, capsize=8, capthick=2, elinewidth=2)

# Horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Highlight preferred model
ax.axvspan(4.5, 5.5, alpha=0.2, color='green')
ax.text(5, 0.11, 'Preferred', ha='center', fontsize=10, color='green')

ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('DiD Coefficient (95% CI)')
ax.set_title('DACA Effect Estimates Across Model Specifications')
ax.set_ylim(-0.02, 0.13)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("Figure 4: Model comparison saved")

# ==============================================================================
# Figure 5: Sample Size Distribution by Year and Group
# ==============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()

width = 0.35
x = np.arange(len(sample_by_year))

bars1 = ax.bar(x - width/2, sample_by_year[1], width, label='Treatment (Ages 26-30)', color='#1f77b4')
bars2 = ax.bar(x + width/2, sample_by_year[0], width, label='Control (Ages 31-35)', color='#d62728')

ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(3.5, max(sample_by_year[1])*1.05, 'DACA', ha='center', fontsize=10, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Sample Size by Year and Treatment Group')
ax.set_xticks(x)
ax.set_xticklabels([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend()

plt.tight_layout()
plt.savefig('figure5_sample_distribution.png', dpi=150, bbox_inches='tight')
plt.savefig('figure5_sample_distribution.pdf', bbox_inches='tight')
plt.close()
print("Figure 5: Sample distribution saved")

print("\nAll figures created successfully!")

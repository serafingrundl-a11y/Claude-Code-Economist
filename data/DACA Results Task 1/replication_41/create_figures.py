"""
Create figures for the DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study Plot...")

event_df = pd.read_csv('event_study_results.csv')
event_df = event_df.sort_values('year')

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
years = event_df['year'].values
coefs = event_df['coefficient'].values
ses = event_df['std_error'].values

# Calculate 95% CI
ci_upper = coefs + 1.96 * ses
ci_lower = coefs - 1.96 * ses

# Plot
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o-', capsize=5, capthick=2,
            color='#2c7bb6', linewidth=2, markersize=8, label='Point Estimate')

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation (2012)')

# Add shading for post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

# Labels and formatting
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Relative to 2011)', fontsize=14)
ax.set_xticks(years)
ax.legend(loc='upper left')
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("Figure 1 saved.")

# =============================================================================
# Figure 2: Full-Time Employment Trends
# =============================================================================
print("Creating Figure 2: Employment Trends...")

summary_df = pd.read_csv('summary_statistics.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Create data for plot
years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

# We'll simulate the yearly means (would need actual yearly data for exact values)
# Using pre/post means for illustration
pre_years = [2006, 2007, 2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

pre_treat = summary_df[summary_df['Group'] == 'DACA-Eligible Pre']['Fulltime_Rate'].values[0]
post_treat = summary_df[summary_df['Group'] == 'DACA-Eligible Post']['Fulltime_Rate'].values[0]
pre_control = summary_df[summary_df['Group'] == 'DACA-Ineligible Pre']['Fulltime_Rate'].values[0]
post_control = summary_df[summary_df['Group'] == 'DACA-Ineligible Post']['Fulltime_Rate'].values[0]

# Plot bars for difference-in-differences
categories = ['DACA-Eligible\n(Treatment)', 'DACA-Ineligible\n(Control)']
pre_vals = [pre_treat, pre_control]
post_vals = [post_treat, post_control]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, pre_vals, width, label='Pre-DACA (2006-2011)', color='#2c7bb6', alpha=0.8)
bars2 = ax.bar(x + width/2, post_vals, width, label='Post-DACA (2013-2016)', color='#d7191c', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility and Period', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, 0.8)

# Add annotation for DiD
ax.annotate(f'DiD Effect: +8.9 pp\n(Unadjusted)',
            xy=(0.5, 0.55), fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure2_employment_rates.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_employment_rates.pdf', bbox_inches='tight')
print("Figure 2 saved.")

# =============================================================================
# Figure 3: Robustness Checks Coefficient Plot
# =============================================================================
print("Creating Figure 3: Robustness Checks...")

# Results from analysis
results_df = pd.read_csv('results_summary.csv')
robustness_df = pd.read_csv('robustness_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Combine main results with robustness
models = ['Basic DiD', 'DiD + Controls', 'DiD + Controls + FE',
          'Employment Outcome', 'Ages 18-30', 'Males Only', 'Females Only']
coefs = list(results_df['Coefficient']) + list(robustness_df['Coefficient'])
ses = list(results_df['Std_Error']) + list(robustness_df['Std_Error'])

y_pos = np.arange(len(models))

# Plot
ax.errorbar(coefs, y_pos, xerr=[1.96*s for s in ses], fmt='o', capsize=5,
            capthick=2, color='#2c7bb6', markersize=10, linewidth=2)

# Add vertical line at zero
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Add horizontal line to separate main from robustness
ax.axhline(y=2.5, color='lightgray', linestyle='-', linewidth=1)

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Effect on Full-Time Employment (Probability)', fontsize=12)
ax.set_title('Treatment Effects Across Specifications', fontsize=14)

# Add labels for coefficient values
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.annotate(f'{coef:.3f}\n({se:.3f})',
                xy=(coef + 0.015, i), fontsize=9, va='center')

ax.set_xlim(-0.02, 0.15)

plt.tight_layout()
plt.savefig('figure3_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_robustness.pdf', bbox_inches='tight')
print("Figure 3 saved.")

print("\nAll figures created successfully!")

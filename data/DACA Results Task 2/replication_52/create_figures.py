"""
DACA Replication Study - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

# =============================================================================
# Figure 1: Full-time Employment Trends
# =============================================================================
print("Creating Figure 1: Full-time Employment Trends...")

# Data from analysis
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treatment_rates = [0.589, 0.601, 0.622, 0.606, 0.619, 0.625, 0.621, 0.626, 0.652, 0.642]
control_rates = [0.637, 0.649, 0.657, 0.628, 0.640, 0.651, 0.594, 0.602, 0.626, 0.623]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, treatment_rates, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, control_rates, 's--', color='#ff7f0e', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.67, 'DACA\nImplementation', fontsize=10, color='red', alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Full-time Employment Trends by Treatment Status')
ax.legend(loc='lower right')
ax.set_ylim(0.55, 0.70)
ax.set_xticks(years)

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray')
ax.axvspan(2012.5, 2016, alpha=0.1, color='blue')
ax.text(2008.5, 0.56, 'Pre-DACA', fontsize=10, alpha=0.5)
ax.text(2014, 0.56, 'Post-DACA', fontsize=10, alpha=0.5)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved figure1_trends.png and figure1_trends.pdf")

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("Creating Figure 2: Event Study Plot...")

event_df = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Convert years to relative time (2011 = 0)
event_df['rel_year'] = event_df['Year'] - 2011

# Plot coefficients with confidence intervals
ax.errorbar(event_df['rel_year'], event_df['Coefficient'],
            yerr=[event_df['Coefficient'] - event_df['CI_Low'],
                  event_df['CI_High'] - event_df['Coefficient']],
            fmt='o', color='#1f77b4', markersize=10, capsize=5, capthick=2, linewidth=2)

# Reference line at 0
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Vertical line at treatment
ax.axvline(x=0.5, color='red', linestyle=':', linewidth=2, alpha=0.7)

ax.set_xlabel('Years Relative to DACA (2011 = Reference)')
ax.set_ylabel('DiD Coefficient (Treatment × Year)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment')
ax.set_xticks(event_df['rel_year'])
ax.set_xticklabels(['-5', '-4', '-3', '-2', '-1', '+2', '+3', '+4', '+5'])

# Add annotation
ax.annotate('Post-DACA', xy=(3, 0.08), fontsize=10, alpha=0.7)
ax.annotate('Pre-DACA', xy=(-4, 0.08), fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("  Saved figure2_eventstudy.png and figure2_eventstudy.pdf")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: Difference-in-Differences Diagram...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data points
pre_treat = 0.611
post_treat = 0.634
pre_control = 0.643
post_control = 0.611

# Plot actual trends
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='#1f77b4', linewidth=2.5, markersize=12, label='Treatment (Actual)')
ax.plot([0, 1], [pre_control, post_control], 's-', color='#ff7f0e', linewidth=2.5, markersize=12, label='Control (Actual)')

# Counterfactual for treatment
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'o--', color='#1f77b4', linewidth=2, markersize=8, alpha=0.5, label='Treatment (Counterfactual)')

# DiD effect arrow
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', verticalalignment='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-time Employment')
ax.legend(loc='upper right')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.55, 0.70)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()

print("  Saved figure3_did.png and figure3_did.pdf")

# =============================================================================
# Figure 4: Coefficient Plot
# =============================================================================
print("Creating Figure 4: Regression Coefficient Plot...")

results_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results_df))
colors = ['#1f77b4'] * len(results_df)

ax.barh(y_pos, results_df['Coefficient'], xerr=[results_df['Coefficient'] - results_df['CI_Low'],
                                                  results_df['CI_High'] - results_df['Coefficient']],
        color=colors, alpha=0.7, capsize=5)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('DiD Coefficient (Treatment Effect)')
ax.set_title('DACA Effect Estimates Across Model Specifications')
ax.invert_yaxis()

# Add significance stars
for i, row in results_df.iterrows():
    if row['P_Value'] < 0.001:
        stars = '***'
    elif row['P_Value'] < 0.01:
        stars = '**'
    elif row['P_Value'] < 0.05:
        stars = '*'
    else:
        stars = ''
    ax.text(row['CI_High'] + 0.005, i, stars, va='center', fontsize=12)

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficients.pdf', bbox_inches='tight')
plt.close()

print("  Saved figure4_coefficients.png and figure4_coefficients.pdf")

# =============================================================================
# Figure 5: Subgroup Analysis
# =============================================================================
print("Creating Figure 5: Subgroup Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

subgroups = ['Full Sample', 'Males', 'Females', 'Narrower\nBandwidth']
coefficients = [0.0620, 0.0621, 0.0313, 0.0519]
ci_low = [0.0394, 0.0377, -0.0044, 0.0186]
ci_high = [0.0847, 0.0864, 0.0670, 0.0853]
errors_low = [c - l for c, l in zip(coefficients, ci_low)]
errors_high = [h - c for c, h in zip(coefficients, ci_high)]

y_pos = np.arange(len(subgroups))
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

ax.barh(y_pos, coefficients, xerr=[errors_low, errors_high],
        color=colors, alpha=0.7, capsize=5)

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Coefficient')
ax.set_title('DACA Effect by Subgroup')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figure5_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_subgroups.pdf', bbox_inches='tight')
plt.close()

print("  Saved figure5_subgroups.png and figure5_subgroups.pdf")

print("\nAll figures created successfully!")

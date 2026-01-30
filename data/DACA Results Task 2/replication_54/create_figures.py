"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11

# ============================================================================
# Figure 1: Trends in Full-time Employment by Group
# ============================================================================
print("Creating Figure 1: Trends in Full-time Employment...")

yearly_rates = pd.read_csv('yearly_ft_rates.csv')

# Treatment group
treat_data = yearly_rates[yearly_rates['treat_group'] == 1].sort_values('YEAR')
# Control group
control_data = yearly_rates[yearly_rates['treat_group'] == 0].sort_values('YEAR')

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(treat_data['YEAR'], treat_data['ft_rate'] * 100, 'o-',
        color='#2563eb', linewidth=2.5, markersize=8,
        label='Treatment (Ages 26-30 in 2012)')
ax.plot(control_data['YEAR'], control_data['ft_rate'] * 100, 's--',
        color='#dc2626', linewidth=2.5, markersize=8,
        label='Control (Ages 31-35 in 2012)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1] - 2, 'DACA\nImplemented',
        fontsize=10, color='gray', ha='left')

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate (%)')
ax.set_title('Full-time Employment Trends by DACA Eligibility Group')
ax.legend(loc='lower right')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(55, 75)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Figure 1 saved.")

# ============================================================================
# Figure 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event Study...")

event_df = pd.read_csv('event_study_results.csv')

# Add 2011 as reference year with coefficient = 0
event_df = pd.concat([event_df, pd.DataFrame({'year': [2011], 'coefficient': [0],
                                              'std_error': [0], 'p_value': [1]})],
                     ignore_index=True).sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate confidence intervals
event_df['ci_lower'] = event_df['coefficient'] - 1.96 * event_df['std_error']
event_df['ci_upper'] = event_df['coefficient'] + 1.96 * event_df['std_error']

# Plot coefficients with confidence intervals
ax.errorbar(event_df['year'], event_df['coefficient'],
            yerr=1.96 * event_df['std_error'],
            fmt='o', color='#2563eb', markersize=10, linewidth=2,
            capsize=5, capthick=2)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1] * 0.9, 'DACA', fontsize=10, color='gray', ha='left')

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (relative to 2011)')
ax.set_title('Event Study: Year-Specific Treatment Effects on Full-time Employment')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("  Figure 2 saved.")

# ============================================================================
# Figure 3: Coefficient Comparison Across Models
# ============================================================================
print("Creating Figure 3: Coefficient Comparison...")

results_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate confidence intervals
results_df['ci_lower'] = results_df['Coefficient'] - 1.96 * results_df['Std_Error']
results_df['ci_upper'] = results_df['Coefficient'] + 1.96 * results_df['Std_Error']

y_pos = np.arange(len(results_df))
colors = ['#64748b', '#64748b', '#64748b', '#2563eb', '#64748b']

ax.barh(y_pos, results_df['Coefficient'], xerr=1.96 * results_df['Std_Error'],
        align='center', color=colors, capsize=5, height=0.6)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(results_df['Model'])
ax.set_xlabel('DiD Coefficient (Effect on Full-time Employment)')
ax.set_title('DACA Effect Estimates Across Model Specifications')

# Add text annotations
for i, (coef, se) in enumerate(zip(results_df['Coefficient'], results_df['Std_Error'])):
    ax.text(coef + 1.96 * se + 0.005, i, f'{coef:.3f}\n({se:.3f})',
            va='center', fontsize=9)

# Highlight preferred model
ax.text(0.02, 3.35, '(Preferred)', fontsize=9, color='#2563eb', fontweight='bold')

plt.tight_layout()
plt.savefig('figure3_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficients.pdf', bbox_inches='tight')
plt.close()

print("  Figure 3 saved.")

# ============================================================================
# Figure 4: DiD Visualization (2x2)
# ============================================================================
print("Creating Figure 4: DiD Visualization...")

# Calculate means by group and period
pre_treat = yearly_rates[(yearly_rates['treat_group'] == 1) & (yearly_rates['YEAR'] < 2012)]['ft_rate'].mean() * 100
post_treat = yearly_rates[(yearly_rates['treat_group'] == 1) & (yearly_rates['YEAR'] > 2012)]['ft_rate'].mean() * 100
pre_control = yearly_rates[(yearly_rates['treat_group'] == 0) & (yearly_rates['YEAR'] < 2012)]['ft_rate'].mean() * 100
post_control = yearly_rates[(yearly_rates['treat_group'] == 0) & (yearly_rates['YEAR'] > 2012)]['ft_rate'].mean() * 100

fig, ax = plt.subplots(figsize=(8, 6))

x = [0, 1]
ax.plot(x, [pre_treat, post_treat], 'o-', color='#2563eb', linewidth=3,
        markersize=12, label='Treatment (Ages 26-30)')
ax.plot(x, [pre_control, post_control], 's--', color='#dc2626', linewidth=3,
        markersize=12, label='Control (Ages 31-35)')

# Add counterfactual
counterfactual_post = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual_post], ':', color='#2563eb',
        linewidth=2, alpha=0.5)

# Arrow showing treatment effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.12, (post_treat + counterfactual_post)/2, f'DiD Effect\n≈ {post_treat - counterfactual_post:.1f} pp',
        va='center', fontsize=11, color='green', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-time Employment Rate (%)')
ax.set_title('Difference-in-Differences Design')
ax.legend(loc='lower right')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(58, 72)

plt.tight_layout()
plt.savefig('figure4_did_design.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did_design.pdf', bbox_inches='tight')
plt.close()

print("  Figure 4 saved.")

# ============================================================================
# Figure 5: Heterogeneous Effects by Sex
# ============================================================================
print("Creating Figure 5: Heterogeneous Effects by Sex...")

fig, ax = plt.subplots(figsize=(8, 5))

effects = [0.0621, 0.0313]  # From analysis output
ses = [0.0124, 0.0182]
labels = ['Male', 'Female']
colors = ['#2563eb', '#dc2626']

y_pos = [0, 1]
for i, (effect, se, label, color) in enumerate(zip(effects, ses, labels, colors)):
    ax.barh(i, effect, xerr=1.96*se, color=color, capsize=5, height=0.5, alpha=0.8)
    ax.text(effect + 1.96*se + 0.005, i, f'{effect:.3f}\n(SE: {se:.3f})', va='center', fontsize=10)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('DiD Coefficient')
ax.set_title('Heterogeneous DACA Effects by Sex')
ax.set_xlim(-0.02, 0.12)

plt.tight_layout()
plt.savefig('figure5_by_sex.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_by_sex.pdf', bbox_inches='tight')
plt.close()

print("  Figure 5 saved.")

print("\nAll figures created successfully!")

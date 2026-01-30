"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Helper function for weighted mean
def weighted_mean(df_sub, col, weight_col='PERWT'):
    return np.average(df_sub[col], weights=df_sub[weight_col])

# =============================================================================
# Figure 1: Full-time Employment Trends by Year and Treatment Group
# =============================================================================
print("Creating Figure 1: Employment trends...")

years = sorted(df['YEAR'].unique())
ft_treatment = []
ft_control = []

for year in years:
    ft_treatment.append(weighted_mean(df[(df['YEAR']==year) & (df['ELIGIBLE']==1)], 'FT'))
    ft_control.append(weighted_mean(df[(df['YEAR']==year) & (df['ELIGIBLE']==0)], 'FT'))

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, ft_treatment, 'o-', color='#2166ac', linewidth=2, markersize=8, label='Treatment (Ages 26-30 in 2012)')
ax.plot(years, ft_control, 's--', color='#b2182b', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplementation', xy=(2012, 0.58), fontsize=10, ha='center', color='gray')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.80)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Difference in FT Employment Rate Over Time
# =============================================================================
print("Creating Figure 2: Treatment-control difference...")

difference = [t - c for t, c in zip(ft_treatment, ft_control)]

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#b2182b' if y < 2012 else '#2166ac' for y in years]
ax.bar(years, difference, color=colors, alpha=0.7, edgecolor='black')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Calculate pre and post means
pre_diff = np.mean([d for y, d in zip(years, difference) if y < 2012])
post_diff = np.mean([d for y, d in zip(years, difference) if y > 2012])

ax.axhline(y=pre_diff, color='#b2182b', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axhline(y=post_diff, color='#2166ac', linestyle='--', linewidth=1.5, alpha=0.7)

ax.annotate(f'Pre-DACA avg: {pre_diff:.3f}', xy=(2009.5, pre_diff+0.01), color='#b2182b', fontsize=10)
ax.annotate(f'Post-DACA avg: {post_diff:.3f}', xy=(2014.5, post_diff+0.01), color='#2166ac', fontsize=10)

ax.set_xlabel('Year')
ax.set_ylabel('Difference (Treatment - Control)')
ax.set_title('Difference in Full-Time Employment Rate: Treatment vs. Control')
ax.set_xticks(years)
ax.grid(True, alpha=0.3, axis='y')

# Add legend
red_patch = mpatches.Patch(color='#b2182b', alpha=0.7, label='Pre-DACA')
blue_patch = mpatches.Patch(color='#2166ac', alpha=0.7, label='Post-DACA')
ax.legend(handles=[red_patch, blue_patch], loc='upper left')

plt.tight_layout()
plt.savefig('figure2_difference.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_difference.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_difference.png/pdf")

# =============================================================================
# Figure 3: Event Study Coefficients
# =============================================================================
print("Creating Figure 3: Event study...")

# Event study coefficients from analysis
event_coefs = {
    2008: {'coef': -0.0681, 'se': 0.0351},
    2009: {'coef': -0.0499, 'se': 0.0359},
    2010: {'coef': -0.0821, 'se': 0.0357},
    2011: {'coef': 0.0000, 'se': 0.0000},  # Reference
    2013: {'coef': 0.0158, 'se': 0.0375},
    2014: {'coef': 0.0000, 'se': 0.0384},
    2015: {'coef': 0.0014, 'se': 0.0381},
    2016: {'coef': 0.0741, 'se': 0.0384},
}

years_event = list(event_coefs.keys())
coefs = [event_coefs[y]['coef'] for y in years_event]
ses = [event_coefs[y]['se'] for y in years_event]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
colors = ['#b2182b' if y < 2012 else '#2166ac' for y in years_event]
ax.errorbar(years_event, coefs, yerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o', capsize=5, capthick=2, markersize=10, color='black', ecolor='gray')

# Color the markers
for i, (y, c) in enumerate(zip(years_event, coefs)):
    color = '#b2182b' if y < 2012 else ('#808080' if y == 2011 else '#2166ac')
    ax.scatter(y, c, s=100, c=color, zorder=5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

ax.annotate('DACA Implementation', xy=(2012, 0.12), fontsize=10, ha='center', color='gray')
ax.annotate('Reference Year', xy=(2011, 0.02), fontsize=9, ha='center', color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(years_event)
ax.grid(True, alpha=0.3)

# Legend
red_patch = mpatches.Patch(color='#b2182b', label='Pre-DACA (placebo)')
blue_patch = mpatches.Patch(color='#2166ac', label='Post-DACA (treatment)')
ax.legend(handles=[red_patch, blue_patch], loc='lower right')

plt.tight_layout()
plt.savefig('figure3_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_eventstudy.png/pdf")

# =============================================================================
# Figure 4: DiD Visualization (2x2)
# =============================================================================
print("Creating Figure 4: DiD visualization...")

# Calculate weighted means for 2x2
pre_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT')
post_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT')
pre_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT')
post_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='#2166ac', linewidth=2.5, markersize=12, label='Treatment (Ages 26-30)')
ax.plot([0, 1], [pre_control, post_control], 's--', color='#b2182b', linewidth=2.5, markersize=12, label='Control (Ages 31-35)')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'o:', color='#2166ac', linewidth=1.5, markersize=8, alpha=0.5, label='Treatment counterfactual')

# Arrow showing treatment effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

# Labels
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.60, 0.75)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Add value labels
for x, y, label in [(0, pre_treat, f'{pre_treat:.3f}'), (1, post_treat, f'{post_treat:.3f}'),
                     (0, pre_control, f'{pre_control:.3f}'), (1, post_control, f'{post_control:.3f}')]:
    ax.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_did.png/pdf")

# =============================================================================
# Figure 5: Subgroup Analysis Forest Plot
# =============================================================================
print("Creating Figure 5: Subgroup forest plot...")

# Subgroup results from analysis
subgroups = {
    'Overall': {'coef': 0.0748, 'se': 0.0181, 'n': 17382},
    'Male': {'coef': 0.0716, 'se': 0.0199, 'n': 9075},
    'Female': {'coef': 0.0527, 'se': 0.0281, 'n': 8307},
    'Married': {'coef': 0.0573, 'se': 0.0255, 'n': 8524},
    'Unmarried': {'coef': 0.0981, 'se': 0.0260, 'n': 8858},
}

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = list(range(len(subgroups)))
labels = list(subgroups.keys())
coefs = [subgroups[s]['coef'] for s in labels]
ses = [subgroups[s]['se'] for s in labels]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.errorbar(coefs, y_pos, xerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o', capsize=5, capthick=2, markersize=10, color='#2166ac', ecolor='gray')

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)')
ax.set_title('DACA Effect by Subgroup')
ax.grid(True, alpha=0.3, axis='x')

# Add coefficient labels
for i, (c, low, high) in enumerate(zip(coefs, ci_low, ci_high)):
    ax.text(high + 0.01, i, f'{c:.3f} [{low:.3f}, {high:.3f}]', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure5_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_subgroups.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure5_subgroups.png/pdf")

# =============================================================================
# Figure 6: Model Comparison
# =============================================================================
print("Creating Figure 6: Model comparison...")

models = {
    'M1: Basic OLS': {'coef': 0.0643, 'se': 0.0153},
    'M4: WLS (robust)': {'coef': 0.0748, 'se': 0.0181},
    'M5: Demographics': {'coef': 0.0648, 'se': 0.0168},
    'M7: Year FE': {'coef': 0.0721, 'se': 0.0181},
    'M9: State FE': {'coef': 0.0737, 'se': 0.0180},
    'M10: Full Model': {'coef': 0.0614, 'se': 0.0167},
}

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = list(range(len(models)))
labels = list(models.keys())
coefs = [models[m]['coef'] for m in labels]
ses = [models[m]['se'] for m in labels]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

colors = ['#2166ac' if i != len(models)-1 else '#b2182b' for i in range(len(models))]
ax.errorbar(coefs, y_pos, xerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='none', capsize=5, capthick=2, ecolor='gray')

for i, (c, col) in enumerate(zip(coefs, colors)):
    ax.scatter(c, i, s=100, c=col, zorder=5)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('DiD Coefficient')
ax.set_title('DACA Effect Estimates Across Model Specifications')
ax.grid(True, alpha=0.3, axis='x')

# Highlight preferred model
ax.annotate('Preferred', xy=(coefs[-1], y_pos[-1]), xytext=(coefs[-1]+0.03, y_pos[-1]+0.3),
            fontsize=10, color='#b2182b', arrowprops=dict(arrowstyle='->', color='#b2182b'))

plt.tight_layout()
plt.savefig('figure6_models.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_models.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure6_models.png/pdf")

print("\nAll figures created successfully!")

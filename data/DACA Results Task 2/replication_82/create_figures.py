"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Load results
yearly_rates = pd.read_csv('yearly_fulltime_rates.csv', index_col=0)
event_study = pd.read_csv('event_study_results.csv')

# =============================================================================
# FIGURE 1: Parallel Trends Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_rates.index.values
control = yearly_rates['Control'].values
treatment = yearly_rates['Treatment'].values

ax.plot(years, control, 'o-', color='blue', linewidth=2, markersize=8, label='Control (Ages 31-35)')
ax.plot(years, treatment, 's-', color='red', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation (2012)')

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray', label='Pre-Period')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green', label='Post-Period')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.55, 0.70)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add annotation
ax.annotate('DACA\nImplemented', xy=(2012, 0.56), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved: parallel trends plot")

# =============================================================================
# FIGURE 2: Event Study Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Add 2011 as reference year with coefficient 0
event_data = event_study.copy()
ref_year = pd.DataFrame({'year': [2011], 'coefficient': [0], 'se': [0], 'ci_low': [0], 'ci_high': [0]})
event_data = pd.concat([event_data, ref_year]).sort_values('year')

years_event = event_data['year'].values
coefs = event_data['coefficient'].values
ci_low = event_data['ci_low'].values
ci_high = event_data['ci_high'].values

# Plot coefficients with confidence intervals
ax.errorbar(years_event, coefs,
            yerr=[coefs - ci_low, ci_high - coefs],
            fmt='o', color='darkblue', markersize=8, capsize=4, capthick=2, linewidth=2)

# Connect points with line
ax.plot(years_event, coefs, '-', color='darkblue', alpha=0.5, linewidth=1)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add vertical line at DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')

# Shade regions
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='gray')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add annotations
ax.annotate('Pre-DACA', xy=(2008.5, ax.get_ylim()[1]-0.01), fontsize=10, ha='center')
ax.annotate('Post-DACA', xy=(2014.5, ax.get_ylim()[1]-0.01), fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved: event study plot")

# =============================================================================
# FIGURE 3: DiD Visualization (2x2)
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Data from analysis
ft_treat_pre = 0.6147
ft_treat_post = 0.6339
ft_ctrl_pre = 0.6461
ft_ctrl_post = 0.6136

# Positions
x_pre = 0
x_post = 1

# Plot control group
ax.plot([x_pre, x_post], [ft_ctrl_pre, ft_ctrl_post], 'o-', color='blue',
        linewidth=2.5, markersize=12, label='Control (Ages 31-35)')

# Plot treatment group
ax.plot([x_pre, x_post], [ft_treat_pre, ft_treat_post], 's-', color='red',
        linewidth=2.5, markersize=12, label='Treatment (Ages 26-30)')

# Plot counterfactual for treatment
counterfactual = ft_treat_pre + (ft_ctrl_post - ft_ctrl_pre)
ax.plot([x_pre, x_post], [ft_treat_pre, counterfactual], 's--', color='red',
        linewidth=1.5, markersize=8, alpha=0.5, label='Treatment Counterfactual')

# Draw arrow showing DiD effect
ax.annotate('', xy=(x_post + 0.05, ft_treat_post),
            xytext=(x_post + 0.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(x_post + 0.1, (ft_treat_post + counterfactual) / 2,
        f'DiD\n= {ft_treat_post - counterfactual:.3f}', fontsize=11, va='center', color='green')

# Labels and formatting
ax.set_xticks([x_pre, x_post])
ax.set_xticklabels(['Pre-Period\n(2006-2011)', 'Post-Period\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.55, 0.68)

# Add value labels
for x, y, label in [(x_pre, ft_ctrl_pre, f'{ft_ctrl_pre:.3f}'),
                     (x_post, ft_ctrl_post, f'{ft_ctrl_post:.3f}'),
                     (x_pre, ft_treat_pre, f'{ft_treat_pre:.3f}'),
                     (x_post, ft_treat_post, f'{ft_treat_post:.3f}')]:
    ax.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved: DiD visualization")

# =============================================================================
# FIGURE 4: Coefficient Comparison Across Models
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

models = ['Model 1\n(Basic)', 'Model 2\n(+Controls)', 'Model 3\n(+Year FE)',
          'Model 4\n(+Controls\n+Year FE)', 'Model 5\n(+State FE)', 'Model 6\n(Weighted)']
coefficients = [0.0516, 0.0451, 0.0515, 0.0450, 0.0442, 0.0459]
std_errors = [0.0100, 0.0092, 0.0099, 0.0092, 0.0092, 0.0107]

x_pos = np.arange(len(models))
ci_low = [c - 1.96*se for c, se in zip(coefficients, std_errors)]
ci_high = [c + 1.96*se for c, se in zip(coefficients, std_errors)]

ax.bar(x_pos, coefficients, color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=1.5)
ax.errorbar(x_pos, coefficients, yerr=[np.array(coefficients) - np.array(ci_low),
                                        np.array(ci_high) - np.array(coefficients)],
            fmt='none', color='black', capsize=5, capthick=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DiD Estimates Across Model Specifications', fontsize=14)

# Add coefficient labels on bars
for i, (coef, se) in enumerate(zip(coefficients, std_errors)):
    ax.text(i, coef + 0.01, f'{coef:.3f}\n({se:.3f})', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved: model comparison")

# =============================================================================
# FIGURE 5: Robustness Checks
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

robustness_tests = [
    ('Main Estimate', 0.0450, 0.0092),
    ('Males Only', 0.0332, 0.0112),
    ('Females Only', 0.0475, 0.0149),
    ('Narrow Age Band\n(27-29 vs 32-34)', 0.0412, 0.0119),
    ('Wide Age Band\n(25-30 vs 31-36)', 0.0522, 0.0084),
    ('Placebo Test\n(Pre-period: 2009-2011)', 0.0032, 0.0108)
]

labels = [r[0] for r in robustness_tests]
coefs = [r[1] for r in robustness_tests]
ses = [r[2] for r in robustness_tests]

y_pos = np.arange(len(labels))

# Plot coefficients with confidence intervals
ax.errorbar(coefs, y_pos, xerr=[1.96*s for s in ses], fmt='o',
            color='darkblue', markersize=10, capsize=5, capthick=2)

# Add vertical line at 0
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Highlight main estimate
ax.axhspan(-0.5, 0.5, alpha=0.2, color='yellow')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel('DiD Coefficient', fontsize=12)
ax.set_title('Robustness Checks: DiD Estimates with 95% Confidence Intervals', fontsize=14)

# Add coefficient labels
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.text(coef + 0.02, i, f'{coef:.3f} ({se:.3f})', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure5_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_robustness.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved: robustness checks")

print("\nAll figures created successfully!")

"""
DACA Replication Study - Figure Generation Script
Replication 45
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json

# Set style for publication-quality figures
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print("Loading results...")

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

yearly_means = pd.read_csv('yearly_means.csv', header=[0,1], index_col=[0,1])

# =============================================================================
# Figure 1: Parallel Trends / Event Study
# =============================================================================
print("Creating Figure 1: Event Study...")

years = [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]
coefs = [results['event_study'][str(y)]['coef'] for y in years]
ses = [results['event_study'][str(y)]['se'] for y in years]

# Add reference year (2011)
years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_plot = coefs[:5] + [0] + coefs[5:]
ses_plot = ses[:5] + [0] + ses[5:]

fig, ax = plt.subplots(figsize=(12, 7))

# Plot point estimates with confidence intervals
ci_low = [c - 1.96*s for c, s in zip(coefs_plot, ses_plot)]
ci_high = [c + 1.96*s for c, s in zip(coefs_plot, ses_plot)]

ax.errorbar(years_plot, coefs_plot, yerr=[np.array(coefs_plot)-np.array(ci_low),
                                           np.array(ci_high)-np.array(coefs_plot)],
            fmt='o', markersize=8, capsize=5, capthick=2, color='#2166AC',
            ecolor='#67A9CF', linewidth=2)

# Reference line at zero
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

# Vertical line at DACA implementation
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation (2012)')

# Shading for pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA Period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)')
ax.set_xticks(years_plot)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure1_event_study.png/pdf")

# =============================================================================
# Figure 2: Treatment and Control Trends
# =============================================================================
print("Creating Figure 2: Treatment and Control Trends...")

# Parse yearly means - reset index to get YEAR and treatment as columns
yearly_means = yearly_means.reset_index()
yearly_means.columns = ['YEAR', 'treatment', 'ft_mean', 'ft_std', 'ft_count', 'perwt_sum']
yearly_means['YEAR'] = yearly_means['YEAR'].astype(int)
yearly_means['treatment'] = yearly_means['treatment'].astype(int)

treat = yearly_means[yearly_means['treatment'] == 1].sort_values('YEAR')
ctrl = yearly_means[yearly_means['treatment'] == 0].sort_values('YEAR')

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(treat['YEAR'], treat['ft_mean'], 'o-', markersize=8, linewidth=2,
        color='#D53E4F', label='Treatment (Ages 26-30 at DACA)')
ax.plot(ctrl['YEAR'], ctrl['ft_mean'], 's-', markersize=8, linewidth=2,
        color='#3288BD', label='Control (Ages 31-35 at DACA)')

# Vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7,
           label='DACA Implementation (June 2012)')

# Shading
ax.axvspan(2005.5, 2011.5, alpha=0.05, color='blue')
ax.axvspan(2012.5, 2016.5, alpha=0.05, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.set_xticks(range(2006, 2017))
ax.set_ylim([0.55, 0.75])
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Add annotations
ax.annotate('Pre-DACA', xy=(2008.5, 0.56), fontsize=11, ha='center', style='italic')
ax.annotate('Post-DACA', xy=(2014.5, 0.56), fontsize=11, ha='center', style='italic')

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure2_trends.png/pdf")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(10, 7))

# Calculate means by period
treat_pre = treat[treat['YEAR'] <= 2011]['ft_mean'].mean()
treat_post = treat[treat['YEAR'] >= 2013]['ft_mean'].mean()
ctrl_pre = ctrl[ctrl['YEAR'] <= 2011]['ft_mean'].mean()
ctrl_post = ctrl[ctrl['YEAR'] >= 2013]['ft_mean'].mean()

# Bar positions
x = np.array([1, 2, 4, 5])
heights = [ctrl_pre, ctrl_post, treat_pre, treat_post]
colors = ['#3288BD', '#3288BD', '#D53E4F', '#D53E4F']
hatches = ['', '///', '', '///']

bars = ax.bar(x, heights, color=colors, edgecolor='black', linewidth=1.5, width=0.7)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

ax.set_xticks([1.5, 4.5])
ax.set_xticklabels(['Control Group\n(Ages 31-35)', 'Treatment Group\n(Ages 26-30)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Full-Time Employment Before and After DACA')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='white', edgecolor='black', label='Pre-DACA (2006-2011)'),
                   Patch(facecolor='white', edgecolor='black', hatch='///', label='Post-DACA (2013-2016)')]
ax.legend(handles=legend_elements, loc='upper right')

# Add value labels
for i, (x_pos, height) in enumerate(zip(x, heights)):
    ax.text(x_pos, height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=11)

# Add DiD annotation
did_estimate = results['model5']['coef']
ax.annotate(f'DiD Estimate: {did_estimate:.4f}***\n(SE: {results["model5"]["se"]:.4f})',
            xy=(5.5, 0.62), fontsize=12, ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.set_ylim([0.55, 0.72])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure3_did.png/pdf")

# =============================================================================
# Figure 4: Regression Coefficient Comparison
# =============================================================================
print("Creating Figure 4: Coefficient Comparison...")

fig, ax = plt.subplots(figsize=(10, 7))

models = ['Model 1\n(Basic)', 'Model 2\n(+Controls)', 'Model 3\n(+State FE)',
          'Model 4\n(+Year FE)', 'Model 5\n(Clustered SE)', 'Weighted\n(PERWT)']
coefs = [results['model1']['coef'], results['model2']['coef'], results['model3']['coef'],
         results['model4']['coef'], results['model5']['coef'], results['model_weighted']['coef']]
ses = [results['model1']['se'], results['model2']['se'], results['model3']['se'],
       results['model4']['se'], results['model5']['se'], results['model_weighted']['se']]

x_pos = np.arange(len(models))
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.errorbar(x_pos, coefs, yerr=[np.array(coefs)-np.array(ci_low),
                                  np.array(ci_high)-np.array(coefs)],
            fmt='o', markersize=10, capsize=6, capthick=2, color='#2166AC',
            ecolor='#67A9CF', linewidth=2)

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=results['model5']['coef'], color='red', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Preferred Estimate: {results["model5"]["coef"]:.4f}')

ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.set_ylabel('DiD Coefficient (Treatment Effect)')
ax.set_title('DACA Effect on Full-Time Employment: Robustness Across Specifications')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure4_robustness.png/pdf")

# =============================================================================
# Figure 5: Subgroup Analysis
# =============================================================================
print("Creating Figure 5: Subgroup Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

subgroups = ['All', 'Male', 'Female']
coefs_sub = [results['model5']['coef'],
             results['subgroup_male']['coef'],
             results['subgroup_female']['coef']]
ses_sub = [results['model5']['se'],
           results['subgroup_male']['se'],
           results['subgroup_female']['se']]

x_pos = np.arange(len(subgroups))
ci_low = [c - 1.96*s for c, s in zip(coefs_sub, ses_sub)]
ci_high = [c + 1.96*s for c, s in zip(coefs_sub, ses_sub)]

colors = ['#2166AC', '#4393C3', '#92C5DE']
ax.barh(x_pos, coefs_sub, xerr=[np.array(coefs_sub)-np.array(ci_low),
                                  np.array(ci_high)-np.array(coefs_sub)],
        color=colors, capsize=5, edgecolor='black', linewidth=1.5, height=0.5)

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_yticks(x_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Coefficient (Treatment Effect)')
ax.set_title('DACA Effect on Full-Time Employment by Subgroup')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (coef, se) in enumerate(zip(coefs_sub, ses_sub)):
    ax.text(coef + 0.005, i, f'{coef:.4f}\n(SE: {se:.4f})', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_subgroups.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure5_subgroups.png/pdf")

print("\nAll figures created successfully!")

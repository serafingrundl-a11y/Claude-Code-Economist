"""
Generate figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'].isin([1, 2])).astype(int)

# Figure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# ============================================================================
# FIGURE 1: Full-time employment rates by group and time period
# ============================================================================

def weighted_mean(data, col, weight):
    return np.average(data[col], weights=data[weight])

# Calculate weighted means by year and treatment status
yearly_rates = []
for year in sorted(df['YEAR'].unique()):
    for elig in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==elig)]
        rate = weighted_mean(subset, 'FT', 'PERWT')
        yearly_rates.append({
            'Year': year,
            'Eligible': elig,
            'FT_Rate': rate,
            'N': len(subset)
        })

yearly_df = pd.DataFrame(yearly_rates)

fig, ax = plt.subplots(figsize=(10, 6))

treat = yearly_df[yearly_df['Eligible']==1]
control = yearly_df[yearly_df['Eligible']==0]

ax.plot(treat['Year'], treat['FT_Rate'], 'o-', color='#2E86AB', linewidth=2,
        markersize=8, label='Treatment (ages 26-30)', zorder=5)
ax.plot(control['Year'], control['FT_Rate'], 's--', color='#A23B72', linewidth=2,
        markersize=8, label='Control (ages 31-35)', zorder=5)

# Add vertical line at 2012
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplementation\n(June 2012)', xy=(2012, 0.58), fontsize=9,
            ha='center', color='gray')

# Add shaded regions
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (Weighted)')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_trends.png/pdf")

# ============================================================================
# FIGURE 2: Difference-in-Differences visualization (2x2)
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

# Get weighted means for 2x2 table
pre_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')
post_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')
pre_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
post_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')

x = [0, 1]
labels = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']

ax.plot(x, [pre_treat, post_treat], 'o-', color='#2E86AB', linewidth=3,
        markersize=12, label='Treatment (ages 26-30)')
ax.plot(x, [pre_control, post_control], 's--', color='#A23B72', linewidth=3,
        markersize=12, label='Control (ages 31-35)')

# Counterfactual line
counterfactual_post = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual_post], ':', color='#2E86AB', linewidth=2,
        alpha=0.5, label='Counterfactual (treatment)')

# DiD arrow
did = (post_treat - pre_treat) - (post_control - pre_control)
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.12, (post_treat + counterfactual_post)/2, f'DiD = {did:.3f}',
        fontsize=11, color='green', va='center', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Full-Time Employment Rate (Weighted)')
ax.set_title('Difference-in-Differences Visualization')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.60, 0.72)
ax.legend(loc='upper left')

# Add annotations for values
ax.annotate(f'{pre_treat:.3f}', xy=(0, pre_treat), xytext=(-0.1, pre_treat-0.015),
            fontsize=9, color='#2E86AB')
ax.annotate(f'{post_treat:.3f}', xy=(1, post_treat), xytext=(0.85, post_treat+0.01),
            fontsize=9, color='#2E86AB')
ax.annotate(f'{pre_control:.3f}', xy=(0, pre_control), xytext=(-0.1, pre_control+0.01),
            fontsize=9, color='#A23B72')
ax.annotate(f'{post_control:.3f}', xy=(1, post_control), xytext=(0.85, post_control-0.015),
            fontsize=9, color='#A23B72')

plt.tight_layout()
plt.savefig('figure2_did.png', dpi=150, bbox_inches='tight')
plt.savefig('figure2_did.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_did.png/pdf")

# ============================================================================
# FIGURE 3: Event Study Coefficients
# ============================================================================

import statsmodels.formula.api as smf

# Create year dummies and interactions
for yr in [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{yr}'] = (df['YEAR'] == yr).astype(int)
    df[f'ELIG_X_{yr}'] = df['ELIGIBLE'] * df[f'YEAR_{yr}']

model_event = smf.wls('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + '
                      'ELIG_X_2008 + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2013 + ELIG_X_2014 + ELIG_X_2015 + ELIG_X_2016',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract coefficients and CIs
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [model_event.params.get(f'ELIG_X_{yr}', 0) for yr in years]
coefs[3] = 0  # 2011 is reference
ses = [model_event.bse.get(f'ELIG_X_{yr}', 0) for yr in years]
ses[3] = 0
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

fig, ax = plt.subplots(figsize=(10, 6))

ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Error bars
ax.errorbar(years, coefs, yerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o', color='#2E86AB', markersize=8, capsize=5, capthick=2, linewidth=2, zorder=5)

# Connect points
ax.plot(years, coefs, '-', color='#2E86AB', linewidth=1, alpha=0.5, zorder=4)

ax.set_xlabel('Year')
ax.set_ylabel('DiD Coefficient (relative to 2011)')
ax.set_title('Event Study: Year-Specific Treatment Effects')
ax.set_xticks(years)

# Annotations
ax.annotate('Pre-DACA', xy=(2009.5, 0.08), fontsize=10, ha='center', color='gray')
ax.annotate('Post-DACA', xy=(2014.5, 0.08), fontsize=10, ha='center', color='gray')
ax.annotate('Reference\nYear (2011)', xy=(2011, 0.005), fontsize=9, ha='center', color='gray')

ax.set_ylim(-0.15, 0.15)

plt.tight_layout()
plt.savefig('figure3_eventstudy.png', dpi=150, bbox_inches='tight')
plt.savefig('figure3_eventstudy.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_eventstudy.png/pdf")

# ============================================================================
# FIGURE 4: Distribution of Key Variables by Treatment Status
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Age distribution
ax = axes[0, 0]
treat_ages = df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012']
control_ages = df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012']
ax.hist(treat_ages, bins=20, alpha=0.6, color='#2E86AB', label='Treatment', density=True)
ax.hist(control_ages, bins=20, alpha=0.6, color='#A23B72', label='Control', density=True)
ax.axvline(x=30.5, color='gray', linestyle='--', linewidth=2)
ax.set_xlabel('Age in June 2012')
ax.set_ylabel('Density')
ax.set_title('A. Age Distribution')
ax.legend()

# Panel B: Education distribution
ax = axes[0, 1]
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
treat_educ = df[df['ELIGIBLE']==1]['EDUC_RECODE'].value_counts(normalize=True).reindex(educ_order).fillna(0)
control_educ = df[df['ELIGIBLE']==0]['EDUC_RECODE'].value_counts(normalize=True).reindex(educ_order).fillna(0)
x = np.arange(len(educ_order))
width = 0.35
ax.bar(x - width/2, treat_educ.values, width, color='#2E86AB', alpha=0.7, label='Treatment')
ax.bar(x + width/2, control_educ.values, width, color='#A23B72', alpha=0.7, label='Control')
ax.set_xlabel('Education Level')
ax.set_ylabel('Proportion')
ax.set_title('B. Education Distribution')
ax.set_xticks(x)
ax.set_xticklabels(['<HS', 'HS', 'Some\nCollege', '2-Year', 'BA+'])
ax.legend()

# Panel C: Gender distribution
ax = axes[1, 0]
treat_sex = df[df['ELIGIBLE']==1]['SEX'].value_counts(normalize=True).sort_index()
control_sex = df[df['ELIGIBLE']==0]['SEX'].value_counts(normalize=True).sort_index()
x = np.arange(2)
ax.bar(x - width/2, treat_sex.values, width, color='#2E86AB', alpha=0.7, label='Treatment')
ax.bar(x + width/2, control_sex.values, width, color='#A23B72', alpha=0.7, label='Control')
ax.set_xlabel('Sex')
ax.set_ylabel('Proportion')
ax.set_title('C. Gender Distribution')
ax.set_xticks(x)
ax.set_xticklabels(['Male', 'Female'])
ax.legend()

# Panel D: FT rates by group
ax = axes[1, 1]
groups = ['Control\nPre', 'Control\nPost', 'Treatment\nPre', 'Treatment\nPost']
rates = [pre_control, post_control, pre_treat, post_treat]
colors = ['#A23B72', '#A23B72', '#2E86AB', '#2E86AB']
alphas = [0.5, 0.9, 0.5, 0.9]
bars = []
for i in range(4):
    bar = ax.bar(i, rates[i], color=colors[i], alpha=alphas[i])
    bars.append(bar[0])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('D. Full-Time Employment by Group and Period')
ax.set_xticks(range(4))
ax.set_xticklabels(groups)
ax.set_ylim(0.55, 0.75)

# Add value labels
for i, (bar, rate) in enumerate(zip(bars, rates)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{rate:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_distributions.png', dpi=150, bbox_inches='tight')
plt.savefig('figure4_distributions.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved: figure4_distributions.png/pdf")

# ============================================================================
# FIGURE 5: Robustness - Coefficient plot
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

# Model results
model_names = [
    'Basic DiD (unweighted)',
    'Basic DiD (weighted)',
    'With controls (unweighted)',
    'With controls (weighted)',
    '+ State FE',
    '+ Year & State FE',
    'Males only',
    'Females only',
    'Narrow bandwidth'
]

coefs = [0.0643, 0.0748, 0.0559, 0.0646, 0.0642, 0.0613, 0.0716, 0.0527, 0.0732]
ses = [0.0153, 0.0181, 0.0142, 0.0167, 0.0167, 0.0166, 0.0199, 0.0281, 0.0223]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

y_pos = np.arange(len(model_names))

ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

# Error bars
ax.errorbar(coefs, y_pos, xerr=[[c-l for c, l in zip(coefs, ci_low)],
                                 [h-c for c, h in zip(coefs, ci_high)]],
            fmt='o', color='#2E86AB', markersize=8, capsize=5, capthick=2, linewidth=2)

# Highlight preferred specification
ax.scatter([coefs[3]], [y_pos[3]], s=200, color='green', marker='o', zorder=10,
           label='Preferred specification')

ax.set_yticks(y_pos)
ax.set_yticklabels(model_names)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)')
ax.set_title('Robustness: Comparison of DiD Estimates Across Specifications')
ax.set_xlim(-0.02, 0.14)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure5_robustness.png', dpi=150, bbox_inches='tight')
plt.savefig('figure5_robustness.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved: figure5_robustness.png/pdf")

print("\nAll figures generated successfully!")

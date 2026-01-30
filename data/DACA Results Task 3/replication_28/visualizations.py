"""
DACA Replication Study - Visualization Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create derived variables
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)

print("Generating visualizations...")

# =============================================================================
# FIGURE 1: Full-Time Employment Trends by Treatment Status
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate weighted means by year and treatment status
def weighted_mean_group(group):
    return np.average(group['FT'], weights=group['PERWT'])

yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean_group).unstack()
yearly_ft.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

# Plot
yearly_ft.plot(ax=ax1, marker='o', linewidth=2, markersize=8)
ax1.axvline(x=2012.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax1.set_xlabel('Year')
ax1.set_ylabel('Full-Time Employment Rate')
ax1.set_title('Full-Time Employment Trends by Treatment Status')
ax1.legend(loc='best')
ax1.set_ylim(0.5, 0.8)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1: Trends saved")

# =============================================================================
# FIGURE 2: Event Study Plot
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Load event study results
event_df = pd.read_csv('event_study_results.csv')

# Add reference year (2011) with zero coefficient
ref_row = pd.DataFrame({'year': [2011], 'coef': [0], 'se': [0], 'ci_low': [0], 'ci_high': [0]})
event_df = pd.concat([event_df, ref_row], ignore_index=True)
event_df = event_df.sort_values('year')

# Plot
ax2.errorbar(event_df['year'], event_df['coef'],
             yerr=[event_df['coef'] - event_df['ci_low'], event_df['ci_high'] - event_df['coef']],
             fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10, color='navy')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax2.fill_between([2012, 2016.5], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='green', label='Post-treatment')
ax2.set_xlabel('Year')
ax2.set_ylabel('Coefficient (Relative to 2011)')
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax2.legend(loc='upper left')
ax2.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax2.set_xlim(2007.5, 2016.5)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2: Event study saved")

# =============================================================================
# FIGURE 3: DiD Visualization (2x2 Design)
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate weighted means for DiD
ft_did = df.groupby(['ELIGIBLE', 'AFTER']).apply(weighted_mean_group).unstack()
ft_did.index = ['Control', 'Treatment']
ft_did.columns = ['Pre-DACA', 'Post-DACA']

# Plot bars
x = np.arange(2)
width = 0.35

bars1 = ax3.bar(x - width/2, ft_did['Pre-DACA'], width, label='Pre-DACA (2008-2011)', color='steelblue')
bars2 = ax3.bar(x + width/2, ft_did['Post-DACA'], width, label='Post-DACA (2013-2016)', color='coral')

ax3.set_xlabel('Group')
ax3.set_ylabel('Full-Time Employment Rate')
ax3.set_title('Difference-in-Differences Design: DACA Effect on Full-Time Employment')
ax3.set_xticks(x)
ax3.set_xticklabels(['Control (Ages 31-35)', 'Treatment (Ages 26-30)'])
ax3.legend()
ax3.set_ylim(0.5, 0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()
print("Figure 3: DiD visualization saved")

# =============================================================================
# FIGURE 4: Heterogeneity Analysis
# =============================================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

# By Sex
het_sex = pd.read_csv('heterogeneity_sex.csv')
ax4a.barh(het_sex['group'], het_sex['coef'], xerr=het_sex['se']*1.96, color=['steelblue', 'coral'], capsize=5)
ax4a.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4a.set_xlabel('DiD Coefficient (95% CI)')
ax4a.set_title('Effect by Sex')
ax4a.set_xlim(-0.05, 0.15)

# By Education
het_edu = pd.read_csv('heterogeneity_education.csv')
colors = ['lightblue', 'steelblue', 'coral']
ax4b.barh(het_edu['group'], het_edu['coef'], xerr=het_edu['se']*1.96, color=colors[:len(het_edu)], capsize=5)
ax4b.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4b.set_xlabel('DiD Coefficient (95% CI)')
ax4b.set_title('Effect by Education Level')
ax4b.set_xlim(-0.05, 0.35)

plt.tight_layout()
plt.savefig('figure4_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_heterogeneity.pdf', bbox_inches='tight')
plt.close()
print("Figure 4: Heterogeneity analysis saved")

# =============================================================================
# FIGURE 5: Model Comparison
# =============================================================================
fig5, ax5 = plt.subplots(figsize=(10, 6))

model_comp = pd.read_csv('model_comparison.csv')
y_pos = np.arange(len(model_comp))

ax5.barh(y_pos, model_comp['DiD Coefficient'], xerr=model_comp['Std Error']*1.96,
         color='steelblue', capsize=5, alpha=0.8)
ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(model_comp['Model'])
ax5.set_xlabel('DiD Coefficient (95% CI)')
ax5.set_title('Comparison of Model Specifications')
ax5.invert_yaxis()

plt.tight_layout()
plt.savefig('figure5_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("Figure 5: Model comparison saved")

# =============================================================================
# Generate Summary Statistics Table
# =============================================================================
print("\nGenerating summary statistics...")

# Pre-DACA period statistics
pre_daca = df[df['AFTER'] == 0]

summary_stats = []
for var in ['AGE', 'female', 'married', 'NCHILD', 'FT']:
    if var in df.columns or var in ['female', 'married']:
        treat_mean = pre_daca[pre_daca['ELIGIBLE'] == 1][var].mean()
        treat_sd = pre_daca[pre_daca['ELIGIBLE'] == 1][var].std()
        control_mean = pre_daca[pre_daca['ELIGIBLE'] == 0][var].mean()
        control_sd = pre_daca[pre_daca['ELIGIBLE'] == 0][var].std()

        summary_stats.append({
            'Variable': var,
            'Treatment Mean': treat_mean,
            'Treatment SD': treat_sd,
            'Control Mean': control_mean,
            'Control SD': control_sd,
            'Difference': treat_mean - control_mean
        })

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv('summary_statistics.csv', index=False)
print("Summary statistics saved to summary_statistics.csv")

# Sample size by year and group
sample_sizes = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_sizes.columns = ['Control', 'Treatment']
sample_sizes.to_csv('sample_sizes_by_year.csv')
print("Sample sizes saved to sample_sizes_by_year.csv")

print("\nAll visualizations generated successfully!")

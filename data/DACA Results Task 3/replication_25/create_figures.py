"""
DACA Replication Study - Figure Generation Script
Creates visualizations for the replication report
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
plt.rcParams['figure.figsize'] = (8, 6)

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# =============================================================================
# Figure 1: Full-Time Employment Rates by Year and Group
# =============================================================================
print("Creating Figure 1: Parallel Trends Plot...")

# Calculate yearly means (unweighted)
yearly = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

years = yearly.index.values
control = yearly['Control (Ages 31-35)'].values
treatment = yearly['Treatment (Ages 26-30)'].values

# Plot lines
ax.plot(years, control, 'o-', color='#2166ac', linewidth=2, markersize=8, label='Control (Ages 31-35)')
ax.plot(years, treatment, 's--', color='#b2182b', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.73, 'DACA Implementation\n(June 2012)', fontsize=10, color='gray', va='top')

# Add shaded regions
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA Period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Year and Treatment Group')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.80)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# =============================================================================
# Figure 2: DiD Illustration (2x2 Table Visual)
# =============================================================================
print("Creating Figure 2: DiD Illustration...")

# Calculate means for 2x2 table
ft_rates = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 7))

# Create bar positions
x = np.array([0, 1])
width = 0.35

control_pre = ft_rates.loc[0, 0]
control_post = ft_rates.loc[0, 1]
treat_pre = ft_rates.loc[1, 0]
treat_post = ft_rates.loc[1, 1]

bars1 = ax.bar(x - width/2, [control_pre, treat_pre], width, label='Pre-DACA (2008-2011)', color='#2166ac', alpha=0.8)
bars2 = ax.bar(x + width/2, [control_post, treat_post], width, label='Post-DACA (2013-2016)', color='#b2182b', alpha=0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Full-Time Employment Rates')
ax.set_xticks(x)
ax.set_xticklabels(['Control Group\n(Ages 31-35)', 'Treatment Group\n(Ages 26-30)'])
ax.legend(loc='upper right')
ax.set_ylim(0, 0.85)

# Add DiD calculation annotation
did = (treat_post - treat_pre) - (control_post - control_pre)
textstr = f'DiD Estimate: {did:.4f}\n\nTreatment change: {treat_post - treat_pre:.4f}\nControl change: {control_post - control_pre:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.98, 0.35, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('figure2_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# =============================================================================
# Figure 3: Coefficient Plot
# =============================================================================
print("Creating Figure 3: Coefficient Plot...")

# Results from analysis
models = ['(1) Basic', '(2) Weighted', '(3) +Covariates', '(4) Weighted+Cov',
          '(5) Clustered', '(6) Clust+Cov', '(7) Year FE']
estimates = [0.0643, 0.0748, 0.0536, 0.0625, 0.0643, 0.0536, 0.0629]
ci_lower = [0.0344, 0.0393, 0.0258, 0.0297, 0.0366, 0.0246, 0.0330]
ci_upper = [0.0941, 0.1102, 0.0813, 0.0953, 0.0919, 0.0825, 0.0928]

fig, ax = plt.subplots(figsize=(10, 7))

y_pos = np.arange(len(models))
errors = [[est - lo for est, lo in zip(estimates, ci_lower)],
          [hi - est for est, hi in zip(estimates, ci_upper)]]

ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', color='#2166ac',
            capsize=5, capthick=2, markersize=10, linewidth=2)

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=estimates[1], color='#b2182b', linestyle=':', linewidth=2, alpha=0.5)  # Preferred estimate

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Treatment Effect (DACA Eligibility on Full-Time Employment)')
ax.set_title('Difference-in-Differences Estimates Across Specifications')
ax.set_xlim(-0.02, 0.15)

# Add annotation for preferred estimate
ax.annotate('Preferred\nEstimate', xy=(estimates[1], 1), xytext=(0.12, 1.5),
            arrowprops=dict(arrowstyle='->', color='#b2182b'),
            fontsize=10, color='#b2182b')

plt.tight_layout()
plt.savefig('figure3_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# =============================================================================
# Figure 4: Distribution of Sample by Year
# =============================================================================
print("Creating Figure 4: Sample Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Sample size by year and group
sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control', 'Treatment']

sample_by_year.plot(kind='bar', ax=ax1, color=['#2166ac', '#b2182b'], alpha=0.8)
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Observations')
ax1.set_title('A. Sample Size by Year and Group')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.legend(title='Group')

# Panel B: Age distribution
df_control = df[df['ELIGIBLE'] == 0]['AGE']
df_treat = df[df['ELIGIBLE'] == 1]['AGE']

ax2.hist(df_control, bins=range(26, 41), alpha=0.7, label='Control (31-35)', color='#2166ac', edgecolor='black')
ax2.hist(df_treat, bins=range(26, 41), alpha=0.7, label='Treatment (26-30)', color='#b2182b', edgecolor='black')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')
ax2.set_title('B. Age Distribution by Group')
ax2.legend()

plt.tight_layout()
plt.savefig('figure4_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_distribution.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# =============================================================================
# Figure 5: Heterogeneity by Sex
# =============================================================================
print("Creating Figure 5: Heterogeneity by Sex...")

fig, ax = plt.subplots(figsize=(8, 6))

# Calculate rates by sex
sex_rates = df.groupby(['SEX', 'ELIGIBLE', 'AFTER'])['FT'].mean().unstack(level=[1,2])

# Positions
x = np.array([0, 1, 2.5, 3.5])
width = 0.35

# Male data
male_data = sex_rates.loc[1].values
# Female data
female_data = sex_rates.loc[2].values

bars1 = ax.bar(x[:2], [male_data[0], male_data[2]], width, label='Pre-DACA', color='#2166ac', alpha=0.8)
bars2 = ax.bar(x[:2] + width, [male_data[1], male_data[3]], width, label='Post-DACA', color='#b2182b', alpha=0.8)

bars3 = ax.bar(x[2:], [female_data[0], female_data[2]], width, color='#2166ac', alpha=0.8)
bars4 = ax.bar(x[2:] + width, [female_data[1], female_data[3]], width, color='#b2182b', alpha=0.8)

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment by Sex, Group, and Period')
ax.set_xticks([0.175, 1.175, 2.675, 3.675])
ax.set_xticklabels(['Control\n(Male)', 'Treatment\n(Male)', 'Control\n(Female)', 'Treatment\n(Female)'])
ax.legend(loc='upper right')
ax.set_ylim(0, 1.0)

# Add section labels
ax.text(0.5, -0.08, 'Males', transform=ax.transData, ha='center', fontsize=12, fontweight='bold')
ax.text(3.0, -0.08, 'Females', transform=ax.transData, ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_heterogeneity_sex.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_heterogeneity_sex.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# =============================================================================
# Figure 6: Pre-Trends Visualization
# =============================================================================
print("Creating Figure 6: Pre-Trends Visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate pre-period trends
pre_df = df[df['AFTER'] == 0]
pre_yearly = pre_df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
pre_yearly.columns = ['Control', 'Treatment']
pre_yearly['Difference'] = pre_yearly['Treatment'] - pre_yearly['Control']

years_pre = pre_yearly.index.values

ax.plot(years_pre, pre_yearly['Difference'], 'ko-', linewidth=2, markersize=10)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Add trend line
z = np.polyfit(years_pre, pre_yearly['Difference'].values, 1)
p = np.poly1d(z)
ax.plot(years_pre, p(years_pre), 'r--', linewidth=2, alpha=0.7, label=f'Linear trend (slope={z[0]:.4f})')

ax.set_xlabel('Year')
ax.set_ylabel('Difference in FT Rate (Treatment - Control)')
ax.set_title('Pre-Treatment Differences: Testing Parallel Trends Assumption')
ax.set_xticks(years_pre)
ax.legend()

plt.tight_layout()
plt.savefig('figure6_pre_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_pre_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

print("\nAll figures created successfully!")

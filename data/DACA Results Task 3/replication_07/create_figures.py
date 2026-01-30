"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Figure 1: Parallel Trends Plot
print("Creating Figure 1: Parallel Trends...")

# Calculate weighted means by year and group
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
trends.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

years = trends.index.tolist()
ax.plot(years, trends['Treatment (Ages 26-30)'], 'o-', color='#2166AC',
        linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, trends['Control (Ages 31-35)'], 's--', color='#B2182B',
        linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1] - 0.01, 'DACA\nImplementation',
        fontsize=10, color='gray', va='top')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Eligibility Group')
ax.legend(loc='lower right')
ax.set_xticks(years)
ax.set_ylim(0.55, 0.80)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# Figure 2: DID 2x2 Diagram
print("Creating Figure 2: DID Visualization...")

# Calculate group means
mean_treat_post = np.average(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['FT'],
                              weights=df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)]['PERWT'])
mean_treat_pre = np.average(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['FT'],
                             weights=df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)]['PERWT'])
mean_control_post = np.average(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['FT'],
                                weights=df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)]['PERWT'])
mean_control_pre = np.average(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['FT'],
                               weights=df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)]['PERWT'])

fig, ax = plt.subplots(figsize=(8, 6))

# Plot actual lines
ax.plot([0, 1], [mean_treat_pre, mean_treat_post], 'o-', color='#2166AC',
        linewidth=2.5, markersize=10, label='Treatment (Ages 26-30)')
ax.plot([0, 1], [mean_control_pre, mean_control_post], 's-', color='#B2182B',
        linewidth=2.5, markersize=10, label='Control (Ages 31-35)')

# Counterfactual line for treatment
counterfactual = mean_treat_pre + (mean_control_post - mean_control_pre)
ax.plot([0, 1], [mean_treat_pre, counterfactual], '--', color='#2166AC',
        linewidth=1.5, alpha=0.5, label='Treatment Counterfactual')

# Draw arrow for treatment effect
ax.annotate('', xy=(1.02, mean_treat_post), xytext=(1.02, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='#1B7837', lw=2))
ax.text(1.08, (mean_treat_post + counterfactual)/2,
        f'DID = {mean_treat_post - counterfactual:.3f}',
        fontsize=11, color='#1B7837', va='center')

ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Visualization')
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure2_did_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_diagram.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_did_diagram.png/pdf")

# Figure 3: Coefficient Plot
print("Creating Figure 3: Coefficient Plot...")

results = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

models = results['Model'].tolist()
estimates = results['DID_Estimate'].tolist()
ses = results['SE'].tolist()

y_pos = np.arange(len(models))

# Calculate confidence intervals
ci_lower = [e - 1.96*se for e, se in zip(estimates, ses)]
ci_upper = [e + 1.96*se for e, se in zip(estimates, ses)]

# Plot
ax.hlines(y_pos, ci_lower, ci_upper, color='#2166AC', linewidth=2, alpha=0.7)
ax.scatter(estimates, y_pos, color='#2166AC', s=100, zorder=5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DID Estimate (ELIGIBLE × AFTER)')
ax.set_title('DACA Effect on Full-Time Employment Across Model Specifications')

# Add reference line at preferred estimate
preferred_est = results[results['Model'] == '(4) WLS Clustered']['DID_Estimate'].values[0]
ax.axvline(x=preferred_est, color='#1B7837', linestyle=':', linewidth=1.5, alpha=0.7,
           label=f'Preferred = {preferred_est:.3f}')
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure3_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_coefficient_plot.png/pdf")

# Figure 4: Subgroup Analysis
print("Creating Figure 4: Subgroup Analysis...")

fig, ax = plt.subplots(figsize=(8, 5))

subgroups = ['Full Sample', 'Males', 'Females']
estimates = [0.0748, 0.0716, 0.0527]
ses = [0.0203, 0.0195, 0.0290]

y_pos = np.arange(len(subgroups))
ci_lower = [e - 1.96*se for e, se in zip(estimates, ses)]
ci_upper = [e + 1.96*se for e, se in zip(estimates, ses)]

ax.hlines(y_pos, ci_lower, ci_upper, color='#2166AC', linewidth=3, alpha=0.7)
ax.scatter(estimates, y_pos, color='#2166AC', s=150, zorder=5)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DID Estimate (ELIGIBLE × AFTER)')
ax.set_title('DACA Effect on Full-Time Employment by Sex')

for i, (est, lo, hi) in enumerate(zip(estimates, ci_lower, ci_upper)):
    ax.text(max(hi, est) + 0.005, i, f'{est:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_subgroup_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_subgroup_analysis.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_subgroup_analysis.png/pdf")

# Figure 5: Sample Distribution
print("Creating Figure 5: Sample Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Observations by Year
ax1 = axes[0, 0]
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.columns = ['Control', 'Treatment']
year_counts.plot(kind='bar', ax=ax1, color=['#B2182B', '#2166AC'])
ax1.set_xlabel('Year')
ax1.set_ylabel('Number of Observations')
ax1.set_title('(A) Sample Size by Year and Group')
ax1.legend(title='Group')
ax1.tick_params(axis='x', rotation=45)

# Panel B: Age Distribution
ax2 = axes[0, 1]
df_treat = df[df['ELIGIBLE'] == 1]['AGE_IN_JUNE_2012']
df_control = df[df['ELIGIBLE'] == 0]['AGE_IN_JUNE_2012']
ax2.hist([df_control, df_treat], bins=10, label=['Control', 'Treatment'],
         color=['#B2182B', '#2166AC'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Age in June 2012')
ax2.set_ylabel('Frequency')
ax2.set_title('(B) Age Distribution by Group')
ax2.legend()

# Panel C: Education Distribution
ax3 = axes[1, 0]
educ_recode = df['EDUC_RECODE'].value_counts()
educ_labels = ['Less than HS', 'High School', 'Some College', 'Two-Year', 'BA+']
# Map if needed
ax3.bar(range(len(educ_labels)), [educ_recode.get(l, 0) for l in educ_labels],
        color='#2166AC', alpha=0.8, edgecolor='black')
ax3.set_xticks(range(len(educ_labels)))
ax3.set_xticklabels(educ_labels, rotation=45, ha='right')
ax3.set_xlabel('Education Level')
ax3.set_ylabel('Frequency')
ax3.set_title('(C) Education Distribution')

# Panel D: FT Employment by Group and Period
ax4 = axes[1, 1]
group_means = df.groupby(['AFTER', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
group_means.columns = ['Control', 'Treatment']
group_means.index = ['Pre-DACA', 'Post-DACA']
group_means.plot(kind='bar', ax=ax4, color=['#B2182B', '#2166AC'])
ax4.set_xlabel('Period')
ax4.set_ylabel('Full-Time Employment Rate')
ax4.set_title('(D) Full-Time Employment by Group and Period')
ax4.legend(title='Group')
ax4.tick_params(axis='x', rotation=0)
ax4.set_ylim(0.5, 0.75)

plt.tight_layout()
plt.savefig('figure5_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_sample_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_sample_distribution.png/pdf")

print("\nAll figures created successfully!")

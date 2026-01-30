"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

output_dir = os.path.dirname(os.path.abspath(__file__))

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ============================================================================
# Figure 1: Event Study Plot
# ============================================================================
print("Creating Figure 1: Event Study Plot...")

event_results = pd.read_csv(os.path.join(output_dir, 'event_study_results.csv'))

fig, ax = plt.subplots(figsize=(10, 6))

# Reference year (2011) is at index 5, but we plot it with coefficient 0
years = event_results['year'].values
coefs = event_results['coef'].values
ses = event_results['se'].values

# Calculate confidence intervals
ci_lower = coefs - 1.96 * ses
ci_upper = coefs + 1.96 * ses

# Plot
ax.plot(years, coefs, 'o-', color='navy', linewidth=2, markersize=8, label='Coefficient')
ax.fill_between(years, ci_lower, ci_upper, alpha=0.2, color='navy', label='95% CI')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Effect on Full-Time Employment (pp)', fontsize=14)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks(years)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure1_event_study.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'figure1_event_study.pdf'), bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 2: Full-Time Employment Trends by Group
# ============================================================================
print("Creating Figure 2: Employment Trends...")

# Load data and compute yearly means
chunks = []
for i, chunk in enumerate(pd.read_csv(os.path.join(output_dir, 'data/data.csv'), chunksize=1000000)):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)

# Define variables
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['daca_eligible'] = (
    (df['age_at_immig'] >= 0) &
    (df['age_at_immig'] < 16) &
    (df['BIRTHYR'] >= 1982) &
    (df['YRIMMIG'] <= 2007) &
    (df['YRIMMIG'] > 0) &
    (df['CITIZEN'] == 3)
).astype(int)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Restrict to working age
df_plot = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)].copy()

# Calculate yearly means by group
yearly_means = df_plot.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(yearly_means.index, yearly_means[0], 'o-', color='gray', linewidth=2, markersize=8, label='Non-Eligible')
ax.plot(yearly_means.index, yearly_means[1], 's-', color='navy', linewidth=2, markersize=8, label='DACA-Eligible')
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')
ax.set_ylim([0.35, 0.70])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure2_trends.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'figure2_trends.pdf'), bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 3: Age Distribution of Sample
# ============================================================================
print("Creating Figure 3: Age Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

eligible = df_plot[df_plot['daca_eligible'] == 1]['AGE']
non_eligible = df_plot[df_plot['daca_eligible'] == 0]['AGE']

ax.hist(non_eligible, bins=range(16, 66), alpha=0.5, color='gray', label='Non-Eligible', density=True)
ax.hist(eligible, bins=range(16, 66), alpha=0.7, color='navy', label='DACA-Eligible', density=True)

ax.set_xlabel('Age', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Age Distribution by DACA Eligibility Status', fontsize=14)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure3_age_dist.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'figure3_age_dist.pdf'), bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 4: Robustness Check Comparison
# ============================================================================
print("Creating Figure 4: Robustness Comparison...")

robustness = pd.read_csv(os.path.join(output_dir, 'robustness_results.csv'))

# Add main result
main_results = pd.read_csv(os.path.join(output_dir, 'main_results.csv'))
preferred = main_results[main_results['Model'].str.contains('Preferred')]

specs = ['Main (Preferred)'] + list(robustness['Specification'])
coefs = [preferred['Coefficient'].values[0]] + list(robustness['Coefficient'])
ses = [preferred['SE'].values[0]] + list(robustness['SE'])

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(specs))
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.barh(y_pos, coefs, xerr=[np.array(coefs) - np.array(ci_lower), np.array(ci_upper) - np.array(coefs)],
        color=['navy'] + ['steelblue']*len(robustness), capsize=5, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(specs)
ax.set_xlabel('Effect on Full-Time Employment (pp)', fontsize=14)
ax.set_title('Robustness Check: DACA Effect Across Specifications', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure4_robustness.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'figure4_robustness.pdf'), bbox_inches='tight')
plt.close()

# ============================================================================
# Figure 5: Difference-in-Differences Illustration
# ============================================================================
print("Creating Figure 5: DiD Illustration...")

# Calculate pre/post means for each group
pre_eligible = df_plot[(df_plot['daca_eligible'] == 1) & (df_plot['YEAR'] < 2012)]['fulltime'].mean()
post_eligible = df_plot[(df_plot['daca_eligible'] == 1) & (df_plot['YEAR'] > 2012)]['fulltime'].mean()
pre_non_eligible = df_plot[(df_plot['daca_eligible'] == 0) & (df_plot['YEAR'] < 2012)]['fulltime'].mean()
post_non_eligible = df_plot[(df_plot['daca_eligible'] == 0) & (df_plot['YEAR'] > 2012)]['fulltime'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

# Treatment group
ax.plot([0, 1], [pre_eligible, post_eligible], 'o-', color='navy', linewidth=3, markersize=12, label='DACA-Eligible')

# Control group
ax.plot([0, 1], [pre_non_eligible, post_non_eligible], 's-', color='gray', linewidth=3, markersize=12, label='Non-Eligible')

# Counterfactual
counterfactual = pre_eligible + (post_non_eligible - pre_non_eligible)
ax.plot([0, 1], [pre_eligible, counterfactual], 'o--', color='navy', linewidth=2, markersize=8, alpha=0.5, label='Counterfactual')

# DiD effect arrow
ax.annotate('', xy=(1.05, post_eligible), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.1, (post_eligible + counterfactual)/2, f'DiD\n≈{(post_eligible-counterfactual)*100:.1f}pp',
        fontsize=12, color='red', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'], fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Difference-in-Differences Design', fontsize=14)
ax.legend(loc='upper left')
ax.set_xlim([-0.2, 1.4])
ax.set_ylim([0.35, 0.65])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'figure5_did_illustration.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'figure5_did_illustration.pdf'), bbox_inches='tight')
plt.close()

print("\nAll figures created successfully!")

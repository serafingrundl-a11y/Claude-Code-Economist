"""
DACA Replication Study - Figure Generation
Creates publication-quality figures for the LaTeX report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150

# Load results
event_df = pd.read_csv('event_study_results.csv')
regression_df = pd.read_csv('regression_results.csv')
summary_df = pd.read_csv('summary_statistics.csv', header=[0,1], index_col=[0,1])

print("Creating figures...")

# ==============================================================================
# Figure 1: Event Study Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Add reference year 2011 with coefficient 0
years_all = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coeffs_all = list(event_df['Coefficient'])
ses_all = list(event_df['SE'])

# Insert 2011 reference year
coeffs_all.insert(5, 0)
ses_all.insert(5, 0)

# Create 95% CI
ci_upper = [c + 1.96*s for c, s in zip(coeffs_all, ses_all)]
ci_lower = [c - 1.96*s for c, s in zip(coeffs_all, ses_all)]

# Plot
ax.fill_between(years_all, ci_lower, ci_upper, alpha=0.3, color='steelblue', label='95% CI')
ax.plot(years_all, coeffs_all, 'o-', color='steelblue', linewidth=2, markersize=8, label='Point Estimate')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (DACA Eligible × Year)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(years_all)
ax.set_xticklabels(years_all)
ax.legend(loc='upper left')
ax.set_ylim(-0.12, 0.15)

plt.tight_layout()
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.savefig('figure1_event_study.png', bbox_inches='tight', dpi=300)
print("  Created figure1_event_study.pdf")

# ==============================================================================
# Figure 2: Parallel Trends - Raw Means
# ==============================================================================
# Reload data to get yearly means
usecols = ['YEAR', 'SEX', 'AGE', 'BIRTHQTR', 'MARST', 'BIRTHYR', 'HISPAN',
           'BPL', 'CITIZEN', 'YRIMMIG', 'EMPSTAT', 'UHRSWORK']
dtypes = {
    'YEAR': 'int16', 'SEX': 'int8', 'AGE': 'int8', 'BIRTHQTR': 'int8',
    'MARST': 'int8', 'BIRTHYR': 'int16', 'HISPAN': 'int8', 'BPL': 'int16',
    'CITIZEN': 'int8', 'YRIMMIG': 'int16', 'EMPSTAT': 'int8', 'UHRSWORK': 'int8'
}

chunks = []
for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes, chunksize=1000000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)]
    chunks.append(filtered)
df = pd.concat(chunks, ignore_index=True)

# Create eligibility
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df['daca_eligible'] = (
    (df['age_at_immigration'] < 16) &
    (df['YRIMMIG'] <= 2007) &
    ((df['BIRTHYR'] >= 1982) | ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'].isin([3, 4]))))
).astype(int)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Filter to working age
df = df[(df['AGE'] >= 16) & (df['AGE'] <= 64)]

# Calculate yearly means
yearly_means = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(yearly_means.index, yearly_means[0], 'o-', color='gray', linewidth=2,
        markersize=8, label='Not DACA Eligible')
ax.plot(yearly_means.index, yearly_means[1], 's-', color='steelblue', linewidth=2,
        markersize=8, label='DACA Eligible')
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.legend(loc='upper right')
ax.set_xticks(range(2006, 2017))
ax.set_ylim(0.35, 0.70)

plt.tight_layout()
plt.savefig('figure2_parallel_trends.pdf', bbox_inches='tight')
plt.savefig('figure2_parallel_trends.png', bbox_inches='tight', dpi=300)
print("  Created figure2_parallel_trends.pdf")

# ==============================================================================
# Figure 3: DiD Coefficient Comparison Across Models
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = regression_df['Model']
coeffs = regression_df['DiD_Coefficient']
ses = regression_df['Standard_Error']

x_pos = np.arange(len(models))
colors = ['steelblue'] * len(models)

ax.barh(x_pos, coeffs, xerr=1.96*ses, capsize=5, color=colors, alpha=0.7, edgecolor='navy')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

ax.set_yticks(x_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (eligible × post)')
ax.set_title('Difference-in-Differences Estimates Across Model Specifications')

# Add coefficient values
for i, (coef, se) in enumerate(zip(coeffs, ses)):
    ax.text(coef + 1.96*se + 0.005, i, f'{coef:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_model_comparison.pdf', bbox_inches='tight')
plt.savefig('figure3_model_comparison.png', bbox_inches='tight', dpi=300)
print("  Created figure3_model_comparison.pdf")

# ==============================================================================
# Figure 4: Difference-in-Differences Visualization
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Raw means from the analysis output
pre_ineligible = 0.6039
post_ineligible = 0.5790
pre_eligible = 0.4309
post_eligible = 0.4962

# Plot actual trends
ax.plot([0, 1], [pre_ineligible, post_ineligible], 'o-', color='gray', linewidth=2.5,
        markersize=12, label='Not DACA Eligible (Actual)')
ax.plot([0, 1], [pre_eligible, post_eligible], 's-', color='steelblue', linewidth=2.5,
        markersize=12, label='DACA Eligible (Actual)')

# Plot counterfactual for eligible group
counterfactual = pre_eligible + (post_ineligible - pre_ineligible)
ax.plot([0, 1], [pre_eligible, counterfactual], 's--', color='steelblue', linewidth=2,
        markersize=8, alpha=0.5, label='DACA Eligible (Counterfactual)')

# Draw the DiD effect
ax.annotate('', xy=(1.05, post_eligible), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.12, (post_eligible + counterfactual)/2, f'DiD = {post_eligible - counterfactual:.3f}',
        fontsize=12, color='red', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Design')
ax.legend(loc='upper right')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.35, 0.70)

plt.tight_layout()
plt.savefig('figure4_did_diagram.pdf', bbox_inches='tight')
plt.savefig('figure4_did_diagram.png', bbox_inches='tight', dpi=300)
print("  Created figure4_did_diagram.pdf")

# ==============================================================================
# Figure 5: Distribution of Hours Worked
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-DACA distribution
df_pre = df[(df['YEAR'] <= 2011) & (df['YEAR'] != 2012)]
df_post = df[(df['YEAR'] >= 2013)]

for i, (data, title, period) in enumerate([(df_pre, 'Pre-DACA (2006-2011)', 'pre'),
                                            (df_post, 'Post-DACA (2013-2016)', 'post')]):
    ax = axes[i]

    eligible = data[data['daca_eligible'] == 1]['UHRSWORK']
    ineligible = data[data['daca_eligible'] == 0]['UHRSWORK']

    bins = np.arange(0, 101, 5)
    ax.hist(ineligible, bins=bins, alpha=0.5, density=True, color='gray', label='Not Eligible')
    ax.hist(eligible, bins=bins, alpha=0.5, density=True, color='steelblue', label='DACA Eligible')

    ax.axvline(x=35, color='red', linestyle='--', linewidth=1.5, label='Full-time threshold (35h)')
    ax.set_xlabel('Usual Hours Worked per Week')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure5_hours_distribution.pdf', bbox_inches='tight')
plt.savefig('figure5_hours_distribution.png', bbox_inches='tight', dpi=300)
print("  Created figure5_hours_distribution.pdf")

print("\nAll figures created successfully!")

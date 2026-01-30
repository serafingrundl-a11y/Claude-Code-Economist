"""
Generate figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "C:/Users/seraf/DACA Results Task 2/replication_36/"

# Load results
event_df = pd.read_csv(OUTPUT_DIR + "event_study_results.csv")
detailed_df = pd.read_csv(OUTPUT_DIR + "detailed_summary.csv")

# ============================================================
# Figure 1: Event Study Plot
# ============================================================
plt.figure(figsize=(10, 6))

# Add reference line for 2011 (omitted year)
years_with_ref = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coeffs_with_ref = list(event_df['coefficient'][:5]) + [0] + list(event_df['coefficient'][5:])
ci_lower_with_ref = list(event_df['ci_lower'][:5]) + [0] + list(event_df['ci_lower'][5:])
ci_upper_with_ref = list(event_df['ci_upper'][:5]) + [0] + list(event_df['ci_upper'][5:])

# Plot confidence intervals
plt.fill_between(years_with_ref, ci_lower_with_ref, ci_upper_with_ref, alpha=0.3, color='blue')

# Plot coefficients
plt.plot(years_with_ref, coeffs_with_ref, 'o-', color='blue', markersize=8, linewidth=2)

# Reference lines
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Treatment Effect on Full-Time Employment', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)
plt.legend(loc='upper left')
plt.xticks(years_with_ref)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "figure1_event_study.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR + "figure1_event_study.pdf", bbox_inches='tight')
plt.close()

print("Figure 1: Event study plot saved")

# ============================================================
# Figure 2: Parallel Trends Visualization
# ============================================================
# Need to recalculate yearly means from the data
# Load original data again for this
print("Loading data for parallel trends plot...")

df = pd.read_csv(OUTPUT_DIR + "data/data.csv")

# Apply same filters as in main analysis
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
df_mex = df_mex[df_mex['CITIZEN'] == 3].copy()

def calc_age_june_2012(row):
    birth_year = row['BIRTHYR']
    birth_qtr = row['BIRTHQTR']
    if birth_qtr in [1, 2]:
        return 2012 - birth_year
    else:
        return 2012 - birth_year - 1

df_mex['age_june_2012'] = df_mex.apply(calc_age_june_2012, axis=1)
df_mex = df_mex[df_mex['YRIMMIG'] > 0].copy()
df_mex['age_at_immig'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex = df_mex[df_mex['age_at_immig'] < 16].copy()
df_mex = df_mex[df_mex['YRIMMIG'] <= 2007].copy()

df_mex['treated'] = ((df_mex['age_june_2012'] >= 26) & (df_mex['age_june_2012'] <= 30)).astype(int)
df_mex['control'] = ((df_mex['age_june_2012'] >= 31) & (df_mex['age_june_2012'] <= 35)).astype(int)
df_analysis = df_mex[(df_mex['treated'] == 1) | (df_mex['control'] == 1)].copy()
df_analysis['fulltime'] = ((df_analysis['UHRSWORK'] >= 35) & (df_analysis['EMPSTAT'] == 1)).astype(int)
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()

# Calculate yearly means by group
yearly_means = df_analysis.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

plt.figure(figsize=(10, 6))

plt.plot(yearly_means.index, yearly_means[1], 'o-', color='blue', markersize=8,
         linewidth=2, label='Treatment (Age 26-30 in 2012)')
plt.plot(yearly_means.index, yearly_means[0], 's-', color='red', markersize=8,
         linewidth=2, label='Control (Age 31-35 in 2012)')

plt.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')

plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment by Treatment Status Over Time', fontsize=14)
plt.legend(loc='lower right')
plt.xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.ylim(0.4, 0.75)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "figure2_parallel_trends.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR + "figure2_parallel_trends.pdf", bbox_inches='tight')
plt.close()

print("Figure 2: Parallel trends plot saved")

# ============================================================
# Figure 3: DiD Visualization (2x2 diagram)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate pre and post means
pre_treat = df_analysis[(df_analysis['YEAR'] <= 2011) & (df_analysis['treated'] == 1)]
post_treat = df_analysis[(df_analysis['YEAR'] >= 2013) & (df_analysis['treated'] == 1)]
pre_control = df_analysis[(df_analysis['YEAR'] <= 2011) & (df_analysis['treated'] == 0)]
post_control = df_analysis[(df_analysis['YEAR'] >= 2013) & (df_analysis['treated'] == 0)]

means = {
    'pre_treat': np.average(pre_treat['fulltime'], weights=pre_treat['PERWT']),
    'post_treat': np.average(post_treat['fulltime'], weights=post_treat['PERWT']),
    'pre_control': np.average(pre_control['fulltime'], weights=pre_control['PERWT']),
    'post_control': np.average(post_control['fulltime'], weights=post_control['PERWT'])
}

# Plot treatment group
ax.plot([0, 1], [means['pre_treat'], means['post_treat']], 'o-', color='blue',
        markersize=12, linewidth=2, label='Treatment (26-30)')

# Plot control group
ax.plot([0, 1], [means['pre_control'], means['post_control']], 's-', color='red',
        markersize=12, linewidth=2, label='Control (31-35)')

# Counterfactual line
counterfactual = means['pre_treat'] + (means['post_control'] - means['pre_control'])
ax.plot([0, 1], [means['pre_treat'], counterfactual], '--', color='blue',
        linewidth=1.5, alpha=0.5, label='Counterfactual')

# Add DiD arrow
ax.annotate('', xy=(1.05, means['post_treat']), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (means['post_treat'] + counterfactual)/2, f'DiD\n{means["post_treat"] - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xlim(-0.2, 1.4)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Illustration', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "figure3_did_diagram.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR + "figure3_did_diagram.pdf", bbox_inches='tight')
plt.close()

print("Figure 3: DiD diagram saved")

# ============================================================
# Figure 4: Sample distribution by age at DACA implementation
# ============================================================
plt.figure(figsize=(10, 6))

age_dist = df_analysis.groupby('age_june_2012')['PERWT'].sum()

colors = ['blue' if 26 <= age <= 30 else 'red' for age in age_dist.index]
plt.bar(age_dist.index, age_dist.values / 1000000, color=colors, edgecolor='black', alpha=0.7)

plt.xlabel('Age as of June 15, 2012', fontsize=12)
plt.ylabel('Population (Millions)', fontsize=12)
plt.title('Sample Distribution by Age at DACA Implementation', fontsize=14)
plt.xticks(range(26, 36))

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, edgecolor='black', label='Treatment (26-30)'),
                   Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Control (31-35)')]
plt.legend(handles=legend_elements, loc='upper right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "figure4_age_distribution.png", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR + "figure4_age_distribution.pdf", bbox_inches='tight')
plt.close()

print("Figure 4: Age distribution saved")

# ============================================================
# Save yearly means for the report
# ============================================================
yearly_means_df = yearly_means.reset_index()
yearly_means_df.columns = ['Year', 'Control', 'Treatment']
yearly_means_df.to_csv(OUTPUT_DIR + "yearly_means.csv", index=False)

print("\nAll figures generated successfully!")
print("Files created:")
print("  - figure1_event_study.png/pdf")
print("  - figure2_parallel_trends.png/pdf")
print("  - figure3_did_diagram.png/pdf")
print("  - figure4_age_distribution.png/pdf")
print("  - yearly_means.csv")

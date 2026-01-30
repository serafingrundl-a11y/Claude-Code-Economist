"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Load event study results
event_df = pd.read_csv('event_study_results.csv')

# ============================================================================
# Figure 1: Event Study Plot
# ============================================================================
print("Creating Figure 1: Event Study...")

fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_low = event_df['CI_low'].values
ci_high = event_df['CI_high'].values

# Calculate error bars
errors = np.array([coefs - ci_low, ci_high - coefs])

# Plot with confidence intervals
ax.errorbar(years, coefs, yerr=errors, fmt='o-', color='navy',
            capsize=4, capthick=2, markersize=8, linewidth=2,
            label='Point estimate with 95% CI')

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Add vertical line at DACA implementation (between 2011 and 2013)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='DACA Implementation (June 2012)')

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-period')

ax.set_xlabel('Year')
ax.set_ylabel('Effect on Full-Time Employment (relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Treatment vs. Control Group)')
ax.set_xticks(years)
ax.set_xlim(2005.5, 2016.5)

ax.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("   Saved figure1_event_study.png and .pdf")
plt.close()

# ============================================================================
# Figure 2: Parallel Trends Visualization
# ============================================================================
print("Creating Figure 2: Parallel Trends...")

# Load data and recreate yearly means
import gc

usecols = ['YEAR', 'STATEFIP', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'MARST',
           'BIRTHYR', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EDUC', 'UHRSWORK']

dtypes = {
    'YEAR': 'int16', 'STATEFIP': 'int8', 'PERWT': 'float32', 'SEX': 'int8',
    'AGE': 'int16', 'BIRTHQTR': 'int8', 'MARST': 'int8', 'BIRTHYR': 'int16',
    'HISPAN': 'int8', 'BPL': 'int16', 'CITIZEN': 'int8', 'YRIMMIG': 'int16',
    'EDUC': 'int8', 'UHRSWORK': 'int8'
}

chunks = []
for chunk in pd.read_csv('data/data.csv', usecols=usecols, dtype=dtypes, chunksize=100000):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)].copy()
    if len(filtered) > 0:
        chunks.append(filtered)
    del chunk
    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()

# Apply filters
df = df[df['CITIZEN'] == 3].copy()
df['age_at_daca'] = 2012 - df['BIRTHYR']
mask_late = df['BIRTHQTR'].isin([3, 4])
df.loc[mask_late, 'age_at_daca'] = df.loc[mask_late, 'age_at_daca'] - 1

df = df[(df['age_at_daca'] >= 26) & (df['age_at_daca'] <= 35)].copy()
df['treat'] = (df['age_at_daca'] <= 30).astype('int8')

# DACA eligibility
df['year_turn_16'] = df['BIRTHYR'] + 16
df = df[df['YRIMMIG'] <= df['year_turn_16']].copy()
df = df[df['YRIMMIG'] <= 2007].copy()

# Exclude 2012
df = df[df['YEAR'] != 2012].copy()
df['fulltime'] = (df['UHRSWORK'] >= 35).astype('int8')

# Calculate weighted means by year and treatment group
yearly_means = df.groupby(['YEAR', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years_plot = yearly_means.index.values
ax.plot(years_plot, yearly_means[0], 'o-', color='darkred', markersize=8,
        linewidth=2, label='Control (Ages 31-35 at DACA)')
ax.plot(years_plot, yearly_means[1], 's-', color='darkblue', markersize=8,
        linewidth=2, label='Treatment (Ages 26-30 at DACA)')

ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7,
           label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Treatment Group')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.55, 0.75)
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('figure2_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_parallel_trends.pdf', bbox_inches='tight')
print("   Saved figure2_parallel_trends.png and .pdf")
plt.close()

# ============================================================================
# Figure 3: DiD Visualization (2x2)
# ============================================================================
print("Creating Figure 3: DiD Visualization...")

# Calculate group means
pre_treat = df[(df['treat']==1) & (df['YEAR']<=2011)]['fulltime'].mean()
pre_ctrl = df[(df['treat']==0) & (df['YEAR']<=2011)]['fulltime'].mean()
post_treat = df[(df['treat']==1) & (df['YEAR']>=2013)]['fulltime'].mean()
post_ctrl = df[(df['treat']==0) & (df['YEAR']>=2013)]['fulltime'].mean()

fig, ax = plt.subplots(figsize=(8, 6))

# Plot actual outcomes
periods = [0, 1]
ax.plot(periods, [pre_ctrl, post_ctrl], 'o-', color='darkred', markersize=12,
        linewidth=2.5, label='Control (Ages 31-35)')
ax.plot(periods, [pre_treat, post_treat], 's-', color='darkblue', markersize=12,
        linewidth=2.5, label='Treatment (Ages 26-30)')

# Counterfactual for treatment group
counterfactual = pre_treat + (post_ctrl - pre_ctrl)
ax.plot(periods, [pre_treat, counterfactual], 's--', color='darkblue',
        markersize=10, linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# Arrow showing treatment effect
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
ax.text(1.05, (post_treat + counterfactual)/2,
        f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', fontweight='bold')

ax.set_xticks(periods)
ax.set_xticklabels(['Pre-Period\n(2006-2011)', 'Post-Period\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Design')
ax.set_ylim(0.55, 0.70)
ax.legend(loc='lower left', framealpha=0.9)

plt.tight_layout()
plt.savefig('figure3_did_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_diagram.pdf', bbox_inches='tight')
print("   Saved figure3_did_diagram.png and .pdf")
plt.close()

# ============================================================================
# Figure 4: Robustness - Bandwidth Analysis
# ============================================================================
print("Creating Figure 4: Bandwidth Sensitivity...")

bandwidths = [3, 4, 5]
coefs_bw = [0.0357, 0.0329, 0.0444]
se_bw = [0.0099, 0.0117, 0.0076]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(bandwidths, coefs_bw, yerr=[1.96*s for s in se_bw],
            fmt='o', color='navy', capsize=6, capthick=2, markersize=10,
            label='Point estimate with 95% CI')

ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)

ax.set_xlabel('Age Bandwidth (years from cutoff)')
ax.set_ylabel('DiD Coefficient')
ax.set_title('Sensitivity to Age Bandwidth Choice')
ax.set_xticks(bandwidths)
ax.set_xticklabels(['±3 years\n(ages 28-33)', '±4 years\n(ages 27-34)', '±5 years\n(ages 26-35)'])

plt.tight_layout()
plt.savefig('figure4_bandwidth.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_bandwidth.pdf', bbox_inches='tight')
print("   Saved figure4_bandwidth.png and .pdf")
plt.close()

print("\nAll figures created successfully!")

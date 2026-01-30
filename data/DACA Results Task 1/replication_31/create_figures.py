"""
DACA Replication Study - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ============================================================================
# Figure 1: Event Study Plot
# ============================================================================
print("Creating Figure 1: Event Study...")

event_df = pd.read_csv('event_study_results.csv')
event_df = event_df.sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate confidence intervals
event_df['ci_low'] = event_df['coefficient'] - 1.96 * event_df['se']
event_df['ci_high'] = event_df['coefficient'] + 1.96 * event_df['se']

# Plot
years = event_df['year'].values
coefs = event_df['coefficient'].values
ci_low = event_df['ci_low'].values
ci_high = event_df['ci_high'].values

# Pre and post periods
pre_mask = years < 2012
post_mask = years >= 2013

ax.errorbar(years[pre_mask], coefs[pre_mask],
            yerr=[coefs[pre_mask] - ci_low[pre_mask], ci_high[pre_mask] - coefs[pre_mask]],
            fmt='o', color='steelblue', capsize=5, capthick=2, markersize=8, label='Pre-DACA')
ax.errorbar(years[post_mask], coefs[post_mask],
            yerr=[coefs[post_mask] - ci_low[post_mask], ci_high[post_mask] - coefs[post_mask]],
            fmt='o', color='firebrick', capsize=5, capthick=2, markersize=8, label='Post-DACA')

# Reference year
ax.scatter([2011], [0], color='darkgreen', s=100, marker='D', zorder=5, label='Reference (2011)')

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA (June 2012)')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Effect on Full-Time Employment\n(Relative to 2011)', fontsize=14)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=16)
ax.legend(loc='upper left')
ax.set_xticks(years)
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_event_study.png/pdf")

# ============================================================================
# Figure 2: Trends in Full-Time Employment by Eligibility
# ============================================================================
print("Creating Figure 2: Employment Trends...")

# Load data for trends
DATA_PATH = 'data/data.csv'
usecols = ['YEAR', 'PERWT', 'AGE', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
           'BIRTHYR', 'BIRTHQTR', 'UHRSWORK', 'EMPSTAT']

chunks = []
for chunk in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=500000):
    mask = (chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) & (chunk['CITIZEN'] == 3)
    filtered = chunk[mask].copy()
    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)

# Create variables
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df['age_june2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_june2012'] = df['age_june2012'] - 1
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007).astype(int)
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)
df['under_31_in_2012'] = (df['age_june2012'] < 31).astype(int)

df['daca_eligible'] = (
    (df['arrived_before_16'] == 1) &
    (df['under_31_in_2012'] == 1) &
    (df['in_us_since_2007'] == 1)
).astype(int)

df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]
df = df[df['YEAR'] != 2012]

# Calculate weighted means by year and eligibility
trends = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(trends.index, trends[0], 'o-', color='steelblue', linewidth=2, markersize=8,
        label='Not DACA-Eligible')
ax.plot(trends.index, trends[1], 's-', color='firebrick', linewidth=2, markersize=8,
        label='DACA-Eligible')

ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA (June 2012)')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=16)
ax.legend(loc='lower right')
ax.set_ylim(0.4, 0.7)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_trends.png/pdf")

# ============================================================================
# Figure 3: DiD Visual
# ============================================================================
print("Creating Figure 3: Difference-in-Differences Visual...")

# Calculate pre/post means
pre_eligible = df[(df['YEAR'] < 2012) & (df['daca_eligible'] == 1)]
pre_noteligible = df[(df['YEAR'] < 2012) & (df['daca_eligible'] == 0)]
post_eligible = df[(df['YEAR'] >= 2013) & (df['daca_eligible'] == 1)]
post_noteligible = df[(df['YEAR'] >= 2013) & (df['daca_eligible'] == 0)]

means = {
    'pre_eligible': np.average(pre_eligible['fulltime'], weights=pre_eligible['PERWT']),
    'pre_noteligible': np.average(pre_noteligible['fulltime'], weights=pre_noteligible['PERWT']),
    'post_eligible': np.average(post_eligible['fulltime'], weights=post_eligible['PERWT']),
    'post_noteligible': np.average(post_noteligible['fulltime'], weights=post_noteligible['PERWT'])
}

fig, ax = plt.subplots(figsize=(8, 6))

# Plot lines for each group
x = [0, 1]
x_labels = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']

ax.plot(x, [means['pre_noteligible'], means['post_noteligible']], 'o-',
        color='steelblue', linewidth=2.5, markersize=12, label='Not DACA-Eligible')
ax.plot(x, [means['pre_eligible'], means['post_eligible']], 's-',
        color='firebrick', linewidth=2.5, markersize=12, label='DACA-Eligible')

# Add counterfactual
counterfactual = means['pre_eligible'] + (means['post_noteligible'] - means['pre_noteligible'])
ax.plot([0, 1], [means['pre_eligible'], counterfactual], 's--',
        color='firebrick', linewidth=2, markersize=10, alpha=0.5,
        label='Counterfactual (DACA-Eligible)')

# Add annotation for treatment effect
did_effect = means['post_eligible'] - counterfactual
ax.annotate('', xy=(1.05, means['post_eligible']), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2))
ax.text(1.1, (means['post_eligible'] + counterfactual)/2, f'DiD Effect\n= {did_effect:.3f}',
        fontsize=11, color='darkgreen', va='center')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=16)
ax.legend(loc='lower right')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(0.45, 0.65)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_did.png/pdf")

# ============================================================================
# Figure 4: Age Distribution by Eligibility
# ============================================================================
print("Creating Figure 4: Age Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

eligible = df[df['daca_eligible'] == 1]['AGE']
not_eligible = df[df['daca_eligible'] == 0]['AGE']

ax.hist(not_eligible, bins=range(18, 66), alpha=0.6, color='steelblue',
        label='Not DACA-Eligible', density=True)
ax.hist(eligible, bins=range(18, 66), alpha=0.6, color='firebrick',
        label='DACA-Eligible', density=True)

ax.set_xlabel('Age', fontsize=14)
ax.set_ylabel('Density', fontsize=14)
ax.set_title('Age Distribution by DACA Eligibility Status', fontsize=16)
ax.legend()

plt.tight_layout()
plt.savefig('figure4_age_dist.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_age_dist.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_age_dist.png/pdf")

print("\nAll figures created successfully!")

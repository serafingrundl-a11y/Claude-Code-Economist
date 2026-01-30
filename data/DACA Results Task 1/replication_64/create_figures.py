"""
DACA Employment Effects Replication Study
Figure Generation Script - Replication 64
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style for publication
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = [6, 4]

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study Plot...")

event_data = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(8, 5))

years = event_data['Year'].values
coefs = event_data['Coefficient'].values
ses = event_data['Std_Error'].values

# Calculate 95% CI
ci_lower = coefs - 1.96 * ses
ci_upper = coefs + 1.96 * ses

# Plot
ax.errorbar(years, coefs, yerr=1.96*ses, fmt='o-', color='navy',
            capsize=3, capthick=1.5, linewidth=1.5, markersize=6)

# Reference lines
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label='DACA Implementation')

# Labels and formatting
ax.set_xlabel('Year')
ax.set_ylabel('DACA Eligible x Year Coefficient')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(years)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Annotate
ax.annotate('Pre-DACA', xy=(2008.5, ax.get_ylim()[1]*0.9), ha='center', fontsize=9, style='italic')
ax.annotate('Post-DACA', xy=(2014.5, ax.get_ylim()[1]*0.9), ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("Figure 1 saved.")

# =============================================================================
# Figure 2: Parallel Trends / Employment Rates Over Time
# =============================================================================
print("Creating Figure 2: Employment Trends...")

# Load data
df = pd.read_csv('data/data.csv')

# Filter to analysis sample
df_mex = df[(df['HISPAN'] == 1) | ((df['HISPAND'] >= 100) & (df['HISPAND'] <= 107))]
df_mex = df_mex[(df_mex['BPL'] == 200) | (df_mex['BPLD'] == 20000)]
df_mex = df_mex[df_mex['CITIZEN'] == 3]
df_mex = df_mex[(df_mex['AGE'] >= 18) & (df_mex['AGE'] <= 64)]

# DACA eligibility
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)
df_mex['born_after_cutoff'] = (
    (df_mex['BIRTHYR'] > 1981) |
    ((df_mex['BIRTHYR'] == 1981) & (df_mex['BIRTHQTR'] >= 3))
).astype(int)
df_mex['arrived_by_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)
df_mex['daca_eligible'] = (
    (df_mex['arrived_before_16'] == 1) &
    (df_mex['born_after_cutoff'] == 1) &
    (df_mex['arrived_by_2007'] == 1)
).astype(int)

df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Weighted mean by year and treatment
def weighted_mean(group):
    return np.average(group['fulltime'], weights=group['PERWT'])

trends = df_mex.groupby(['YEAR', 'daca_eligible']).apply(weighted_mean).unstack()

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(trends.index, trends[0], 'o-', color='gray', linewidth=2, markersize=6, label='DACA Ineligible')
ax.plot(trends.index, trends[1], 's-', color='navy', linewidth=2, markersize=6, label='DACA Eligible')

ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.45, 0.70])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
print("Figure 2 saved.")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(7, 5))

# Pre and Post means
pre_inelig = trends.loc[2006:2011, 0].mean()
post_inelig = trends.loc[2013:2016, 0].mean()
pre_elig = trends.loc[2006:2011, 1].mean()
post_elig = trends.loc[2013:2016, 1].mean()

# Plot bars
x = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x - width/2, [pre_inelig, post_inelig], width, label='DACA Ineligible', color='gray', alpha=0.7)
bars2 = ax.bar(x + width/2, [pre_elig, post_elig], width, label='DACA Eligible', color='navy', alpha=0.7)

# Counterfactual line
cf_elig = pre_elig + (post_inelig - pre_inelig)  # Counterfactual
ax.plot([1-width/2, 1+width/2], [post_inelig, cf_elig], 'r--', linewidth=2, label='Counterfactual')
ax.scatter([1+width/2], [cf_elig], color='red', s=50, zorder=5)

# Treatment effect arrow
ax.annotate('', xy=(1+width/2, post_elig), xytext=(1+width/2, cf_elig),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1+width/2+0.08, (post_elig + cf_elig)/2, f'DiD\n+{(post_elig-cf_elig):.3f}',
        color='green', fontsize=10, va='center')

ax.set_ylabel('Full-Time Employment Rate')
ax.set_xlabel('Period')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'])
ax.set_title('Difference-in-Differences Visualization')
ax.legend(loc='upper left')
ax.set_ylim([0.45, 0.70])

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
print("Figure 3 saved.")

# =============================================================================
# Figure 4: Robustness Checks Summary
# =============================================================================
print("Creating Figure 4: Robustness Checks...")

rob_data = pd.read_csv('robustness_results.csv')

fig, ax = plt.subplots(figsize=(8, 5))

specs = rob_data['Specification'].values
coefs = rob_data['DiD Coefficient'].values
ses = rob_data['Std Error'].values

y_pos = np.arange(len(specs))

# Main estimate for reference
main_coef = 0.0194
main_se = 0.0037

ax.errorbar(coefs, y_pos, xerr=1.96*ses, fmt='o', color='navy',
            capsize=4, capthick=1.5, markersize=8)

# Reference line for main estimate
ax.axvline(x=main_coef, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

ax.fill_betweenx([-0.5, len(specs)-0.5], main_coef - 1.96*main_se, main_coef + 1.96*main_se,
                  color='red', alpha=0.15, label='Main Estimate 95% CI')

ax.set_yticks(y_pos)
ax.set_yticklabels(specs)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)')
ax.set_title('Robustness Checks: DACA Effect Across Specifications')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
print("Figure 4 saved.")

print("\nAll figures created successfully!")

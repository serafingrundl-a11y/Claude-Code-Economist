"""
DACA Replication Study - Figure Generation Script
Creates figures for the replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

print("Loading data...")
df = pd.read_csv('data/data.csv')

# Apply same restrictions as main analysis
df = df[df['HISPAN'] == 1].copy()
df = df[df['BPL'] == 200].copy()
df = df[df['CITIZEN'] == 3].copy()
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)].copy()
df = df[df['YRIMMIG'] > 0].copy()
df = df[df['BIRTHYR'] > 0].copy()

# Create variables
df['age_at_immig'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_immig'] < 16).astype(int)
df['under_31_2012'] = ((df['BIRTHYR'] > 1981) |
                        ((df['BIRTHYR'] == 1981) & (df['BIRTHQTR'] >= 3))).astype(int)
df['arrived_by_2007'] = (df['YRIMMIG'] <= 2007).astype(int)
df['daca_eligible'] = ((df['arrived_before_16'] == 1) &
                        (df['under_31_2012'] == 1) &
                        (df['arrived_by_2007'] == 1)).astype(int)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

print("Creating figures...")

# =============================================================================
# Figure 1: Full-time Employment Trends by DACA Eligibility
# =============================================================================
print("  Figure 1: Employment trends...")

# Calculate annual means
annual_means = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(annual_means.index, annual_means[0], 'b-o', label='Not DACA Eligible', linewidth=2, markersize=8)
ax.plot(annual_means.index, annual_means[1], 'r-s', label='DACA Eligible', linewidth=2, markersize=8)
ax.axvline(x=2012.5, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (mid-2012)')
ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Full-time Employment Trends by DACA Eligibility Status')
ax.legend(loc='lower left')
ax.set_ylim(0.45, 0.70)
ax.set_xticks(range(2006, 2017))
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("  Figure 2: Event study...")

# Event study coefficients from the analysis
event_coefs = {
    2006: (0.0085, 0.0093),
    2007: (0.0057, 0.0053),
    2008: (0.0125, 0.0092),
    2009: (0.0116, 0.0081),
    2010: (0.0059, 0.0104),
    2011: (0, 0),  # Reference year
    2013: (0.0043, 0.0092),
    2014: (0.0219, 0.0128),
    2015: (0.0378, 0.0096),
    2016: (0.0380, 0.0077)
}

years = sorted(event_coefs.keys())
coefs = [event_coefs[y][0] for y in years]
ses = [event_coefs[y][1] for y in years]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, coefs, yerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o-', color='navy', linewidth=2, markersize=8, capsize=5, capthick=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2)
ax.fill_between([2006, 2011.5], -0.04, 0.07, alpha=0.1, color='blue', label='Pre-DACA')
ax.fill_between([2012.5, 2016], -0.04, 0.07, alpha=0.1, color='red', label='Post-DACA')
ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment')
ax.set_xticks(years)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(-0.04, 0.07)
ax.legend(loc='upper left')
ax.text(2012.2, 0.06, 'DACA\nImplemented', fontsize=10, ha='left')
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 3: Age Distribution by DACA Eligibility
# =============================================================================
print("  Figure 3: Age distribution...")

fig, ax = plt.subplots(figsize=(10, 6))
df_2016 = df[df['YEAR'] == 2016]
df_2016[df_2016['daca_eligible']==0]['AGE'].hist(ax=ax, bins=range(18, 65), alpha=0.5,
                                                   label='Not Eligible', density=True, color='blue')
df_2016[df_2016['daca_eligible']==1]['AGE'].hist(ax=ax, bins=range(18, 65), alpha=0.5,
                                                   label='DACA Eligible', density=True, color='red')
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Age Distribution by DACA Eligibility Status (2016)')
ax.legend()
plt.tight_layout()
plt.savefig('figure3_age_dist.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_age_dist.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 4: Coefficient Plot - Model Comparison
# =============================================================================
print("  Figure 4: Coefficient comparison...")

models = ['Basic DiD', 'Demographics', 'Education', 'State FE', 'State+Year FE\n(Preferred)']
coefs = [0.0605, 0.0279, 0.0262, 0.0255, 0.0185]
ses = [0.0032, 0.0046, 0.0044, 0.0045, 0.0042]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(models))
ax.barh(y_pos, coefs, xerr=[1.96*s for s in ses], align='center', color='steelblue',
        alpha=0.8, capsize=5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (with 95% CI)')
ax.set_title('Effect of DACA Eligibility on Full-time Employment: Model Comparison')
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.text(c + 1.96*s + 0.002, i, f'{c:.4f}', va='center', fontsize=10)
ax.set_xlim(-0.01, 0.10)
plt.tight_layout()
plt.savefig('figure4_coef_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coef_comparison.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# Figure 5: Robustness Checks
# =============================================================================
print("  Figure 5: Robustness checks...")

rob_models = ['Preferred\n(Baseline)', 'Employment\nOutcome', 'Labor Force\nOnly',
              'Males Only', 'Females Only', 'Young Adults\n(16-35)', 'Weighted']
rob_coefs = [0.0185, 0.0296, 0.0056, 0.0113, 0.0165, 0.0085, 0.0176]
rob_ses = [0.0042, 0.0075, 0.0030, 0.0036, 0.0073, 0.0057, 0.0036]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(rob_models))
colors = ['navy'] + ['steelblue']*6
ax.barh(y_pos, rob_coefs, xerr=[1.96*s for s in rob_ses], align='center',
        color=colors, alpha=0.8, capsize=5)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.0185, color='navy', linestyle='--', linewidth=1, alpha=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(rob_models)
ax.set_xlabel('DiD Coefficient (with 95% CI)')
ax.set_title('Robustness Checks: Effect of DACA Eligibility on Full-time Employment')
for i, (c, s) in enumerate(zip(rob_coefs, rob_ses)):
    ax.text(c + 1.96*s + 0.002, i, f'{c:.4f}', va='center', fontsize=10)
ax.set_xlim(-0.02, 0.08)
plt.tight_layout()
plt.savefig('figure5_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_robustness.pdf', bbox_inches='tight')
plt.close()

print("All figures created successfully!")
print("Files created:")
print("  - figure1_trends.png/pdf")
print("  - figure2_eventstudy.png/pdf")
print("  - figure3_age_dist.png/pdf")
print("  - figure4_coef_comparison.png/pdf")
print("  - figure5_robustness.png/pdf")

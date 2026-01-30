"""
Create figures for DACA replication report
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

print("Loading data...")
df = pd.read_csv('data/data.csv')

# Apply same filters as main analysis
df_target = df[df['HISPAN'] == 1].copy()
df_target = df_target[df_target['BPL'] == 200].copy()
df_target = df_target[df_target['CITIZEN'] == 3].copy()
df_target = df_target[(df_target['AGE'] >= 16) & (df_target['AGE'] <= 64)].copy()
df_target = df_target[df_target['YEAR'] != 2012].copy()

# Create eligibility indicator
df_target['age_at_arrival'] = df_target['YRIMMIG'] - df_target['BIRTHYR']
df_target['daca_eligible'] = (
    (df_target['age_at_arrival'] < 16) &
    (df_target['age_at_arrival'] >= 0) &
    (df_target['BIRTHYR'] >= 1981) &
    (df_target['YRIMMIG'] <= 2007) &
    (df_target['YRIMMIG'] > 0)
).astype(int)

df_target['fulltime'] = (df_target['UHRSWORK'] >= 35).astype(int)

print("Creating figures...")

# ============================================================================
# Figure 1: Parallel Trends / Event Study
# ============================================================================
print("Figure 1: Event study plot...")

# Calculate mean full-time rate by year and eligibility
trends = df_target.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trends.index, trends[0], 'b-o', label='Non-eligible', markersize=8, linewidth=2)
ax.plot(trends.index, trends[1], 'r-s', label='DACA-eligible', markersize=8, linewidth=2)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Full-time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.legend(loc='best', fontsize=11)
ax.set_ylim(0.35, 0.70)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure1_trends.png")

# ============================================================================
# Figure 2: Event Study Coefficients
# ============================================================================
print("Figure 2: Event study coefficients...")

# Event study coefficients from analysis
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
# These are from the analysis output (relative to 2011)
coefs = [-0.0171, -0.0139, -0.0012, 0.0023, 0.0075, 0, 0.0082, 0.0198, 0.0401, 0.0388]
ses = [0.0078, 0.0076, 0.0077, 0.0076, 0.0074, 0, 0.0073, 0.0073, 0.0073, 0.0074]

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(years, coefs, yerr=[1.96*s for s in ses], fmt='ko', capsize=5,
            markersize=8, linewidth=2, capthick=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA implementation')
ax.fill_between([2012.5, 2016.5], -0.05, 0.08, alpha=0.2, color='green', label='Post-DACA period')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: DACA Eligibility Effect on Full-time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=11)
ax.set_xticks(years)
ax.set_ylim(-0.05, 0.08)
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure2_eventstudy.png")

# ============================================================================
# Figure 3: Distribution of Hours Worked
# ============================================================================
print("Figure 3: Hours distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-DACA
pre_eligible = df_target[(df_target['YEAR'] < 2012) & (df_target['daca_eligible'] == 1)]['UHRSWORK']
pre_nonelig = df_target[(df_target['YEAR'] < 2012) & (df_target['daca_eligible'] == 0)]['UHRSWORK']

axes[0].hist(pre_eligible, bins=range(0, 100, 5), alpha=0.7, label='DACA-eligible', density=True)
axes[0].hist(pre_nonelig, bins=range(0, 100, 5), alpha=0.5, label='Non-eligible', density=True)
axes[0].axvline(x=35, color='red', linestyle='--', linewidth=2, label='Full-time threshold')
axes[0].set_xlabel('Usual Hours Worked per Week')
axes[0].set_ylabel('Density')
axes[0].set_title('Pre-DACA Period (2006-2011)')
axes[0].legend()

# Post-DACA
post_eligible = df_target[(df_target['YEAR'] >= 2013) & (df_target['daca_eligible'] == 1)]['UHRSWORK']
post_nonelig = df_target[(df_target['YEAR'] >= 2013) & (df_target['daca_eligible'] == 0)]['UHRSWORK']

axes[1].hist(post_eligible, bins=range(0, 100, 5), alpha=0.7, label='DACA-eligible', density=True)
axes[1].hist(post_nonelig, bins=range(0, 100, 5), alpha=0.5, label='Non-eligible', density=True)
axes[1].axvline(x=35, color='red', linestyle='--', linewidth=2, label='Full-time threshold')
axes[1].set_xlabel('Usual Hours Worked per Week')
axes[1].set_ylabel('Density')
axes[1].set_title('Post-DACA Period (2013-2016)')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure3_hours.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure3_hours.png")

# ============================================================================
# Figure 4: Sample Composition
# ============================================================================
print("Figure 4: Sample composition...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
ax = axes[0, 0]
eligible_ages = df_target[df_target['daca_eligible'] == 1]['AGE']
nonelig_ages = df_target[df_target['daca_eligible'] == 0]['AGE']
ax.hist(eligible_ages, bins=range(16, 66, 2), alpha=0.7, label='DACA-eligible', density=True)
ax.hist(nonelig_ages, bins=range(16, 66, 2), alpha=0.5, label='Non-eligible', density=True)
ax.set_xlabel('Age')
ax.set_ylabel('Density')
ax.set_title('Age Distribution')
ax.legend()

# Education distribution
ax = axes[0, 1]
edu_elig = df_target[df_target['daca_eligible'] == 1]['EDUC'].value_counts(normalize=True).sort_index()
edu_nonelig = df_target[df_target['daca_eligible'] == 0]['EDUC'].value_counts(normalize=True).sort_index()
x = np.arange(len(edu_elig))
width = 0.35
ax.bar(x - width/2, edu_elig.values, width, label='DACA-eligible', alpha=0.7)
ax.bar(x + width/2, edu_nonelig.values, width, label='Non-eligible', alpha=0.7)
ax.set_xlabel('Education Level')
ax.set_ylabel('Proportion')
ax.set_title('Education Distribution')
ax.legend()

# Year of immigration
ax = axes[1, 0]
immig_elig = df_target[(df_target['daca_eligible'] == 1) & (df_target['YRIMMIG'] > 1980)]['YRIMMIG']
immig_nonelig = df_target[(df_target['daca_eligible'] == 0) & (df_target['YRIMMIG'] > 1980)]['YRIMMIG']
ax.hist(immig_elig, bins=range(1980, 2017, 2), alpha=0.7, label='DACA-eligible', density=True)
ax.hist(immig_nonelig, bins=range(1980, 2017, 2), alpha=0.5, label='Non-eligible', density=True)
ax.axvline(x=2007, color='red', linestyle='--', linewidth=2, label='Continuous presence cutoff')
ax.set_xlabel('Year of Immigration')
ax.set_ylabel('Density')
ax.set_title('Immigration Year Distribution')
ax.legend()

# Sample sizes by year
ax = axes[1, 1]
sample_sizes = df_target.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_sizes.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Sample Size by Year and Eligibility')
ax.legend(['Non-eligible', 'DACA-eligible'])
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('figure4_composition.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure4_composition.png")

# ============================================================================
# Figure 5: Heterogeneity
# ============================================================================
print("Figure 5: Heterogeneity plot...")

# Effects from heterogeneity analysis
categories = ['Overall', 'Male', 'Female', 'Less than HS', 'HS or more', 'Ages 18-35']
effects = [0.0349, 0.0332, 0.0301, 0.0233, 0.0310, 0.0237]
ses = [0.0035, 0.0046, 0.0051, 0.0050, 0.0048, 0.0042]

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(categories))
ax.barh(y_pos, effects, xerr=[1.96*s for s in ses], capsize=5, alpha=0.7, color='steelblue')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(categories)
ax.set_xlabel('Effect on Full-time Employment (percentage points)')
ax.set_title('Heterogeneity in DACA Effect by Subgroup')
ax.set_xlim(-0.01, 0.06)

# Add value labels
for i, (e, s) in enumerate(zip(effects, ses)):
    ax.text(e + 1.96*s + 0.002, i, f'{e:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.close()
print("   Saved figure5_heterogeneity.png")

print("\nAll figures created successfully!")

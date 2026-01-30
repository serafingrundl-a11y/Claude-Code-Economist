"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (8, 5)

print("Creating figures for DACA replication report...")

# Load event study results
event_df = pd.read_csv('event_study_results.csv')
print("\nEvent study data:")
print(event_df)

# Figure 1: Event Study Plot
fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_lower = event_df['CI_Lower'].values
ci_upper = event_df['CI_Upper'].values

# Add reference year (2011) with zero effect
years_full = np.concatenate([[2006, 2007, 2008, 2009, 2010, 2011], [2013, 2014, 2015, 2016]])
coefs_full = np.concatenate([coefs[:5], [0], coefs[5:]])
ci_lower_full = np.concatenate([ci_lower[:5], [0], ci_lower[5:]])
ci_upper_full = np.concatenate([ci_upper[:5], [0], ci_upper[5:]])

# Plot coefficients with confidence intervals
ax.errorbar(years_full, coefs_full,
            yerr=[coefs_full - ci_lower_full, ci_upper_full - coefs_full],
            fmt='o', markersize=8, capsize=4, capthick=2, linewidth=2,
            color='navy', ecolor='steelblue')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add vertical line for DACA implementation (between 2011 and 2012)
ax.axvline(x=2011.5, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')

# Shade pre-period
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA Period')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA Period')

ax.set_xlabel('Year')
ax.set_ylabel('Effect on Full-Time Employment (relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("Saved figure1_event_study.png/pdf")
plt.close()

# Load main data for additional figures
print("\nLoading main data for additional figures...")
df = pd.read_csv('data/data.csv')

# Filter to Hispanic-Mexican Mexican-born non-citizens
df_mex = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)].copy()

# Create eligibility criteria
df_mex['age_2012'] = 2012 - df_mex['BIRTHYR']
df_mex.loc[df_mex['BIRTHQTR'].isin([3, 4]), 'age_2012'] = df_mex['age_2012'] - 1
df_mex['under31_2012'] = (df_mex['age_2012'] < 31).astype(int)
df_mex['age_at_arrival'] = df_mex['YRIMMIG'] - df_mex['BIRTHYR']
df_mex['arrived_before_16'] = (df_mex['age_at_arrival'] < 16).astype(int)
df_mex.loc[df_mex['YRIMMIG'] == 0, 'arrived_before_16'] = np.nan
df_mex['in_us_since_2007'] = (df_mex['YRIMMIG'] <= 2007).astype(int)
df_mex.loc[df_mex['YRIMMIG'] == 0, 'in_us_since_2007'] = np.nan

df_mex['daca_eligible'] = (
    (df_mex['under31_2012'] == 1) &
    (df_mex['arrived_before_16'] == 1) &
    (df_mex['in_us_since_2007'] == 1)
).astype(float)
df_mex.loc[df_mex['YRIMMIG'] == 0, 'daca_eligible'] = np.nan

df_mex['fulltime'] = (df_mex['UHRSWORK'] >= 35).astype(int)

# Filter working age and valid data
df_working = df_mex[(df_mex['AGE'] >= 16) & (df_mex['AGE'] <= 64)].copy()
df_working = df_working.dropna(subset=['daca_eligible'])

# Figure 2: Full-time employment rates by group over time
fig, ax = plt.subplots(figsize=(10, 6))

years_all = sorted(df_working['YEAR'].unique())
elig_rates = df_working[df_working['daca_eligible']==1].groupby('YEAR')['fulltime'].mean()
inelig_rates = df_working[df_working['daca_eligible']==0].groupby('YEAR')['fulltime'].mean()

ax.plot(elig_rates.index, elig_rates.values, 'o-', linewidth=2, markersize=8,
        color='blue', label='DACA Eligible')
ax.plot(inelig_rates.index, inelig_rates.values, 's-', linewidth=2, markersize=8,
        color='red', label='DACA Ineligible')

ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')
ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.set_xticks(years_all)
ax.legend(loc='best')
ax.set_ylim(0.3, 0.7)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
print("Saved figure2_trends.png/pdf")
plt.close()

# Figure 3: Sample distribution by age and eligibility
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-DACA
pre_elig = df_working[(df_working['daca_eligible']==1) & (df_working['YEAR'] < 2012)]['AGE']
pre_inelig = df_working[(df_working['daca_eligible']==0) & (df_working['YEAR'] < 2012)]['AGE']

axes[0].hist(pre_elig, bins=range(16, 66, 2), alpha=0.6, label='Eligible', color='blue', density=True)
axes[0].hist(pre_inelig, bins=range(16, 66, 2), alpha=0.6, label='Ineligible', color='red', density=True)
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Density')
axes[0].set_title('Age Distribution Pre-DACA (2006-2011)')
axes[0].legend()

# Post-DACA
post_elig = df_working[(df_working['daca_eligible']==1) & (df_working['YEAR'] >= 2013)]['AGE']
post_inelig = df_working[(df_working['daca_eligible']==0) & (df_working['YEAR'] >= 2013)]['AGE']

axes[1].hist(post_elig, bins=range(16, 66, 2), alpha=0.6, label='Eligible', color='blue', density=True)
axes[1].hist(post_inelig, bins=range(16, 66, 2), alpha=0.6, label='Ineligible', color='red', density=True)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Density')
axes[1].set_title('Age Distribution Post-DACA (2013-2016)')
axes[1].legend()

plt.tight_layout()
plt.savefig('figure3_age_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_age_distribution.pdf', bbox_inches='tight')
print("Saved figure3_age_distribution.png/pdf")
plt.close()

# Figure 4: Regression coefficients comparison
fig, ax = plt.subplots(figsize=(10, 6))

results_df = pd.read_csv('regression_results.csv')
models = ['Basic', '+ Demog.', '+ Year FE', '+ State FE', '+ Family', 'Weighted']
coefs = results_df['Coefficient'].values
ci_lower = results_df['CI_Lower'].values
ci_upper = results_df['CI_Upper'].values

y_pos = np.arange(len(models))
ax.barh(y_pos, coefs, xerr=[coefs - ci_lower, ci_upper - coefs],
        align='center', color='steelblue', capsize=4, alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)')
ax.set_title('DACA Effect Estimates Across Model Specifications')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

# Add coefficient labels
for i, (c, lo, hi) in enumerate(zip(coefs, ci_lower, ci_upper)):
    ax.text(c + 0.005, i, f'{c:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficients.pdf', bbox_inches='tight')
print("Saved figure4_coefficients.png/pdf")
plt.close()

print("\nAll figures created successfully!")

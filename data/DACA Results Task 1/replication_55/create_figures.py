"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Figure 1: Event Study Plot
print("Creating Figure 1: Event Study...")
event_df = pd.read_csv('event_study_results.csv')

# Add 2011 as reference year with 0 coefficient
ref_year = pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'SE': [0]})
event_df = pd.concat([event_df, ref_year]).sort_values('Year').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate confidence intervals
event_df['CI_lower'] = event_df['Coefficient'] - 1.96 * event_df['SE']
event_df['CI_upper'] = event_df['Coefficient'] + 1.96 * event_df['SE']

# Plot
ax.errorbar(event_df['Year'], event_df['Coefficient'],
            yerr=1.96 * event_df['SE'],
            fmt='o-', capsize=4, capthick=2, markersize=8,
            color='#2C3E50', ecolor='#7F8C8D')

# Add horizontal line at 0
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='red', linestyle=':', alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Effect on Full-time Employment)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_event_study.png and .pdf")

# Figure 2: Trends in Full-time Employment by Group
print("Creating Figure 2: Trends...")
sample_df = pd.read_csv('sample_by_year.csv')

# Calculate full-time employment rates by year and eligibility
# We need to reload the data for this
chunks = []
for chunk in pd.read_csv('data/data.csv', chunksize=1000000,
                          usecols=['YEAR', 'PERWT', 'AGE', 'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
                                  'BIRTHYR', 'UHRSWORK']):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200) &
                     (chunk['CITIZEN'] == 3) & (chunk['YRIMMIG'] > 0) &
                     (chunk['AGE'] >= 16) & (chunk['AGE'] <= 64)]
    if len(filtered) > 0:
        # Create eligibility indicator
        filtered = filtered.copy()
        filtered['age_at_arrival'] = filtered['YRIMMIG'] - filtered['BIRTHYR']
        filtered['daca_eligible'] = ((filtered['age_at_arrival'] < 16) &
                                     (filtered['BIRTHYR'] >= 1982) &
                                     (filtered['YRIMMIG'] <= 2007)).astype(int)
        filtered['fulltime'] = (filtered['UHRSWORK'] >= 35).astype(int)
        chunks.append(filtered[['YEAR', 'daca_eligible', 'fulltime', 'PERWT']])

df = pd.concat(chunks, ignore_index=True)

# Calculate weighted means by year and eligibility
trends = df.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()
trends.columns = ['Ineligible', 'Eligible']

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(trends.index, trends['Eligible'] * 100, 'o-',
        label='DACA Eligible', color='#E74C3C', markersize=8, linewidth=2)
ax.plot(trends.index, trends['Ineligible'] * 100, 's-',
        label='DACA Ineligible', color='#3498DB', markersize=8, linewidth=2)

ax.axvline(x=2012, color='gray', linestyle=':', alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate (%)')
ax.set_title('Trends in Full-time Employment by DACA Eligibility Status')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='best')
ax.set_ylim([30, 70])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_trends.png and .pdf")

# Figure 3: Main Results - Coefficient Plot
print("Creating Figure 3: Main Results...")
results_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(8, 5))

models = results_df['Model'].tolist()
coefs = results_df['Coefficient'].tolist()
ses = results_df['SE'].tolist()

y_pos = np.arange(len(models))

ax.barh(y_pos, coefs, xerr=[1.96 * se for se in ses],
        capsize=4, color='#2980B9', alpha=0.8, edgecolor='#1A5276')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DID Coefficient (Effect on Full-time Employment Probability)')
ax.set_title('Effect of DACA Eligibility on Full-time Employment\nAcross Model Specifications')

# Add coefficient values as text
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.text(coef + 0.005, i, f'{coef:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_main_results.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_main_results.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_main_results.png and .pdf")

# Figure 4: Robustness Checks
print("Creating Figure 4: Robustness...")
robust_df = pd.read_csv('robustness_results.csv')

fig, ax = plt.subplots(figsize=(8, 5))

models = robust_df['Model'].tolist()
coefs = robust_df['Coefficient'].tolist()
ses = robust_df['SE'].tolist()

y_pos = np.arange(len(models))

colors = ['#27AE60' if m != 'Placebo' else '#E74C3C' for m in models]

ax.barh(y_pos, coefs, xerr=[1.96 * se for se in ses],
        capsize=4, color=colors, alpha=0.8)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Add main estimate line for reference
ax.axvline(x=0.036, color='#2980B9', linestyle='-', alpha=0.5,
           label=f'Main Estimate (0.036)')

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DID Coefficient')
ax.set_title('Robustness Checks')
ax.legend(loc='best')

# Add coefficient values
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.text(max(coef + 0.005, 0.005), i, f'{coef:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_robustness.png and .pdf")

print("\nAll figures created successfully!")

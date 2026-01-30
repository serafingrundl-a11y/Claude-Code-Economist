"""
Create figures for DACA replication study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Read results
event_df = pd.read_csv('event_study_results.csv')
results_df = pd.read_csv('regression_results.csv')
sample_df = pd.read_csv('sample_by_year.csv')

# ============================================================================
# Figure 1: Event Study Plot
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_lower = event_df['CI_Lower'].values
ci_upper = event_df['CI_Upper'].values

# Add reference year 2011 with coefficient 0
years_plot = np.insert(years, 5, 2011)
coefs_plot = np.insert(coefs, 5, 0)
ci_lower_plot = np.insert(ci_lower, 5, 0)
ci_upper_plot = np.insert(ci_upper, 5, 0)

# Calculate error bars
errors = np.vstack([coefs_plot - ci_lower_plot, ci_upper_plot - coefs_plot])

ax.errorbar(years_plot, coefs_plot, yerr=errors, fmt='o-', capsize=4,
            color='#2E86AB', markersize=8, linewidth=2, capthick=2)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks(years_plot)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
print("Figure 1 saved: Event study plot")

# ============================================================================
# Figure 2: Full-time Employment Trends by Group
# ============================================================================
# Read data again for this
df = pd.read_csv('data/data.csv')

# Apply filters
df = df[df['HISPAN'] == 1]
df = df[df['BPL'] == 200]
df = df[df['CITIZEN'] == 3]

# Calculate DACA eligibility
df['age_jun2012'] = 2012 - df['BIRTHYR']
df.loc[df['BIRTHQTR'].isin([3, 4]), 'age_jun2012'] -= 1
df['under_31_jun2012'] = df['age_jun2012'] < 31
df['age_at_arrival'] = df['YRIMMIG'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_arrival'] < 16) & (df['YRIMMIG'] > 0)
df['in_us_since_2007'] = (df['YRIMMIG'] <= 2007) & (df['YRIMMIG'] > 0)
df['daca_eligible'] = df['under_31_jun2012'] & df['arrived_before_16'] & df['in_us_since_2007']

# Employment outcome
df['employed'] = (df['EMPSTAT'] == 1).astype(int)
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['employed'] == 1)).astype(int)

# Restrict to working age
df = df[(df['AGE'] >= 18) & (df['AGE'] <= 64)]

# Calculate yearly means
trends = df.groupby(['YEAR', 'daca_eligible'])['fulltime'].mean().unstack()
trends.columns = ['Ineligible', 'Eligible']

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(trends.index, trends['Eligible'], 'o-', color='#E94F37', linewidth=2,
        markersize=8, label='DACA-Eligible')
ax.plot(trends.index, trends['Ineligible'], 's-', color='#2E86AB', linewidth=2,
        markersize=8, label='DACA-Ineligible')
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility', fontsize=14)
ax.set_xticks(trends.index)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 0.65)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
print("Figure 2 saved: Trends plot")

# ============================================================================
# Figure 3: Sample Size by Year and Group
# ============================================================================
sample_counts = df.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_counts.columns = ['Ineligible', 'Eligible']

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(sample_counts.index))

bars1 = ax.bar(x - width/2, sample_counts['Eligible']/1000, width,
               label='DACA-Eligible', color='#E94F37')
bars2 = ax.bar(x + width/2, sample_counts['Ineligible']/1000, width,
               label='DACA-Ineligible', color='#2E86AB')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sample Size (thousands)', fontsize=12)
ax.set_title('Sample Size by Year and DACA Eligibility', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(sample_counts.index)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_sample_size.pdf', bbox_inches='tight')
print("Figure 3 saved: Sample size plot")

# ============================================================================
# Figure 4: Coefficient Comparison Across Models
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = results_df['Model'].values
coefs = results_df['DiD_Coefficient'].values
errors = 1.96 * results_df['Std_Error'].values

colors = ['#2E86AB' if 'FE' not in m else '#E94F37' for m in models]

bars = ax.barh(range(len(models)), coefs, xerr=errors, capsize=5,
               color=colors, alpha=0.8)
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficients.pdf', bbox_inches='tight')
print("Figure 4 saved: Coefficient comparison plot")

print("\nAll figures created successfully!")

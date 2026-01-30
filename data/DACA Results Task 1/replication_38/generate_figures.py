"""
Generate figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json

os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_38")

# Load results
with open('results_38.json', 'r') as f:
    results = json.load(f)

#############################################################################
# Figure 1: Event Study Plot
#############################################################################
print("Creating Figure 1: Event Study...")

# Event study data
event_data = [
    {'year': 2006, 'coef': 0.0326, 'se': 0.0158},
    {'year': 2007, 'coef': 0.0254, 'se': 0.0107},
    {'year': 2008, 'coef': 0.0392, 'se': 0.0135},
    {'year': 2009, 'coef': 0.0197, 'se': 0.0115},
    {'year': 2010, 'coef': 0.0206, 'se': 0.0132},
    {'year': 2011, 'coef': 0.0000, 'se': 0.0000},
    {'year': 2013, 'coef': 0.0170, 'se': 0.0108},
    {'year': 2014, 'coef': 0.0202, 'se': 0.0175},
    {'year': 2015, 'coef': 0.0224, 'se': 0.0162},
    {'year': 2016, 'coef': 0.0357, 'se': 0.0129},
]

years = [d['year'] for d in event_data]
coefs = [d['coef'] for d in event_data]
ses = [d['se'] for d in event_data]
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(years, coefs, yerr=[np.array(coefs)-np.array(ci_lower),
            np.array(ci_upper)-np.array(coefs)],
            fmt='o-', capsize=5, capthick=2, markersize=8,
            color='navy', ecolor='gray', label='Point Estimate')

# Reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation (2012)')

# Labels and formatting
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Relative to 2011)', fontsize=14)
ax.set_xticks(years)
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure1_event_study.png")

#############################################################################
# Figure 2: Trends in Full-Time Employment by Treatment Status
#############################################################################
print("Creating Figure 2: Trends plot...")

# Re-load data to create trends
chunks = []
for chunk in pd.read_csv('data/data.csv', chunksize=500000, low_memory=False):
    chunk_filtered = chunk[
        (chunk['HISPAN'] == 1) &
        (chunk['BPL'] == 200)
    ]
    chunks.append(chunk_filtered)

df = pd.concat(chunks, ignore_index=True)

# Create variables
df['year'] = df['YEAR'].astype(int)
df['birthyr'] = df['BIRTHYR'].astype(int)
df['age'] = df['AGE'].astype(int)
df['yrimmig'] = df['YRIMMIG'].replace(0, np.nan)
df['age_at_arrival'] = df['yrimmig'] - df['birthyr']
df['noncitizen'] = (df['CITIZEN'] == 3).astype(int)
df['birthqtr'] = df['BIRTHQTR'].fillna(0).astype(int)
df['age_june_2012'] = 2012 - df['birthyr']
df.loc[df['birthqtr'].isin([3, 4]), 'age_june_2012'] -= 1
df['arrived_before_16'] = (df['age_at_arrival'] < 16).astype(int)
df['under_31_2012'] = (df['age_june_2012'] < 31).astype(int)
df['in_us_by_2007'] = (df['yrimmig'] <= 2007).astype(int)
df['uhrswork'] = df['UHRSWORK'].fillna(0).astype(int)
df['fulltime'] = (df['uhrswork'] >= 35).astype(int)
df['perwt'] = df['PERWT'].astype(float)

# Define groups
df['treat'] = (
    (df['noncitizen'] == 1) &
    (df['arrived_before_16'] == 1) &
    (df['under_31_2012'] == 1) &
    (df['in_us_by_2007'] == 1)
).astype(int)

df['control'] = (
    (df['noncitizen'] == 1) &
    (df['arrived_before_16'] == 1) &
    (df['in_us_by_2007'] == 1) &
    (df['under_31_2012'] == 0)
).astype(int)

# Restrict sample
df_analysis = df[
    (df['age'] >= 18) &
    (df['age'] <= 55) &
    (df['noncitizen'] == 1) &
    (df['yrimmig'].notna()) &
    ((df['treat'] == 1) | (df['control'] == 1))
].copy()

# Calculate weighted means by year and group
trends = df_analysis.groupby(['year', 'treat']).apply(
    lambda x: np.average(x['fulltime'], weights=x['perwt'])
).reset_index(name='fulltime_rate')

treat_trends = trends[trends['treat'] == 1]
control_trends = trends[trends['treat'] == 0]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(treat_trends['year'], treat_trends['fulltime_rate'],
        'o-', linewidth=2, markersize=8, color='blue', label='Treatment (DACA Eligible)')
ax.plot(control_trends['year'], control_trends['fulltime_rate'],
        's--', linewidth=2, markersize=8, color='red', label='Control (Age Ineligible)')

ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.58), fontsize=10, ha='center')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Trends in Full-Time Employment by DACA Eligibility Status', fontsize=14)
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim([0.45, 0.75])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure2_trends.png")

#############################################################################
# Figure 3: Coefficient Comparison Across Specifications
#############################################################################
print("Creating Figure 3: Coefficient comparison...")

models = ['Model 1:\nBasic DiD', 'Model 2:\n+ Controls',
          'Model 3:\n+ State FE', 'Model 4:\n+ Year FE\n(Preferred)']
coefs = [0.0800, 0.0162, 0.0152, 0.0012]
ses = [0.0054, 0.0057, 0.0059, 0.0053]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.6

bars = ax.bar(x, coefs, width, yerr=[1.96*s for s in ses],
              capsize=5, color=['lightblue', 'steelblue', 'royalblue', 'navy'],
              edgecolor='black', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.annotate(f'{c:.4f}', xy=(i, c + 1.96*s + 0.005),
                ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure3_coefficients.png")

#############################################################################
# Figure 4: Gender Heterogeneity
#############################################################################
print("Creating Figure 4: Gender heterogeneity...")

genders = ['Male', 'Female']
gender_coefs = [-0.0216, 0.0214]
gender_ses = [0.0071, 0.0076]

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(genders))
width = 0.5

colors = ['#3498db', '#e74c3c']
bars = ax.bar(x, gender_coefs, width, yerr=[1.96*s for s in gender_ses],
              capsize=8, color=colors, edgecolor='black', alpha=0.8)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('Effect of DACA Eligibility on Full-Time Employment by Gender', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(genders, fontsize=12)
ax.grid(True, alpha=0.3, axis='y')

for i, (c, s) in enumerate(zip(gender_coefs, gender_ses)):
    pos = c + 1.96*s + 0.005 if c > 0 else c - 1.96*s - 0.01
    ax.annotate(f'{c:.4f}', xy=(i, pos), ha='center', fontsize=11)

plt.tight_layout()
plt.savefig('figure4_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: figure4_gender.png")

print("\nAll figures generated successfully!")

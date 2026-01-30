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
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study...")

event_results = pd.read_csv('event_study_results.csv')

# Add 2012 as a gap year
fig, ax = plt.subplots(figsize=(10, 6))

years = event_results['Year'].values
coeffs = event_results['Coefficient'].values
ses = event_results['SE'].values

# Calculate 95% CI
ci_upper = coeffs + 1.96 * ses
ci_lower = coeffs - 1.96 * ses

# Separate pre and post
pre_mask = years <= 2011
post_mask = years >= 2013

# Plot pre-treatment
ax.errorbar(years[pre_mask], coeffs[pre_mask], yerr=1.96*ses[pre_mask],
           fmt='o', color='blue', capsize=4, capthick=2, markersize=8, label='Pre-DACA')

# Plot post-treatment
ax.errorbar(years[post_mask], coeffs[post_mask], yerr=1.96*ses[post_mask],
           fmt='s', color='red', capsize=4, capthick=2, markersize=8, label='Post-DACA')

# Reference line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

# Labels and formatting
ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Treatment Group vs Control Group)')
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left')
ax.set_ylim(-0.10, 0.12)

# Add note
ax.text(0.02, 0.02, 'Notes: Coefficients represent difference between treatment and control groups relative to 2011.\n95% confidence intervals shown. 2012 excluded due to mid-year DACA implementation.',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure1_event_study.png/pdf")

# =============================================================================
# Figure 2: Full-time Employment Trends by Group
# =============================================================================
print("Creating Figure 2: Trends by Group...")

# Load raw data to create trends
df = pd.read_csv('data/data.csv')

# Apply sample restrictions
df_sample = df[(df['HISPAN'] == 1) &
               (df['BPL'] == 200) &
               (df['CITIZEN'] == 3)]
df_sample = df_sample.copy()
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample = df_sample[(df_sample['age_at_arrival'] < 16) &
                      (df_sample['YRIMMIG'] <= 2007)]

# Define treatment and control
df_sample['treatment'] = ((df_sample['BIRTHYR'] >= 1982) &
                          (df_sample['BIRTHYR'] <= 1986)).astype(int)
df_sample['control'] = ((df_sample['BIRTHYR'] >= 1977) &
                        (df_sample['BIRTHYR'] <= 1981)).astype(int)
df_sample = df_sample[(df_sample['treatment'] == 1) | (df_sample['control'] == 1)]

# Create outcome
df_sample['fulltime'] = ((df_sample['EMPSTAT'] == 1) &
                         (df_sample['UHRSWORK'] >= 35)).astype(int)

# Calculate yearly means by group
trends = df_sample.groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack() * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(trends.index, trends[0], 'o-', color='blue', linewidth=2, markersize=8, label='Control (Age 31-35 in June 2012)')
ax.plot(trends.index, trends[1], 's-', color='red', linewidth=2, markersize=8, label='Treatment (Age 26-30 in June 2012)')

# DACA implementation line
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group\n(Hispanic-Mexican, Mexican-Born, Non-Citizens)')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')
ax.set_ylim(45, 70)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure2_trends.png/pdf")

# =============================================================================
# Figure 3: Difference in Full-time Employment Over Time
# =============================================================================
print("Creating Figure 3: Treatment-Control Difference...")

# Calculate difference
diff = trends[1] - trends[0]

fig, ax = plt.subplots(figsize=(10, 6))

# Pre-treatment periods
pre_years = [y for y in diff.index if y <= 2011]
post_years = [y for y in diff.index if y >= 2013]

ax.bar([y for y in pre_years], [diff[y] for y in pre_years], color='blue', alpha=0.7, label='Pre-DACA')
ax.bar([y for y in post_years], [diff[y] for y in post_years], color='red', alpha=0.7, label='Post-DACA')
if 2012 in diff.index:
    ax.bar([2012], [diff[2012]], color='gray', alpha=0.5, label='2012 (Excluded)')

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Difference in Full-Time Employment Rate (pp)')
ax.set_title('Difference in Full-Time Employment: Treatment Minus Control\n(Treatment: Age 26-30; Control: Age 31-35 in June 2012)')
ax.set_xticks(range(2006, 2017))
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure3_difference.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_difference.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure3_difference.png/pdf")

# =============================================================================
# Figure 4: Sample Composition
# =============================================================================
print("Creating Figure 4: Sample Composition...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sample sizes by year and group
sample_sizes = df_sample.groupby(['YEAR', 'treatment']).size().unstack()

ax1 = axes[0]
width = 0.35
x = np.arange(len(sample_sizes.index))
ax1.bar(x - width/2, sample_sizes[0], width, color='blue', alpha=0.7, label='Control')
ax1.bar(x + width/2, sample_sizes[1], width, color='red', alpha=0.7, label='Treatment')
ax1.set_xlabel('Year')
ax1.set_ylabel('Sample Size')
ax1.set_title('Sample Size by Year and Group')
ax1.set_xticks(x)
ax1.set_xticklabels(sample_sizes.index)
ax1.legend()
ax1.axvline(x=6, color='gray', linestyle='--', alpha=0.5)  # 2012

# Gender composition
ax2 = axes[1]
gender_comp = df_sample.groupby(['treatment', 'SEX']).size().unstack()
gender_comp.columns = ['Male', 'Female']
gender_comp.index = ['Control', 'Treatment']
gender_pct = gender_comp.div(gender_comp.sum(axis=1), axis=0) * 100

gender_pct.plot(kind='bar', ax=ax2, color=['steelblue', 'coral'], alpha=0.8)
ax2.set_xlabel('Group')
ax2.set_ylabel('Percentage')
ax2.set_title('Gender Composition by Group')
ax2.set_xticklabels(['Control (31-35)', 'Treatment (26-30)'], rotation=0)
ax2.legend(title='Gender')

plt.tight_layout()
plt.savefig('figure4_composition.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_composition.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure4_composition.png/pdf")

print("\nAll figures created successfully!")

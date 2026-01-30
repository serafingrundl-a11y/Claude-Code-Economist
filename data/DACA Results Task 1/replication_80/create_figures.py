"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import pickle

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

print("Creating figures for report...")

# Load results
with open('event_study_results.pkl', 'rb') as f:
    event_results = pickle.load(f)

with open('summary_stats.pkl', 'rb') as f:
    summary_stats = pickle.load(f)

# -----------------------------------------------------------------------------
# Figure 1: Full-Time Employment Trends by DACA Eligibility
# -----------------------------------------------------------------------------
print("Creating Figure 1: Employment trends...")

ft_table = summary_stats['ft_table']

fig, ax = plt.subplots(figsize=(10, 6))

years = ft_table.index.values
eligible = ft_table['DACA-Eligible'].values * 100
ineligible = ft_table['Non-Eligible'].values * 100

ax.plot(years, eligible, 'o-', color='#2166AC', linewidth=2, markersize=8, label='DACA-Eligible')
ax.plot(years, ineligible, 's--', color='#B2182B', linewidth=2, markersize=8, label='Non-Eligible')

# Add vertical line at DACA implementation
ax.axvline(x=2012.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.6, 65, 'DACA\nImplementation', fontsize=10, color='gray', ha='left', va='top')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status')
ax.legend(loc='upper right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(35, 70)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_trends.png/pdf")

# -----------------------------------------------------------------------------
# Figure 2: Event Study Plot
# -----------------------------------------------------------------------------
print("Creating Figure 2: Event study...")

years_event = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []

for year in years_event:
    if year == 2011:
        coefs.append(0)
        ses.append(0)
    else:
        coefs.append(event_results[str(year)]['coef'])
        ses.append(event_results[str(year)]['se'])

coefs = np.array(coefs) * 100  # Convert to percentage points
ses = np.array(ses) * 100

fig, ax = plt.subplots(figsize=(10, 6))

# Plot confidence intervals
ax.fill_between(years_event, coefs - 1.96*ses, coefs + 1.96*ses,
                alpha=0.2, color='#2166AC')
ax.plot(years_event, coefs, 'o-', color='#2166AC', linewidth=2, markersize=8)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 5, 'DACA', fontsize=10, color='gray', ha='left', va='bottom')

ax.set_xlabel('Year')
ax.set_ylabel('Effect on Full-Time Employment (pp)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Relative to 2011, Omitting 2012)')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks(years_event)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_eventstudy.png/pdf")

# -----------------------------------------------------------------------------
# Figure 3: Difference in Full-Time Employment Gap Over Time
# -----------------------------------------------------------------------------
print("Creating Figure 3: Employment gap...")

fig, ax = plt.subplots(figsize=(10, 6))

years = ft_table.index.values
gap = (ft_table['Non-Eligible'].values - ft_table['DACA-Eligible'].values) * 100

ax.bar(years, gap, color='#2166AC', alpha=0.7, edgecolor='black')

# Add vertical line at DACA implementation
ax.axvline(x=2012.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.6, 18, 'DACA\nImplementation', fontsize=10, color='red', ha='left', va='top')

ax.set_xlabel('Year')
ax.set_ylabel('Employment Gap (pp): Non-Eligible - Eligible')
ax.set_title('Full-Time Employment Gap Between DACA-Eligible and Non-Eligible Groups')
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure3_gap.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_gap.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_gap.png/pdf")

# -----------------------------------------------------------------------------
# Figure 4: Sample Composition
# -----------------------------------------------------------------------------
print("Creating Figure 4: Sample composition...")

sample_table = summary_stats['sample_table']

fig, ax = plt.subplots(figsize=(10, 6))

years = sample_table.index.values
eligible_n = sample_table['DACA-Eligible'].values / 1000
ineligible_n = sample_table['Non-Eligible'].values / 1000

width = 0.35
x = np.arange(len(years))

ax.bar(x - width/2, ineligible_n, width, label='Non-Eligible', color='#B2182B', alpha=0.7)
ax.bar(x + width/2, eligible_n, width, label='DACA-Eligible', color='#2166AC', alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Sample Size (thousands)')
ax.set_title('Sample Composition by Year and DACA Eligibility Status')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend()

plt.tight_layout()
plt.savefig('figure4_sample.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_sample.png/pdf")

print("\nAll figures created successfully!")

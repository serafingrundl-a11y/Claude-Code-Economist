"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load results
with open('results_summary.pkl', 'rb') as f:
    results = pickle.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# ==============================================================================
# Figure 1: Trends in Full-time Employment by DACA Eligibility
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ft_rates = results['ft_rates']
years = ft_rates.index.tolist()
not_eligible = ft_rates['Not Eligible'].values
daca_eligible = ft_rates['DACA Eligible'].values

ax.plot(years, not_eligible, 'o-', color='#1f77b4', linewidth=2, markersize=8,
        label='Not DACA Eligible')
ax.plot(years, daca_eligible, 's-', color='#d62728', linewidth=2, markersize=8,
        label='DACA Eligible')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 0.72, 'DACA\nImplementation', fontsize=10, color='gray', va='top')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Full-time Employment Trends by DACA Eligibility Status', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.35, 0.75)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Created Figure 1: Trends in Full-time Employment")

# ==============================================================================
# Figure 2: Event Study Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

event_coefs = results['event_coefs']
years_ev = [x[0] for x in event_coefs]
coefs = [x[1] for x in event_coefs]
ses = [x[2] for x in event_coefs]

# Add 2011 as reference year with 0 coefficient
years_ev.insert(5, 2011)
coefs.insert(5, 0)
ses.insert(5, 0)

ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.errorbar(years_ev, coefs, yerr=[np.array(coefs)-np.array(ci_low),
                                     np.array(ci_high)-np.array(coefs)],
            fmt='o', color='#1f77b4', capsize=4, capthick=2, markersize=8, linewidth=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Shade pre and post periods
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment',
             fontsize=14, fontweight='bold')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("Created Figure 2: Event Study")

# ==============================================================================
# Figure 3: Difference-in-Differences Visual
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

pre_post = results['pre_post_table']

# Data for grouped bar chart
periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
not_elig_rates = [pre_post.loc['Pre-DACA (2006-2011)', 'Not Eligible'],
                  pre_post.loc['Post-DACA (2013-2016)', 'Not Eligible']]
elig_rates = [pre_post.loc['Pre-DACA (2006-2011)', 'DACA Eligible'],
              pre_post.loc['Post-DACA (2013-2016)', 'DACA Eligible']]

x = np.arange(len(periods))
width = 0.35

bars1 = ax.bar(x - width/2, not_elig_rates, width, label='Not DACA Eligible',
               color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, elig_rates, width, label='DACA Eligible',
               color='#d62728', alpha=0.8)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Full-time Employment by Period and DACA Eligibility', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.75)

# Add DiD calculation annotation
did = (elig_rates[1] - elig_rates[0]) - (not_elig_rates[1] - not_elig_rates[0])
ax.annotate(f'DiD Effect: {did:.3f}', xy=(0.5, 0.55), xycoords='axes fraction',
            fontsize=12, ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()
print("Created Figure 3: DiD Bar Chart")

print("\nAll figures created successfully.")

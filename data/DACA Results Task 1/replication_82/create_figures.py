"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_82")

# Load results
with open('results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study...")

event_data = results['event_study']
years = sorted([int(y) for y in event_data.keys()])
coefficients = [event_data[str(y)] for y in years]

# Add 2012 as reference year (coefficient = 0)
years_full = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
coef_full = []
for y in years_full:
    if y == 2012:
        coef_full.append(0)
    elif str(y) in event_data:
        coef_full.append(event_data[str(y)])
    else:
        coef_full.append(np.nan)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients
ax.plot(years_full, coef_full, 'bo-', markersize=8, linewidth=2, label='Point Estimate')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Coefficient (Effect on Full-time Employment)', fontsize=14)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment', fontsize=16)
ax.legend(loc='upper left')
ax.set_xticks(years_full)
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()
print("   Saved figure1_event_study.png/pdf")

# =============================================================================
# Figure 2: Full-time Employment Rates by Group Over Time
# =============================================================================
print("Creating Figure 2: Trends by Group...")

# Recalculate from summary data
summary = results['summary_stats']

# Pre-period
pre_control = summary['daca0_post0']['fulltime_weighted']
pre_treat = summary['daca1_post0']['fulltime_weighted']
# Post-period
post_control = summary['daca0_post1']['fulltime_weighted']
post_treat = summary['daca1_post1']['fulltime_weighted']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot both groups
ax.plot([0, 1], [pre_control, post_control], 'bs-', markersize=10, linewidth=2.5, label='Control (Not DACA Eligible)')
ax.plot([0, 1], [pre_treat, post_treat], 'ro-', markersize=10, linewidth=2.5, label='Treatment (DACA Eligible)')

ax.set_xlim(-0.2, 1.2)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-time Employment Rate', fontsize=14)
ax.set_title('Full-time Employment Trends by DACA Eligibility', fontsize=16)
ax.legend(loc='lower right', fontsize=12)
ax.set_ylim(0.3, 0.7)

# Add grid
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()
print("   Saved figure2_trends.png/pdf")

# =============================================================================
# Figure 3: Robustness Checks
# =============================================================================
print("Creating Figure 3: Robustness Checks...")

robust = results['robustness']
preferred = results['preferred_estimate']['coefficient']
preferred_se = results['preferred_estimate']['std_error']

labels = ['Preferred\nEstimate', 'Include\n2012', 'Employment\nOutcome', 'Ages\n18-30', 'Males\nOnly', 'Females\nOnly']
coeffs = [preferred, robust['include_2012'], robust['employment_outcome'],
          robust['ages_18_30'], robust['males_only'], robust['females_only']]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(labels))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#95190C']

bars = ax.bar(x, coeffs, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Add value labels
for i, (bar, coef) in enumerate(zip(bars, coeffs)):
    height = bar.get_height()
    ax.annotate(f'{coef:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('DiD Coefficient', fontsize=14)
ax.set_title('Robustness Checks: Difference-in-Differences Estimates', fontsize=16)

plt.tight_layout()
plt.savefig('figure3_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_robustness.pdf', bbox_inches='tight')
plt.close()
print("   Saved figure3_robustness.png/pdf")

# =============================================================================
# Figure 4: Sample Composition
# =============================================================================
print("Creating Figure 4: Sample Composition...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Sample sizes by group
groups = ['Control\nPre-DACA', 'Control\nPost-DACA', 'Treated\nPre-DACA', 'Treated\nPost-DACA']
sizes = [summary['daca0_post0']['n'], summary['daca0_post1']['n'],
         summary['daca1_post0']['n'], summary['daca1_post1']['n']]
sizes_weighted = [summary['daca0_post0']['n_weighted']/1e6, summary['daca0_post1']['n_weighted']/1e6,
                  summary['daca1_post0']['n_weighted']/1e6, summary['daca1_post1']['n_weighted']/1e6]

x = np.arange(len(groups))
colors = ['#4A90A4', '#4A90A4', '#E07A5F', '#E07A5F']
hatches = ['', '//', '', '//']

bars = axes[0].bar(x, sizes_weighted, color=colors, edgecolor='black', linewidth=1.5)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

axes[0].set_xticks(x)
axes[0].set_xticklabels(groups, fontsize=10)
axes[0].set_ylabel('Population (Millions)', fontsize=12)
axes[0].set_title('A. Weighted Sample Size by Group', fontsize=14)

for i, (bar, val) in enumerate(zip(bars, sizes_weighted)):
    axes[0].annotate(f'{val:.1f}M', xy=(bar.get_x() + bar.get_width()/2, val),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

# Panel B: Characteristics comparison
chars = ['Age', 'Female\n(%)', 'Married\n(%)', 'HS+\n(%)']
control_pre = [summary['daca0_post0']['age_mean'],
               summary['daca0_post0']['female_mean']*100,
               summary['daca0_post0']['married_mean']*100,
               summary['daca0_post0']['educ_hs_mean']*100]
treat_pre = [summary['daca1_post0']['age_mean'],
             summary['daca1_post0']['female_mean']*100,
             summary['daca1_post0']['married_mean']*100,
             summary['daca1_post0']['educ_hs_mean']*100]

x = np.arange(len(chars))
width = 0.35

bars1 = axes[1].bar(x - width/2, control_pre, width, label='Control', color='#4A90A4', edgecolor='black')
bars2 = axes[1].bar(x + width/2, treat_pre, width, label='Treatment', color='#E07A5F', edgecolor='black')

axes[1].set_xticks(x)
axes[1].set_xticklabels(chars, fontsize=10)
axes[1].set_ylabel('Value', fontsize=12)
axes[1].set_title('B. Pre-Period Characteristics by Group', fontsize=14)
axes[1].legend()

plt.tight_layout()
plt.savefig('figure4_sample.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample.pdf', bbox_inches='tight')
plt.close()
print("   Saved figure4_sample.png/pdf")

print("\nAll figures created successfully!")

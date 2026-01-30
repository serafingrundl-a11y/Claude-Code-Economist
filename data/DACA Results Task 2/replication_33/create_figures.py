"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.figsize'] = (8, 5)

# Load results
yearly = pd.read_csv('yearly_results.csv')
event = pd.read_csv('event_study_results.csv')
main = pd.read_csv('main_results.csv')

print("Creating figures...")

# Figure 1: Parallel trends plot
fig, ax = plt.subplots(figsize=(10, 6))

years = yearly['Year'].values
treat_rate = yearly['Treat_FT_Weighted'].values
control_rate = yearly['Control_FT_Weighted'].values

ax.plot(years, treat_rate, 'o-', color='#2166ac', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, control_rate, 's-', color='#b2182b', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line at DACA implementation
ax.axvline(x=2012.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012.5, 0.72), fontsize=10, ha='center', color='gray')

# Shade post-treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.legend(loc='lower left', frameon=True)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.55, 0.76)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
print("  Saved figure1_parallel_trends.png/pdf")
plt.close()

# Figure 2: Difference in full-time rates over time
fig, ax = plt.subplots(figsize=(10, 6))

diff = yearly['Diff_Weighted'].values

colors = ['#2166ac' if y < 2012 else '#1a9850' for y in years]
bars = ax.bar(years, diff, color=colors, edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Add annotations
pre_mean = np.mean(diff[years < 2012])
post_mean = np.mean(diff[years > 2012])
ax.axhline(y=pre_mean, xmin=0, xmax=0.55, color='#2166ac', linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(y=post_mean, xmin=0.6, xmax=1, color='#1a9850', linestyle=':', linewidth=2, alpha=0.7)

ax.annotate(f'Pre-DACA mean: {pre_mean:.3f}', xy=(2008, pre_mean+0.01), fontsize=10, color='#2166ac')
ax.annotate(f'Post-DACA mean: {post_mean:.3f}', xy=(2014, post_mean+0.01), fontsize=10, color='#1a9850')
ax.annotate(f'DiD = {post_mean - pre_mean:.3f}', xy=(2014.5, 0.06), fontsize=11, fontweight='bold')

ax.set_xlabel('Year')
ax.set_ylabel('Difference in Full-Time Rate (Treatment - Control)')
ax.set_title('Difference in Full-Time Employment Between Treatment and Control Groups')
ax.set_xticks(years)

# Legend
pre_patch = mpatches.Patch(color='#2166ac', label='Pre-DACA (2006-2011)')
post_patch = mpatches.Patch(color='#1a9850', label='Post-DACA (2013-2016)')
ax.legend(handles=[pre_patch, post_patch], loc='lower right')

plt.tight_layout()
plt.savefig('figure2_difference_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_difference_trends.pdf', bbox_inches='tight')
print("  Saved figure2_difference_trends.png/pdf")
plt.close()

# Figure 3: Event study plot
fig, ax = plt.subplots(figsize=(10, 6))

# Add reference year (2011) with 0 coefficient
event_years = list(event['Year'].values)
event_coef = list(event['Coefficient'].values)
event_ci_low = list(event['CI_low'].values)
event_ci_high = list(event['CI_high'].values)

# Insert 2011 as reference
ref_idx = event_years.index(2010) + 1
event_years.insert(ref_idx, 2011)
event_coef.insert(ref_idx, 0)
event_ci_low.insert(ref_idx, 0)
event_ci_high.insert(ref_idx, 0)

event_years = np.array(event_years)
event_coef = np.array(event_coef)
event_ci_low = np.array(event_ci_low)
event_ci_high = np.array(event_ci_high)

# Calculate error bars
errors = np.array([event_coef - event_ci_low, event_ci_high - event_coef])

# Color by pre/post
colors = ['#2166ac' if y <= 2011 else '#1a9850' for y in event_years]

ax.errorbar(event_years, event_coef, yerr=errors, fmt='o', capsize=5,
            color='black', markersize=8, linewidth=1.5, capthick=1.5)

for i, (y, c, col) in enumerate(zip(event_years, event_coef, colors)):
    ax.scatter(y, c, color=col, s=80, zorder=5)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA\nImplemented\n(mid-2012)', xy=(2011.5, 0.1), fontsize=9, ha='center', color='red')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Treatment Effect by Year\n(Reference Year: 2011)')
ax.set_xticks(event_years)

plt.tight_layout()
plt.savefig('figure3_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_event_study.pdf', bbox_inches='tight')
print("  Saved figure3_event_study.png/pdf")
plt.close()

# Figure 4: Model comparison
fig, ax = plt.subplots(figsize=(10, 6))

models = main['Model'].values
estimates = main['Estimate'].values
ci_low = main['CI_low'].values
ci_high = main['CI_high'].values

y_pos = np.arange(len(models))
errors = np.array([estimates - ci_low, ci_high - estimates])

ax.barh(y_pos, estimates, xerr=errors, height=0.6, color='#2166ac', capsize=5, edgecolor='black')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment)')
ax.set_title('DACA Effect Estimates Across Model Specifications')

# Add value labels
for i, (est, low, high) in enumerate(zip(estimates, ci_low, ci_high)):
    ax.annotate(f'{est:.3f}', xy=(est + 0.005, i), va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
print("  Saved figure4_model_comparison.png/pdf")
plt.close()

# Figure 5: Sample flow diagram (text-based representation)
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')

# Create boxes
box_style = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', linewidth=1.5)
arrow_style = dict(arrowstyle='->', color='black', linewidth=1.5)

# Text for sample flow
flow_text = """
SAMPLE SELECTION FLOW DIAGRAM

Initial ACS Data (2006-2016)
    33,851,424 observations
          |
          v
Filter: Hispanic-Mexican (HISPAN=1), Born in Mexico (BPL=200), Non-citizen (CITIZEN=3)
    701,347 observations
          |
          v
Restrict to ages 26-35 as of June 15, 2012
    181,229 observations
          |
          v
Apply DACA eligibility criteria:
  - Arrived in US before age 16
  - In US by 2007 (continuous presence)
    47,418 observations
          |
          v
Exclude 2012 (ambiguous year)
    43,238 observations (ANALYSIS SAMPLE)
          |
          |---> Treatment group (ages 26-30): 25,470 observations
          |
          |---> Control group (ages 31-35): 17,768 observations
"""

ax.text(0.5, 0.5, flow_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

plt.savefig('figure5_sample_flow.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_sample_flow.pdf', bbox_inches='tight')
print("  Saved figure5_sample_flow.png/pdf")
plt.close()

print("\nAll figures created successfully!")

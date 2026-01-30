"""
DACA Replication Study - Figure Generation
Creates figures for the replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

os.chdir(r"C:\Users\seraf\DACA Results Task 2\replication_06")

# Load parallel trends data
trends = pd.read_csv('parallel_trends_data.csv')

# Figure 1: Parallel Trends Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot control group
ctrl = trends[trends['Treatment'] == 0]
ax.plot(ctrl['Year'], ctrl['FullTime_Rate'], 'b-o', label='Control (Ages 31-35)',
        linewidth=2, markersize=8)

# Plot treatment group
treat = trends[trends['Treatment'] == 1]
ax.plot(treat['Year'], treat['FullTime_Rate'], 'r-s', label='Treatment (Ages 26-30)',
        linewidth=2, markersize=8)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')

# Shade pre and post periods
ax.axvspan(2006, 2012, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012, 2016, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status\n(Hispanic-Mexican, Mexican-Born Non-Citizens)', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.5, 0.75)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
print("Figure 1 saved: figure1_parallel_trends.png/pdf")

# Figure 2: Event Study Plot
# Year-specific treatment effects from the analysis
event_study_data = {
    'Year': [2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'Estimate': [0.00995, 0.01526, 0.02706, 0.02028, 0.03008, 0.04322, 0.04610, 0.03001, 0.03557],
    'SE': [0.01033, 0.00966, 0.01061, 0.00857, 0.01035, 0.01155, 0.01009, 0.00946, 0.00883]
}
event_df = pd.DataFrame(event_study_data)
event_df['CI_lower'] = event_df['Estimate'] - 1.96 * event_df['SE']
event_df['CI_upper'] = event_df['Estimate'] + 1.96 * event_df['SE']

# Add reference year 2006
ref_row = pd.DataFrame({'Year': [2006], 'Estimate': [0], 'SE': [0], 'CI_lower': [0], 'CI_upper': [0]})
event_df = pd.concat([ref_row, event_df], ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot estimates with confidence intervals
ax.errorbar(event_df['Year'], event_df['Estimate'],
            yerr=[event_df['Estimate'] - event_df['CI_lower'],
                  event_df['CI_upper'] - event_df['Estimate']],
            fmt='ko', capsize=5, capthick=2, linewidth=2, markersize=8)

# Connect points
ax.plot(event_df['Year'], event_df['Estimate'], 'k-', linewidth=1, alpha=0.5)

# Add reference lines
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

# Shade post-DACA period
ax.axvspan(2012, 2016.5, alpha=0.1, color='red')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects\n(Reference Year: 2006)', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
print("Figure 2 saved: figure2_event_study.png/pdf")

# Figure 3: DD Visualization
fig, ax = plt.subplots(figsize=(8, 6))

# Pre and Post means
pre_ctrl = ctrl[ctrl['Year'] <= 2011]['FullTime_Rate'].mean()
pre_treat = treat[treat['Year'] <= 2011]['FullTime_Rate'].mean()
post_ctrl = ctrl[ctrl['Year'] >= 2013]['FullTime_Rate'].mean()
post_treat = treat[treat['Year'] >= 2013]['FullTime_Rate'].mean()

# Plot lines
x_pre = [0, 1]
x_post = [2, 3]

# Control line
ax.plot([0.5, 2.5], [pre_ctrl, post_ctrl], 'b-o', label='Control (31-35)',
        linewidth=2, markersize=10)
# Treatment line
ax.plot([0.5, 2.5], [pre_treat, post_treat], 'r-s', label='Treatment (26-30)',
        linewidth=2, markersize=10)

# Counterfactual
counterfactual_post = pre_treat + (post_ctrl - pre_ctrl)
ax.plot([0.5, 2.5], [pre_treat, counterfactual_post], 'r--',
        label='Treatment Counterfactual', linewidth=2, alpha=0.5)

# DD arrow
ax.annotate('', xy=(2.7, post_treat), xytext=(2.7, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(2.85, (post_treat + counterfactual_post)/2, f'DD = {post_treat - counterfactual_post:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks([0.5, 2.5])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(-0.2, 3.5)
ax.set_ylim(0.55, 0.68)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_dd_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_dd_visual.pdf', bbox_inches='tight')
print("Figure 3 saved: figure3_dd_visual.png/pdf")

# Figure 4: Results by sex
fig, ax = plt.subplots(figsize=(8, 5))

models = ['Main\n(All)', 'Male\nOnly', 'Female\nOnly', 'DACA\nEligible']
estimates = [0.0236, 0.0319, 0.0018, 0.0484]
se = [0.00405, 0.00653, 0.00592, 0.01059]
ci_lower = [e - 1.96*s for e, s in zip(estimates, se)]
ci_upper = [e + 1.96*s for e, s in zip(estimates, se)]

colors = ['blue', 'green', 'purple', 'orange']
x_pos = range(len(models))

ax.bar(x_pos, estimates, yerr=[np.array(estimates)-np.array(ci_lower),
                                np.array(ci_upper)-np.array(estimates)],
       color=colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('DD Estimate\n(Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add significance stars
for i, (est, se_val) in enumerate(zip(estimates, se)):
    pval = 2 * (1 - 0.999) if abs(est/se_val) > 3 else 2 * (1 - 0.95) if abs(est/se_val) > 2 else 0.5
    if abs(est/se_val) > 2.58:
        ax.text(i, est + 1.96*se_val + 0.005, '***', ha='center', fontsize=12)
    elif abs(est/se_val) > 1.96:
        ax.text(i, est + 1.96*se_val + 0.005, '**', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', bbox_inches='tight')
print("Figure 4 saved: figure4_robustness.png/pdf")

print("\nAll figures generated successfully!")

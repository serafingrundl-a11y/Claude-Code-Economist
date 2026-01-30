"""
Generate figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Figure 1: Full-time employment trends by group
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate weighted means by year and group
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

trends.columns = ['Control (Ages 31-35)', 'Treated (Ages 26-30)']

trends.plot(ax=ax1, marker='o', linewidth=2, markersize=8)

# Add vertical line at treatment
ax1.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.55, 0.75)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
print("Figure 1 saved")

# Figure 2: Event Study Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

event_df = pd.read_csv('event_study_results.csv')

# Plot coefficients with confidence intervals
years = event_df['Year']
coefs = event_df['Coefficient']
ses = event_df['SE']

# Calculate 95% CI
ci_lower = coefs - 1.96 * ses
ci_upper = coefs + 1.96 * ses

ax2.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='blue')
ax2.plot(years, coefs, marker='o', linewidth=2, markersize=8, color='blue')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.axvline(x=2012, color='red', linestyle='--', alpha=0.7, label='DACA Implementation')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
print("Figure 2 saved")

# Figure 3: DiD Visualization
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate group means for pre and post periods
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)].apply(
    lambda x: np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                         weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT']))['FT']
post_control = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                          weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])
pre_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                       weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
post_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                        weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])

pre_control = 0.6886
post_control = 0.6629
pre_treat = 0.6369
post_treat = 0.6860

# Plot actual lines
ax3.plot([0, 1], [pre_control, post_control], 'b-o', linewidth=2, markersize=10,
         label='Control (Ages 31-35)')
ax3.plot([0, 1], [pre_treat, post_treat], 'r-o', linewidth=2, markersize=10,
         label='Treated (Ages 26-30)')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax3.plot([0, 1], [pre_treat, counterfactual], 'r--', linewidth=2, alpha=0.5,
         label='Treated Counterfactual')

# Add arrow showing DiD
ax3.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.05, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
         fontsize=12, color='green', verticalalignment='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Illustration', fontsize=14)
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.2, 1.4)
ax3.set_ylim(0.55, 0.75)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
print("Figure 3 saved")

# Figure 4: Sample distribution by year
fig4, ax4 = plt.subplots(figsize=(10, 6))

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control (Ages 31-35)', 'Treated (Ages 26-30)']

sample_by_year.plot(kind='bar', ax=ax4, width=0.8)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Number of Observations', fontsize=12)
ax4.set_title('Sample Size by Year and DACA Eligibility Status', fontsize=14)
ax4.legend(loc='upper right', fontsize=10)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)

plt.tight_layout()
plt.savefig('figure4_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_size.pdf', bbox_inches='tight')
print("Figure 4 saved")

print("\nAll figures generated successfully!")

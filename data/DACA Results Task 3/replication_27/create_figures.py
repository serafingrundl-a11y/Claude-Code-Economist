"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Figure 1: Full-time employment trends by group
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate yearly means by ELIGIBLE status
yearly_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

ax1.plot(yearly_means.index, yearly_means[0], 'b-o', label='Control (Age 31-35 in 2012)', linewidth=2, markersize=8)
ax1.plot(yearly_means.index, yearly_means[1], 'r-s', label='Treatment (Age 26-30 in 2012)', linewidth=2, markersize=8)
ax1.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (June 2012)')
ax1.axvspan(2012, 2016.5, alpha=0.1, color='green', label='Post-treatment period')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-time Employment Rate', fontsize=12)
ax1.set_title('Full-time Employment Trends by DACA Eligibility Status', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim([0.55, 0.75])
ax1.grid(True, alpha=0.3)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 1 saved: figure1_trends.png")

# Figure 2: Event study plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Event study coefficients (from analysis output)
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [-0.0611, -0.0413, -0.0673, 0.0, 0.0178, -0.0121, 0.0293, 0.0482]
ses = [0.0233, 0.0309, 0.0197, 0.0, 0.0269, 0.0216, 0.0355, 0.0215]

# Calculate 95% CI
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

ax2.plot(years, coefs, 'ko-', linewidth=2, markersize=8, label='Point Estimate')
ax2.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=2011.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Treatment Effect (Relative to 2011)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(years)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 2 saved: figure2_eventstudy.png")

# Figure 3: DiD visualization (2x2)
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Calculate means for 2x2 table
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

# Plot
x = [0, 1]
ax3.plot(x, [pre_control, post_control], 'b-o', linewidth=3, markersize=12, label='Control (Age 31-35)')
ax3.plot(x, [pre_treat, post_treat], 'r-s', linewidth=3, markersize=12, label='Treatment (Age 26-30)')

# Add counterfactual line
counterfactual = pre_treat + (post_control - pre_control)
ax3.plot([0, 1], [pre_treat, counterfactual], 'r--', linewidth=2, alpha=0.7, label='Counterfactual')

# Arrow showing DiD
ax3.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.1, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
         fontsize=12, color='green', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)'], fontsize=11)
ax3.set_ylabel('Full-time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.legend(loc='lower right', fontsize=10)
ax3.set_ylim([0.58, 0.72])
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 3 saved: figure3_did.png")

# Figure 4: Sample composition
fig4, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age distribution
ax4a = axes[0]
df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].hist(bins=20, alpha=0.7, label='Treatment (26-30)', ax=ax4a, color='red')
df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].hist(bins=20, alpha=0.7, label='Control (31-35)', ax=ax4a, color='blue')
ax4a.set_xlabel('Age in June 2012', fontsize=12)
ax4a.set_ylabel('Count', fontsize=12)
ax4a.set_title('Age Distribution by Treatment Status', fontsize=14)
ax4a.legend()

# Sample size by year and group
ax4b = axes[1]
counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
x = np.arange(len(counts.index))
width = 0.35
bars1 = ax4b.bar(x - width/2, counts[0], width, label='Control (31-35)', color='blue', alpha=0.7)
bars2 = ax4b.bar(x + width/2, counts[1], width, label='Treatment (26-30)', color='red', alpha=0.7)
ax4b.set_xlabel('Year', fontsize=12)
ax4b.set_ylabel('Sample Size', fontsize=12)
ax4b.set_title('Sample Size by Year and Treatment Status', fontsize=14)
ax4b.set_xticks(x)
ax4b.set_xticklabels(counts.index)
ax4b.legend()

plt.tight_layout()
plt.savefig('figure4_sample.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 4 saved: figure4_sample.png")

# Figure 5: Heterogeneity by gender
fig5, ax5 = plt.subplots(figsize=(8, 6))

# Calculate means by gender
male_data = df[df['SEX']==1].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
female_data = df[df['SEX']==2].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

ax5.plot(male_data.index, male_data[1], 'b-o', linewidth=2, markersize=6, label='Male Treatment')
ax5.plot(male_data.index, male_data[0], 'b--s', linewidth=2, markersize=6, label='Male Control')
ax5.plot(female_data.index, female_data[1], 'r-o', linewidth=2, markersize=6, label='Female Treatment')
ax5.plot(female_data.index, female_data[0], 'r--s', linewidth=2, markersize=6, label='Female Control')
ax5.axvline(x=2012, color='gray', linestyle=':', linewidth=2)

ax5.set_xlabel('Year', fontsize=12)
ax5.set_ylabel('Full-time Employment Rate', fontsize=12)
ax5.set_title('Full-time Employment Trends by Gender', fontsize=14)
ax5.legend(loc='lower right', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure5_gender.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figure 5 saved: figure5_gender.png")

print("\nAll figures created successfully!")

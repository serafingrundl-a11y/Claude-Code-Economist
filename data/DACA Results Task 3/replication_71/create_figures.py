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

# Create dummy variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['female'] = (df['SEX'] == 2).astype(int)
df['married'] = (df['MARST'].isin([1, 2])).astype(int)
df['has_children'] = (df['NCHILD'] > 0).astype(int)

# Figure 1: Full-time employment trends over time
fig1, ax1 = plt.subplots(figsize=(10, 6))
year_means = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
years = year_means.index.tolist()
control = year_means[0].values
treated = year_means[1].values

ax1.plot(years, control, 'o-', color='blue', linewidth=2, markersize=8, label='Control (Ages 31-35)')
ax1.plot(years, treated, 's--', color='red', linewidth=2, markersize=8, label='Treated (Ages 26-30)')
ax1.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Trends: Treatment vs. Control Groups', fontsize=14)
ax1.legend(loc='lower left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.55, 0.75)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=150, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
print("Figure 1 saved: figure1_trends.png/pdf")

# Figure 2: Event study plot
import statsmodels.formula.api as smf

# Create year dummies
for year in df['YEAR'].unique():
    df[f'year_{year}'] = (df['YEAR'] == year).astype(int)

# Create interactions
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'ELIG_{year}'] = df['ELIGIBLE'] * df[f'year_{year}']

event_model = smf.ols('FT ~ ELIGIBLE + year_2009 + year_2010 + year_2011 + year_2013 + year_2014 + year_2015 + year_2016 + ELIG_2009 + ELIG_2010 + ELIG_2011 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016',
                      data=df).fit(cov_type='HC1')

years_event = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [0]  # 2008 is reference
ses = [0]
for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    coefs.append(event_model.params[f'ELIG_{year}'])
    ses.append(event_model.bse[f'ELIG_{year}'])

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.errorbar(years_event, coefs, yerr=[1.96*s for s in ses], fmt='o-',
             color='navy', linewidth=2, markersize=8, capsize=5, capthick=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (relative to 2008)', fontsize=12)
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=150, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
print("Figure 2 saved: figure2_eventstudy.png/pdf")

# Figure 3: DiD visualization (2x2)
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Pre and post means
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()
pre_treated = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treated = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()

x_control = [0, 1]
x_treated = [0, 1]
y_control = [pre_control, post_control]
y_treated = [pre_treated, post_treated]

ax3.plot(x_control, y_control, 'o-', color='blue', linewidth=2, markersize=10, label='Control (Ages 31-35)')
ax3.plot(x_treated, y_treated, 's-', color='red', linewidth=2, markersize=10, label='Treated (Ages 26-30)')

# Counterfactual line for treated
y_counterfactual = [pre_treated, pre_treated + (post_control - pre_control)]
ax3.plot(x_treated, y_counterfactual, 's--', color='red', linewidth=1.5, markersize=8, alpha=0.5, label='Treated Counterfactual')

# Arrow showing treatment effect
ax3.annotate('', xy=(1.05, post_treated), xytext=(1.05, y_counterfactual[1]),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.1, (post_treated + y_counterfactual[1])/2, f'DiD = {post_treated - y_counterfactual[1]:.3f}',
         fontsize=11, color='green', va='center')

ax3.set_xlabel('Period', fontsize=12)
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences Visualization', fontsize=14)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)'])
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0.58, 0.72)
ax3.set_xlim(-0.2, 1.4)
plt.tight_layout()
plt.savefig('figure3_did.png', dpi=150, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
print("Figure 3 saved: figure3_did.png/pdf")

# Figure 4: Heterogeneity by sex
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Male trends
male_means = df[df['SEX']==1].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
female_means = df[df['SEX']==2].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

ax4.plot(male_means.index, male_means[0], 'o-', color='blue', linewidth=2, markersize=6, label='Control Males')
ax4.plot(male_means.index, male_means[1], 's-', color='navy', linewidth=2, markersize=6, label='Treated Males')
ax4.plot(female_means.index, female_means[0], 'o--', color='red', linewidth=2, markersize=6, label='Control Females')
ax4.plot(female_means.index, female_means[1], 's--', color='darkred', linewidth=2, markersize=6, label='Treated Females')
ax4.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Year', fontsize=12)
ax4.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax4.set_title('Full-Time Employment by Sex and Eligibility', fontsize=14)
ax4.legend(loc='lower left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure4_bysex.png', dpi=150, bbox_inches='tight')
plt.savefig('figure4_bysex.pdf', bbox_inches='tight')
print("Figure 4 saved: figure4_bysex.png/pdf")

# Figure 5: Sample composition by year
fig5, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Sample size by year and eligibility
sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
sample_by_year.columns = ['Control', 'Treated']
sample_by_year.plot(kind='bar', ax=axes[0], color=['blue', 'red'], alpha=0.7)
axes[0].set_xlabel('Year', fontsize=12)
axes[0].set_ylabel('Sample Size', fontsize=12)
axes[0].set_title('Sample Size by Year and Eligibility', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].tick_params(axis='x', rotation=45)

# Right: Education distribution
educ_counts = df['EDUC_RECODE'].value_counts()
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_counts = educ_counts.reindex(educ_order)
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
axes[1].barh(range(len(educ_counts)), educ_counts.values, color=colors)
axes[1].set_yticks(range(len(educ_counts)))
axes[1].set_yticklabels(educ_counts.index)
axes[1].set_xlabel('Count', fontsize=12)
axes[1].set_title('Education Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('figure5_descriptive.png', dpi=150, bbox_inches='tight')
plt.savefig('figure5_descriptive.pdf', bbox_inches='tight')
print("Figure 5 saved: figure5_descriptive.png/pdf")

print("\nAll figures created successfully!")

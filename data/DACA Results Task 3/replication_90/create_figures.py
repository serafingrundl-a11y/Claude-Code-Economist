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

# Create groups
df['Group'] = np.where(df['ELIGIBLE'] == 1, 'Eligible (26-30)', 'Comparison (31-35)')

# Figure 1: Full-time employment rates over time by group
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

fig1_data = df.groupby(['YEAR', 'Group']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()

plt.figure(figsize=(10, 6))
plt.plot(fig1_data.index, fig1_data['Eligible (26-30)'], 'b-o', linewidth=2, markersize=8, label='Eligible (26-30)')
plt.plot(fig1_data.index, fig1_data['Comparison (31-35)'], 'r--s', linewidth=2, markersize=8, label='Comparison (31-35)')
plt.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Full-Time Employment Rates by DACA Eligibility Group', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0.55, 0.75)
plt.xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_trends.png")

# Figure 2: DiD visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate means
pre_treat = weighted_mean(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 0)], 'FT', 'PERWT')
post_treat = weighted_mean(df[(df['ELIGIBLE'] == 1) & (df['AFTER'] == 1)], 'FT', 'PERWT')
pre_control = weighted_mean(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 0)], 'FT', 'PERWT')
post_control = weighted_mean(df[(df['ELIGIBLE'] == 0) & (df['AFTER'] == 1)], 'FT', 'PERWT')

# Plot actual trends
ax.plot([0, 1], [pre_treat, post_treat], 'b-o', linewidth=2, markersize=10, label='Eligible (Actual)')
ax.plot([0, 1], [pre_control, post_control], 'r-s', linewidth=2, markersize=10, label='Comparison (Actual)')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'b--', linewidth=2, alpha=0.5, label='Eligible (Counterfactual)')

# Add arrow showing treatment effect
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.05, (post_treat + counterfactual)/2, f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', verticalalignment='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-Treatment\n(2008-2011)', 'Post-Treatment\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.55, 0.75)
plt.tight_layout()
plt.savefig('figure2_did.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_did.png")

# Figure 3: Event study plot
import statsmodels.formula.api as smf

# Create year-specific treatment effects
for year in df['YEAR'].unique():
    df[f'ELIGIBLE_Y{year}'] = (df['ELIGIBLE'] * (df['YEAR'] == year)).astype(int)

df['YEAR_str'] = df['YEAR'].astype(str)
df['STATE_str'] = df['STATEFIP'].astype(str)

year_vars = [f'ELIGIBLE_Y{y}' for y in sorted(df['YEAR'].unique()) if y != 2011]
year_formula = ' + '.join(year_vars)

model_event = smf.ols(f'FT ~ ELIGIBLE + {year_formula} + C(YEAR_str) + C(STATE_str)',
                      data=df).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

years = [2008, 2009, 2010, 2013, 2014, 2015, 2016]
coefs = [model_event.params[f'ELIGIBLE_Y{y}'] for y in years]
ses = [model_event.bse[f'ELIGIBLE_Y{y}'] for y in years]
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

# Insert reference year
years_plot = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_plot = coefs[:3] + [0] + coefs[3:]
ci_lower_plot = ci_lower[:3] + [0] + ci_lower[3:]
ci_upper_plot = ci_upper[:3] + [0] + ci_upper[3:]

plt.figure(figsize=(10, 6))
plt.errorbar(years_plot, coefs_plot, yerr=[np.array(coefs_plot)-np.array(ci_lower_plot),
                                            np.array(ci_upper_plot)-np.array(coefs_plot)],
             fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8, color='navy')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Treatment Effect (relative to 2011)', fontsize=12)
plt.title('Event Study: Year-Specific Treatment Effects', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(years_plot)
plt.tight_layout()
plt.savefig('figure3_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_eventstudy.png")

# Figure 4: Heterogeneity by gender
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Male
male_data = df[df['SEX'] == 1].groupby(['YEAR', 'Group']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()

axes[0].plot(male_data.index, male_data['Eligible (26-30)'], 'b-o', linewidth=2, markersize=6, label='Eligible')
axes[0].plot(male_data.index, male_data['Comparison (31-35)'], 'r--s', linewidth=2, markersize=6, label='Comparison')
axes[0].axvline(x=2012, color='gray', linestyle=':', linewidth=2)
axes[0].set_title('Males', fontsize=12)
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Full-Time Employment Rate', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.55, 0.85)

# Female
female_data = df[df['SEX'] == 2].groupby(['YEAR', 'Group']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()

axes[1].plot(female_data.index, female_data['Eligible (26-30)'], 'b-o', linewidth=2, markersize=6, label='Eligible')
axes[1].plot(female_data.index, female_data['Comparison (31-35)'], 'r--s', linewidth=2, markersize=6, label='Comparison')
axes[1].axvline(x=2012, color='gray', linestyle=':', linewidth=2)
axes[1].set_title('Females', fontsize=12)
axes[1].set_xlabel('Year', fontsize=11)
axes[1].set_ylabel('Full-Time Employment Rate', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0.45, 0.65)

plt.suptitle('Full-Time Employment by Gender and DACA Eligibility', fontsize=14)
plt.tight_layout()
plt.savefig('figure4_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved: figure4_gender.png")

# Figure 5: Sample distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
age_counts = df['AGE_IN_JUNE_2012'].value_counts().sort_index()
axes[0, 0].bar(age_counts.index, age_counts.values, width=0.2, color='steelblue', alpha=0.7)
axes[0, 0].axvline(x=30.5, color='red', linestyle='--', linewidth=2, label='Eligibility cutoff')
axes[0, 0].set_xlabel('Age in June 2012', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].set_title('Distribution of Age at DACA Implementation', fontsize=12)
axes[0, 0].legend()

# Year distribution
year_counts = df['YEAR'].value_counts().sort_index()
colors = ['steelblue' if y < 2012 else 'coral' for y in year_counts.index]
axes[0, 1].bar(year_counts.index, year_counts.values, color=colors, alpha=0.7)
axes[0, 1].set_xlabel('Year', fontsize=11)
axes[0, 1].set_ylabel('Count', fontsize=11)
axes[0, 1].set_title('Sample Size by Year', fontsize=12)
axes[0, 1].legend(handles=[plt.Rectangle((0,0),1,1,color='steelblue',alpha=0.7,label='Pre-DACA'),
                           plt.Rectangle((0,0),1,1,color='coral',alpha=0.7,label='Post-DACA')])

# Education distribution
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_counts = df['EDUC_RECODE'].value_counts()
educ_counts = educ_counts.reindex(educ_order)
axes[1, 0].barh(range(len(educ_counts)), educ_counts.values, color='steelblue', alpha=0.7)
axes[1, 0].set_yticks(range(len(educ_counts)))
axes[1, 0].set_yticklabels(educ_counts.index, fontsize=10)
axes[1, 0].set_xlabel('Count', fontsize=11)
axes[1, 0].set_title('Education Distribution', fontsize=12)

# State distribution (top 10)
state_counts = df['statename'].value_counts().head(10)
axes[1, 1].barh(range(len(state_counts)), state_counts.values[::-1], color='steelblue', alpha=0.7)
axes[1, 1].set_yticks(range(len(state_counts)))
axes[1, 1].set_yticklabels(state_counts.index[::-1], fontsize=10)
axes[1, 1].set_xlabel('Count', fontsize=11)
axes[1, 1].set_title('Top 10 States by Sample Size', fontsize=12)

plt.tight_layout()
plt.savefig('figure5_sample.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved: figure5_sample.png")

# Figure 6: Coefficient plot
models = ['Basic DiD', 'Year FE', 'State+Year FE', 'Demographics', 'State Policies']
estimates = [0.0643, 0.0629, 0.0626, 0.0520, 0.0511]
ses = [0.0153, 0.0139, 0.0144, 0.0151, 0.0149]
ci_lower = [e - 1.96*s for e, s in zip(estimates, ses)]
ci_upper = [e + 1.96*s for e, s in zip(estimates, ses)]

plt.figure(figsize=(10, 6))
y_pos = range(len(models))
plt.errorbar(estimates, y_pos, xerr=[np.array(estimates)-np.array(ci_lower),
                                      np.array(ci_upper)-np.array(estimates)],
             fmt='o', capsize=5, capthick=2, markersize=10, color='navy', linewidth=2)
plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)
plt.yticks(y_pos, models, fontsize=11)
plt.xlabel('DiD Effect Estimate (with 95% CI)', fontsize=12)
plt.title('DACA Effect on Full-Time Employment: Model Comparison', fontsize=14)
plt.grid(True, alpha=0.3, axis='x')
plt.xlim(-0.02, 0.12)
plt.tight_layout()
plt.savefig('figure6_coefplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved: figure6_coefplot.png")

print("\nAll figures created successfully!")

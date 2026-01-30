"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load data
data = pd.read_csv('data/prepared_data_numeric_version.csv')
data_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

# Figure 1: Parallel Trends
print("Creating Figure 1: Parallel Trends...")
trends = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))
years = trends.index.tolist()
ax.plot(years, trends[0], 'b-o', label='Control (Ages 31-35 in June 2012)', linewidth=2, markersize=8)
ax.plot(years, trends[1], 'r-s', label='Treatment (Ages 26-30 in June 2012)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png")

# Figure 2: Event Study
print("Creating Figure 2: Event Study...")
event_df = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))
years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ses = event_df['SE'].values

# Calculate 95% CIs
ci_low = coefs - 1.96 * ses
ci_high = coefs + 1.96 * ses

ax.errorbar(years, coefs, yerr=1.96*ses, fmt='ko', capsize=5, markersize=8, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png")

# Figure 3: Distribution of Key Variables
print("Creating Figure 3: Variable Distributions...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution by group
ax1 = axes[0, 0]
data[data['ELIGIBLE']==1]['AGE'].hist(ax=ax1, bins=20, alpha=0.5, label='Treatment', color='red')
data[data['ELIGIBLE']==0]['AGE'].hist(ax=ax1, bins=20, alpha=0.5, label='Control', color='blue')
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')
ax1.set_title('Age Distribution by Group')
ax1.legend()

# Employment status
ax2 = axes[0, 1]
emp_data = data_labels.groupby('EMPSTAT').size()
emp_data.plot(kind='bar', ax=ax2, color=['green', 'gray', 'orange'])
ax2.set_xlabel('Employment Status')
ax2.set_ylabel('Frequency')
ax2.set_title('Employment Status Distribution')
ax2.tick_params(axis='x', rotation=45)

# Full-time rate by year and group
ax3 = axes[1, 0]
ft_by_year = data.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
ft_by_year.plot(kind='bar', ax=ax3, color=['blue', 'red'], alpha=0.7)
ax3.set_xlabel('Year')
ax3.set_ylabel('Full-Time Employment Rate')
ax3.set_title('FT Rate by Year and Eligibility')
ax3.legend(['Control', 'Treatment'])
ax3.tick_params(axis='x', rotation=45)

# Sex distribution by group
ax4 = axes[1, 1]
sex_by_group = data.groupby(['ELIGIBLE', 'SEX']).size().unstack()
sex_by_group.plot(kind='bar', ax=ax4, color=['blue', 'pink'])
ax4.set_xlabel('ELIGIBLE (0=Control, 1=Treatment)')
ax4.set_ylabel('Frequency')
ax4.set_title('Sex Distribution by Group')
ax4.legend(['Male', 'Female'])
ax4.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('figure3_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_distributions.png")

# Figure 4: DiD Coefficient Comparison
print("Creating Figure 4: DiD Coefficients Comparison...")
results = pd.read_csv('results_summary.csv')

fig, ax = plt.subplots(figsize=(10, 6))
models = results['Model'].values
coefs = results['Coefficient'].values
ses = results['SE'].values

x_pos = np.arange(len(models))
ax.bar(x_pos, coefs, yerr=1.96*ses, capsize=5, color=['steelblue', 'forestgreen', 'coral', 'purple', 'goldenrod'], alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect Estimates Across Specifications', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

# Add coefficient values on bars
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.annotate(f'{c:.3f}', xy=(i, c + 1.96*s + 0.01), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure4_coefficients.png")

# Figure 5: Subgroup Analysis
print("Creating Figure 5: Subgroup Analysis...")

# Re-run subgroup analyses to get data
data['ELIGIBLE_AFTER'] = data['ELIGIBLE'] * data['AFTER']
data['FEMALE'] = (data['SEX'] == 2).astype(int)
data['ED_LTHS'] = (data['EDUC'] < 6).astype(int)

import statsmodels.formula.api as smf

# By sex
model_male = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data[data['FEMALE']==0]).fit()
model_female = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data[data['FEMALE']==1]).fit()

# By marital status
data['MARRIED'] = (data['MARST'].isin([1, 2])).astype(int)
model_married = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data[data['MARRIED']==1]).fit()
model_unmarried = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=data[data['MARRIED']==0]).fit()

subgroups = ['Male', 'Female', 'Married', 'Unmarried']
subgroup_coefs = [
    model_male.params['ELIGIBLE_AFTER'],
    model_female.params['ELIGIBLE_AFTER'],
    model_married.params['ELIGIBLE_AFTER'],
    model_unmarried.params['ELIGIBLE_AFTER']
]
subgroup_ses = [
    model_male.bse['ELIGIBLE_AFTER'],
    model_female.bse['ELIGIBLE_AFTER'],
    model_married.bse['ELIGIBLE_AFTER'],
    model_unmarried.bse['ELIGIBLE_AFTER']
]

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(subgroups))
colors = ['steelblue', 'coral', 'forestgreen', 'purple']
ax.barh(x_pos, subgroup_coefs, xerr=[1.96*s for s in subgroup_ses], capsize=5, color=colors, alpha=0.7)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(x_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect by Subgroup', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add coefficient values
for i, (c, s) in enumerate(zip(subgroup_coefs, subgroup_ses)):
    ax.annotate(f'{c:.3f}', xy=(c + 1.96*s + 0.01, i), ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_subgroups.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure5_subgroups.png")

print("\nAll figures created successfully!")

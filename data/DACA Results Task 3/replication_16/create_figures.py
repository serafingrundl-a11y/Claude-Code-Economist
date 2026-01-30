"""
Create figures for the DACA replication study
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# =============================
# Figure 1: Full-Time Employment Rates by Year and Group
# =============================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate yearly FT rates by group
yearly_rates = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_rates.columns = ['Control (Age 31-35)', 'Treatment (Age 26-30)']

# Plot
years = yearly_rates.index.tolist()
ax1.plot(years, yearly_rates['Treatment (Age 26-30)'], 'o-', color='blue', linewidth=2, markersize=8, label='Treatment (Age 26-30, DACA-eligible)')
ax1.plot(years, yearly_rates['Control (Age 31-35)'], 's--', color='red', linewidth=2, markersize=8, label='Control (Age 31-35)')

# Add vertical line at DACA implementation
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation (June 2012)')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax1.set_title('Full-Time Employment Rates by Year and Treatment Group', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_ylim(0.55, 0.75)
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_ft_rates_by_year.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_ft_rates_by_year.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: Full-time employment rates by year")

# =============================
# Figure 2: Event Study Plot
# =============================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Load event study results
event_df = pd.read_csv('event_study_results.csv')

# Add 2008 as reference year with coefficient 0
event_df = pd.concat([
    pd.DataFrame({'Year': [2008], 'Coefficient': [0], 'SE': [0], 'CI_Lower': [0], 'CI_Upper': [0], 'P_Value': [np.nan]}),
    event_df
]).reset_index(drop=True)

# Sort by year
event_df = event_df.sort_values('Year')

# Calculate SE for error bars
years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_lower = event_df['CI_Lower'].values
ci_upper = event_df['CI_Upper'].values

# Plot coefficients with error bars
ax2.errorbar(years, coefs,
             yerr=[coefs - ci_lower, ci_upper - coefs],
             fmt='o', color='blue', markersize=10, capsize=5, capthick=2, linewidth=2)

# Connect points with line
ax2.plot(years, coefs, 'b-', alpha=0.5, linewidth=1.5)

# Add horizontal line at 0
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation (between 2011 and 2013)
ax2.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax2.annotate('DACA\nImplementation', xy=(2012, ax2.get_ylim()[1]*0.9),
             ha='center', fontsize=10, color='gray')

# Shade pre-treatment period
ax2.axvspan(2007.5, 2012, alpha=0.1, color='gray')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Treatment Effect (Relative to 2008)', fontsize=12)
ax2.set_title('Event Study: Year-Specific Treatment Effects on Full-Time Employment', fontsize=14)
ax2.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: Event study plot")

# =============================
# Figure 3: DiD Visualization (2x2 Plot)
# =============================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate means for 2x2
treat_before = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
treat_after = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
control_before = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
control_after = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

# X positions
x_before = 0.3
x_after = 0.7

# Plot treatment group
ax3.plot([x_before, x_after], [treat_before, treat_after], 'o-', color='blue',
         linewidth=3, markersize=15, label='Treatment (Age 26-30, DACA-eligible)')

# Plot control group
ax3.plot([x_before, x_after], [control_before, control_after], 's-', color='red',
         linewidth=3, markersize=15, label='Control (Age 31-35)')

# Plot counterfactual (dashed line)
counterfactual = treat_before + (control_after - control_before)
ax3.plot([x_before, x_after], [treat_before, counterfactual], 'b--',
         linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# Annotate DiD
did = treat_after - counterfactual
ax3.annotate('', xy=(x_after + 0.05, treat_after), xytext=(x_after + 0.05, counterfactual),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(x_after + 0.08, (treat_after + counterfactual)/2, f'DiD = {did:.3f}',
         fontsize=12, color='green', fontweight='bold')

# Annotate values
ax3.text(x_before - 0.08, treat_before, f'{treat_before:.3f}', fontsize=10, va='center', color='blue')
ax3.text(x_before - 0.08, control_before, f'{control_before:.3f}', fontsize=10, va='center', color='red')
ax3.text(x_after + 0.02, treat_after + 0.01, f'{treat_after:.3f}', fontsize=10, va='bottom', color='blue')
ax3.text(x_after + 0.02, control_after - 0.01, f'{control_after:.3f}', fontsize=10, va='top', color='red')

ax3.set_xlim(0, 1)
ax3.set_ylim(0.55, 0.75)
ax3.set_xticks([x_before, x_after])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=12)
ax3.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax3.set_title('Difference-in-Differences: DACA Impact on Full-Time Employment', fontsize=14)
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: DiD visualization")

# =============================
# Figure 4: Regression Coefficients Forest Plot
# =============================
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Load results
results_df = pd.read_csv('regression_results.csv')

# Reverse order for plotting (bottom to top)
results_df = results_df.iloc[::-1].reset_index(drop=True)

y_pos = np.arange(len(results_df))

# Plot coefficients with error bars
ax4.errorbar(results_df['DiD_Estimate'], y_pos,
             xerr=[results_df['DiD_Estimate'] - results_df['CI_Lower'],
                   results_df['CI_Upper'] - results_df['DiD_Estimate']],
             fmt='o', color='blue', markersize=10, capsize=5, capthick=2, elinewidth=2)

# Highlight preferred model
preferred_idx = results_df[results_df['Model'].str.contains('PREFERRED')].index[0]
ax4.scatter([results_df.loc[preferred_idx, 'DiD_Estimate']], [preferred_idx],
            s=200, color='green', zorder=5, marker='*', label='Preferred Model')

# Add vertical line at 0
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax4.set_yticks(y_pos)
ax4.set_yticklabels(results_df['Model'], fontsize=10)
ax4.set_xlabel('DiD Estimate (Effect on Full-Time Employment)', fontsize=12)
ax4.set_title('Comparison of DiD Estimates Across Model Specifications', fontsize=14)
ax4.legend(loc='lower right', fontsize=10)
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figure4_forest_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_forest_plot.pdf', bbox_inches='tight')
plt.close()

print("Figure 4 saved: Forest plot of regression coefficients")

# =============================
# Figure 5: Subgroup Analysis by Sex
# =============================
fig5, ax5 = plt.subplots(figsize=(10, 6))

# Calculate subgroup means by year
male_rates = df[df['SEX']==1].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
female_rates = df[df['SEX']==2].groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

years = male_rates.index.tolist()

# Plot males
ax5.plot(years, male_rates[1], 'o-', color='blue', linewidth=2, markersize=8, label='Male Treatment')
ax5.plot(years, male_rates[0], 's--', color='blue', linewidth=2, markersize=8, alpha=0.5, label='Male Control')

# Plot females
ax5.plot(years, female_rates[1], 'o-', color='red', linewidth=2, markersize=8, label='Female Treatment')
ax5.plot(years, female_rates[0], 's--', color='red', linewidth=2, markersize=8, alpha=0.5, label='Female Control')

# Add vertical line at DACA implementation
ax5.axvline(x=2012, color='gray', linestyle=':', linewidth=2)

ax5.set_xlabel('Year', fontsize=12)
ax5.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax5.set_title('Full-Time Employment Rates by Year, Group, and Sex', fontsize=14)
ax5.legend(loc='lower right', fontsize=9)
ax5.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure5_subgroup_sex.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_subgroup_sex.pdf', bbox_inches='tight')
plt.close()

print("Figure 5 saved: Subgroup analysis by sex")

# =============================
# Figure 6: Distribution of Key Variables
# =============================
fig6, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
ax = axes[0, 0]
df[df['ELIGIBLE']==1]['AGE'].hist(ax=ax, bins=20, alpha=0.5, label='Treatment', color='blue')
df[df['ELIGIBLE']==0]['AGE'].hist(ax=ax, bins=20, alpha=0.5, label='Control', color='red')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.set_title('Age Distribution by Treatment Group')
ax.legend()

# Years in USA distribution
ax = axes[0, 1]
df[df['ELIGIBLE']==1]['YRSUSA1'].hist(ax=ax, bins=20, alpha=0.5, label='Treatment', color='blue')
df[df['ELIGIBLE']==0]['YRSUSA1'].hist(ax=ax, bins=20, alpha=0.5, label='Control', color='red')
ax.set_xlabel('Years in USA')
ax.set_ylabel('Frequency')
ax.set_title('Years in USA Distribution by Treatment Group')
ax.legend()

# Education distribution (bar chart)
ax = axes[1, 0]
educ_dist = df.groupby(['ELIGIBLE', 'EDUC_RECODE']).size().unstack(fill_value=0)
educ_dist = educ_dist.div(educ_dist.sum(axis=1), axis=0)  # Convert to proportions
educ_dist.index = ['Control', 'Treatment']
educ_dist.plot(kind='bar', ax=ax, width=0.8)
ax.set_xlabel('Treatment Group')
ax.set_ylabel('Proportion')
ax.set_title('Education Distribution by Treatment Group')
ax.legend(title='Education', loc='upper right', fontsize=8)
ax.set_xticklabels(['Control', 'Treatment'], rotation=0)

# Number of children distribution
ax = axes[1, 1]
df[df['ELIGIBLE']==1]['NCHILD'].value_counts().sort_index().plot(kind='bar', ax=ax, alpha=0.5, label='Treatment', color='blue', position=0, width=0.4)
df[df['ELIGIBLE']==0]['NCHILD'].value_counts().sort_index().plot(kind='bar', ax=ax, alpha=0.5, label='Control', color='red', position=1, width=0.4)
ax.set_xlabel('Number of Children')
ax.set_ylabel('Frequency')
ax.set_title('Number of Children Distribution by Treatment Group')
ax.legend()

plt.tight_layout()
plt.savefig('figure6_distributions.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_distributions.pdf', bbox_inches='tight')
plt.close()

print("Figure 6 saved: Variable distributions")

print("\nAll figures created successfully!")

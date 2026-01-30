"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)

# Figure 1: Time series of full-time employment by group
print("Creating Figure 1: Time series...")
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: pd.Series({
        'FT_mean': np.average(x['FT'], weights=x['PERWT']),
        'FT_se': np.sqrt(np.average((x['FT'] - np.average(x['FT'], weights=x['PERWT']))**2, weights=x['PERWT']) / len(x))
    })
).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment group
treat = yearly_means[yearly_means['ELIGIBLE'] == 1]
control = yearly_means[yearly_means['ELIGIBLE'] == 0]

ax.plot(treat['YEAR'], treat['FT_mean'], 'o-', label='Treatment (Ages 26-30)', color='#2171b5', linewidth=2, markersize=8)
ax.plot(control['YEAR'], control['FT_mean'], 's-', label='Control (Ages 31-35)', color='#cb181d', linewidth=2, markersize=8)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates Before and After DACA')
ax.legend(loc='lower right')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figures/fig1_time_series.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig1_time_series.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig1_time_series.png")

# Figure 2: Event Study Plot
print("Creating Figure 2: Event study...")
event_df = pd.read_csv('event_study_results.csv')

# Add reference year (2011) with zero effect
ref_year = pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'SE': [0], 'CI_Low': [0], 'CI_High': [0], 'p_value': [1]})
event_df = pd.concat([event_df, ref_year]).sort_values('Year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(event_df['Year'], event_df['Coefficient'],
            yerr=[event_df['Coefficient'] - event_df['CI_Low'], event_df['CI_High'] - event_df['Coefficient']],
            fmt='o', color='#2171b5', capsize=5, capthick=2, linewidth=2, markersize=8)

# Connect points
ax.plot(event_df['Year'], event_df['Coefficient'], '-', color='#2171b5', linewidth=1, alpha=0.5)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at treatment
ax.axvline(x=2011.5, color='gray', linestyle='--', linewidth=2)

# Shade post-period
ax.axvspan(2011.5, 2016.5, alpha=0.1, color='green')

# Add annotation
ax.annotate('DACA\nImplementation', xy=(2012, 0.08), fontsize=10, ha='center')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xlim(2007.5, 2016.5)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figures/fig2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig2_event_study.png")

# Figure 3: DiD visualization (2x2 plot)
print("Creating Figure 3: DiD visualization...")
summary = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).reset_index()
summary.columns = ['ELIGIBLE', 'AFTER', 'FT_mean']

fig, ax = plt.subplots(figsize=(10, 6))

# Get values
pre_treat = summary[(summary['ELIGIBLE']==1) & (summary['AFTER']==0)]['FT_mean'].values[0]
post_treat = summary[(summary['ELIGIBLE']==1) & (summary['AFTER']==1)]['FT_mean'].values[0]
pre_control = summary[(summary['ELIGIBLE']==0) & (summary['AFTER']==0)]['FT_mean'].values[0]
post_control = summary[(summary['ELIGIBLE']==0) & (summary['AFTER']==1)]['FT_mean'].values[0]

# Plot lines
x_positions = [0, 1]
ax.plot(x_positions, [pre_treat, post_treat], 'o-', label='Treatment (Ages 26-30)',
        color='#2171b5', linewidth=3, markersize=12)
ax.plot(x_positions, [pre_control, post_control], 's-', label='Control (Ages 31-35)',
        color='#cb181d', linewidth=3, markersize=12)

# Add counterfactual line
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], '--', color='#2171b5', linewidth=2, alpha=0.5, label='Counterfactual')

# Add arrows showing DiD
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.12, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
        fontsize=12, va='center', color='green', fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences Visualization')
ax.legend(loc='lower left')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.60, 0.72)

plt.tight_layout()
plt.savefig('figures/fig3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig3_did_visual.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig3_did_visual.png")

# Figure 4: Sample composition by group
print("Creating Figure 4: Sample composition...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Education distribution
educ_data = df.groupby(['ELIGIBLE', 'EDUC_RECODE']).size().unstack(fill_value=0)
educ_data_pct = educ_data.div(educ_data.sum(axis=1), axis=0) * 100

educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_data_pct = educ_data_pct[educ_order]

x = np.arange(len(educ_order))
width = 0.35

bars1 = axes[0].bar(x - width/2, educ_data_pct.loc[1], width, label='Treatment (Ages 26-30)', color='#2171b5')
bars2 = axes[0].bar(x + width/2, educ_data_pct.loc[0], width, label='Control (Ages 31-35)', color='#cb181d')

axes[0].set_ylabel('Percentage')
axes[0].set_title('Education Distribution')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['< HS', 'HS', 'Some\nCollege', '2-Year', 'BA+'])
axes[0].legend()

# Sex distribution
sex_data = df.groupby(['ELIGIBLE', 'SEX']).size().unstack(fill_value=0)
sex_data_pct = sex_data.div(sex_data.sum(axis=1), axis=0) * 100

x = np.arange(2)
bars1 = axes[1].bar(x - width/2, sex_data_pct.loc[1], width, label='Treatment (Ages 26-30)', color='#2171b5')
bars2 = axes[1].bar(x + width/2, sex_data_pct.loc[0], width, label='Control (Ages 31-35)', color='#cb181d')

axes[1].set_ylabel('Percentage')
axes[1].set_title('Sex Distribution')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['Male', 'Female'])
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/fig4_sample_composition.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig4_sample_composition.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig4_sample_composition.png")

# Figure 5: Coefficient comparison across models
print("Creating Figure 5: Coefficient comparison...")
results = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results))
ax.barh(y_pos, results['Coefficient'], xerr=[results['Coefficient'] - results['CI_Low'],
        results['CI_High'] - results['Coefficient']], align='center', color='#2171b5',
        capsize=5, alpha=0.8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(results['Model'])
ax.set_xlabel('DiD Coefficient (Change in FT Employment Rate)')
ax.set_title('DACA Effect Estimates Across Model Specifications')

# Add coefficient values
for i, (coef, ci_l, ci_h) in enumerate(zip(results['Coefficient'], results['CI_Low'], results['CI_High'])):
    ax.text(coef + 0.012, i, f'{coef:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig5_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig5_coefficients.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig5_coefficients.png")

# Figure 6: Subgroup analysis
print("Creating Figure 6: Subgroup analysis...")
# Compute subgroup estimates
subgroup_results = []
for sex_val, sex_name in [(1, 'Male'), (2, 'Female')]:
    df_sub = df[df['SEX'] == sex_val]
    import statsmodels.formula.api as smf
    df_sub['ELIGIBLE_AFTER'] = df_sub['ELIGIBLE'] * df_sub['AFTER']
    model_sub = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df_sub, weights=df_sub['PERWT']).fit()
    subgroup_results.append({
        'Group': sex_name,
        'Coefficient': model_sub.params['ELIGIBLE_AFTER'],
        'CI_Low': model_sub.conf_int().loc['ELIGIBLE_AFTER', 0],
        'CI_High': model_sub.conf_int().loc['ELIGIBLE_AFTER', 1],
        'N': len(df_sub)
    })

# Add overall
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
model_all = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
subgroup_results.append({
    'Group': 'Overall',
    'Coefficient': model_all.params['ELIGIBLE_AFTER'],
    'CI_Low': model_all.conf_int().loc['ELIGIBLE_AFTER', 0],
    'CI_High': model_all.conf_int().loc['ELIGIBLE_AFTER', 1],
    'N': len(df)
})

subgroup_df = pd.DataFrame(subgroup_results)

fig, ax = plt.subplots(figsize=(8, 5))

y_pos = np.arange(len(subgroup_df))
colors = ['#2171b5', '#cb181d', '#238b45']

ax.barh(y_pos, subgroup_df['Coefficient'], xerr=[subgroup_df['Coefficient'] - subgroup_df['CI_Low'],
        subgroup_df['CI_High'] - subgroup_df['Coefficient']], align='center', color=colors,
        capsize=5, alpha=0.8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(subgroup_df['Group'])
ax.set_xlabel('DiD Coefficient (Change in FT Employment Rate)')
ax.set_title('DACA Effect by Sex')

# Add coefficient values
for i, (coef, ci_l, ci_h) in enumerate(zip(subgroup_df['Coefficient'], subgroup_df['CI_Low'], subgroup_df['CI_High'])):
    ax.text(ci_h + 0.005, i, f'{coef:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figures/fig6_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/fig6_subgroups.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figures/fig6_subgroups.png")

print("\nAll figures created successfully!")

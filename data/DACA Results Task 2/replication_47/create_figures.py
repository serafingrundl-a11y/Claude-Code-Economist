"""
DACA Replication Analysis - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Read results
summary_df = pd.read_csv('summary_statistics.csv')
main_results = pd.read_csv('main_results.csv')
event_df = pd.read_csv('event_study_results.csv')
het_df = pd.read_csv('heterogeneity_results.csv')

# Figure 1: Trends in Full-time Employment by Group
print("Creating Figure 1: Trends...")

# Read the raw data to compute yearly means
dtype_dict = {
    'YEAR': 'int32', 'BIRTHYR': 'int32', 'HISPAN': 'int16',
    'BPL': 'int16', 'CITIZEN': 'int16', 'YRIMMIG': 'int32',
    'UHRSWORK': 'int16', 'PERWT': 'float64', 'EMPSTAT': 'int16'
}

df = pd.read_csv('data/data.csv', usecols=list(dtype_dict.keys()), dtype=dtype_dict)

# Filter to sample
treatment_birthyears = [1982, 1983, 1984, 1985, 1986]
control_birthyears = [1977, 1978, 1979, 1980, 1981]

df_sample = df[(df['HISPAN'] == 1) &
               (df['BPL'] == 200) &
               (df['CITIZEN'] == 3) &
               (df['BIRTHYR'].isin(treatment_birthyears + control_birthyears)) &
               (df['YRIMMIG'] > 0)].copy()

df_sample = df_sample[(df_sample['YRIMMIG'] - df_sample['BIRTHYR']) < 16]
df_sample = df_sample[df_sample['YRIMMIG'] <= 2007]
df_sample['treated'] = df_sample['BIRTHYR'].isin(treatment_birthyears).astype(int)
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Calculate weighted means by year and group
yearly_means = df_sample.groupby(['YEAR', 'treated']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).reset_index(name='fulltime_rate')

yearly_means_treat = yearly_means[yearly_means['treated'] == 1].set_index('YEAR')['fulltime_rate']
yearly_means_control = yearly_means[yearly_means['treated'] == 0].set_index('YEAR')['fulltime_rate']

years = sorted(df_sample['YEAR'].unique())

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years, [yearly_means_treat.get(y, np.nan) for y in years], 'b-o', linewidth=2, markersize=8, label='Treatment (Ages 26-30 in 2012)')
ax.plot(years, [yearly_means_control.get(y, np.nan) for y in years], 'r-s', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Trends in Full-time Employment by Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.45, 0.75)
ax.grid(True, alpha=0.3)
ax.set_xticks(years)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# Figure 2: Event Study Plot
print("Creating Figure 2: Event Study...")

fig, ax = plt.subplots(figsize=(10, 6))

# Add reference year (2011) with 0 coefficient
event_plot = event_df.copy()
event_plot = pd.concat([event_plot, pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'SE': [0], 'CI_Lower': [0], 'CI_Upper': [0]})], ignore_index=True)
event_plot = event_plot.sort_values('Year')

years_plot = event_plot['Year'].values
coefs = event_plot['Coefficient'].values
ci_lower = event_plot['CI_Lower'].values
ci_upper = event_plot['CI_Upper'].values

ax.errorbar(years_plot, coefs, yerr=[coefs - ci_lower, ci_upper - coefs],
            fmt='o', capsize=5, capthick=2, color='navy', markersize=8, linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='red', linestyle='--', linewidth=2, label='DACA (Reference: 2011)')
ax.fill_between([2006, 2011.5], [-0.15, -0.15], [0.15, 0.15], alpha=0.1, color='gray')
ax.fill_between([2011.5, 2016], [-0.15, -0.15], [0.15, 0.15], alpha=0.1, color='blue')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (vs. 2011)', fontsize=12)
ax.set_title('Event Study: Treatment Effects by Year', fontsize=14)
ax.set_xticks(years_plot)
ax.set_ylim(-0.1, 0.15)
ax.grid(True, alpha=0.3)
ax.text(2008.5, 0.12, 'Pre-DACA', fontsize=11, ha='center')
ax.text(2014, 0.12, 'Post-DACA', fontsize=11, ha='center')
plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# Figure 3: Heterogeneity by Gender
print("Creating Figure 3: Heterogeneity...")

fig, ax = plt.subplots(figsize=(8, 5))

groups = het_df['Group'].values
coefs = het_df['Coefficient'].values
ci_lower = het_df['CI_Lower'].values
ci_upper = het_df['CI_Upper'].values

x = np.arange(len(groups))
ax.bar(x, coefs, width=0.6, color=['steelblue', 'coral'], alpha=0.8)
ax.errorbar(x, coefs, yerr=[coefs - ci_lower, ci_upper - coefs],
            fmt='none', capsize=8, capthick=2, color='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=12)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('Heterogeneity in Treatment Effects by Gender', fontsize=14)
ax.set_ylim(-0.05, 0.15)
ax.grid(True, alpha=0.3, axis='y')

# Add coefficient labels
for i, (coef, ci_l, ci_u) in enumerate(zip(coefs, ci_lower, ci_upper)):
    ax.text(i, coef + 0.02, f'{coef:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_heterogeneity.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# Figure 4: DiD Visual
print("Creating Figure 4: DiD Visual...")

fig, ax = plt.subplots(figsize=(10, 6))

# Get the four means
treat_pre = summary_df[(summary_df['Group'] == 'Treatment') & (summary_df['Period'] == 'Pre')]['Fulltime_Rate'].values[0]
treat_post = summary_df[(summary_df['Group'] == 'Treatment') & (summary_df['Period'] == 'Post')]['Fulltime_Rate'].values[0]
control_pre = summary_df[(summary_df['Group'] == 'Control') & (summary_df['Period'] == 'Pre')]['Fulltime_Rate'].values[0]
control_post = summary_df[(summary_df['Group'] == 'Control') & (summary_df['Period'] == 'Post')]['Fulltime_Rate'].values[0]

# Counterfactual for treatment group (same change as control)
treat_counterfactual = treat_pre + (control_post - control_pre)

# Plot
ax.plot([0, 1], [control_pre, control_post], 'r-s', linewidth=2, markersize=10, label='Control Group')
ax.plot([0, 1], [treat_pre, treat_post], 'b-o', linewidth=2, markersize=10, label='Treatment Group')
ax.plot([0, 1], [treat_pre, treat_counterfactual], 'b--', linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# Annotate the DiD
ax.annotate('', xy=(1.05, treat_post), xytext=(1.05, treat_counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (treat_post + treat_counterfactual)/2, f'DiD = {treat_post - treat_counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xlim(-0.2, 1.5)
ax.set_ylim(0.5, 0.75)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did_visual.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

print("\nAll figures created successfully!")

"""
Create figures for the DACA replication report
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['figure.dpi'] = 150

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Calculate weighted means by year and group
def calc_weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

yearly_rates = df.groupby(['YEAR', 'ELIGIBLE']).apply(calc_weighted_mean).unstack()
yearly_rates.columns = ['Control (31-35)', 'Treatment (26-30)']

# =============================================================================
# Figure 1: Parallel Trends
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_rates.index.values
control = yearly_rates['Control (31-35)'].values
treatment = yearly_rates['Treatment (26-30)'].values

# Plot lines
ax.plot(years, treatment, 'o-', color='#2166ac', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, control, 's--', color='#b2182b', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.75, 'DACA\nImplementation\n(June 2012)', fontsize=9, color='gray', va='top')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.legend(loc='lower right')
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.80)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved: Parallel trends plot")

# =============================================================================
# Figure 2: DiD Visualization
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate pre and post means
pre_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                       weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
pre_ctrl = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                      weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
post_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                        weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])
post_ctrl = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                       weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])

# Plot
x = [0, 1]
ax.plot(x, [pre_treat, post_treat], 'o-', color='#2166ac', linewidth=3, markersize=12, label='Treatment (Ages 26-30)')
ax.plot(x, [pre_ctrl, post_ctrl], 's--', color='#b2182b', linewidth=3, markersize=12, label='Control (Ages 31-35)')

# Counterfactual line
counterfactual = pre_treat + (post_ctrl - pre_ctrl)
ax.plot([0, 1], [pre_treat, counterfactual], ':', color='#2166ac', linewidth=2, alpha=0.6, label='Treatment (Counterfactual)')

# Arrow showing treatment effect
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.05, (post_treat + counterfactual)/2, f'DiD = {post_treat-counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-Period\n(2008-2011)', 'Post-Period\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-Time Employment')
ax.legend(loc='upper left')
ax.set_ylim(0.55, 0.75)
ax.set_xlim(-0.3, 1.4)

plt.tight_layout()
plt.savefig('figure2_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_visualization.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved: DiD visualization")

# =============================================================================
# Figure 3: Coefficient Plot
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Results from analysis
models = ['Basic OLS\n(Unweighted)', 'Basic WLS\n(Weighted)', 'WLS\n(Robust SE)',
          '+ Demographics', '+ Education', '+ Year FE', '+ Year & State FE\n(Preferred)']
estimates = [0.0643, 0.0748, 0.0748, 0.0642, 0.0611, 0.0582, 0.0576]
ci_lower = [0.0343, 0.0450, 0.0393, 0.0313, 0.0283, 0.0255, 0.0250]
ci_upper = [0.0942, 0.1045, 0.1102, 0.0971, 0.0938, 0.0909, 0.0902]

y_pos = np.arange(len(models))

# Calculate error bars
errors = [[est - low for est, low in zip(estimates, ci_lower)],
          [high - est for est, high in zip(estimates, ci_upper)]]

ax.errorbar(estimates, y_pos, xerr=errors, fmt='o', color='#2166ac',
            markersize=10, capsize=5, capthick=2, elinewidth=2)

# Add vertical line at zero
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

# Highlight preferred model
ax.scatter([estimates[-1]], [y_pos[-1]], s=200, color='#2166ac', zorder=5,
           edgecolors='gold', linewidths=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment Probability)')
ax.set_title('Estimated Effect of DACA Eligibility on Full-Time Employment')
ax.set_xlim(-0.02, 0.14)

# Add annotation
ax.text(0.10, -0.5, 'Point estimates with 95% confidence intervals',
        fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('figure3_coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficient_plot.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved: Coefficient plot")

# =============================================================================
# Figure 4: Sample Distribution by Year
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

yearly_n = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
yearly_n.columns = ['Control', 'Treatment']

x = np.arange(len(yearly_n.index))
width = 0.35

bars1 = ax.bar(x - width/2, yearly_n['Treatment'], width, label='Treatment (26-30)', color='#2166ac')
bars2 = ax.bar(x + width/2, yearly_n['Control'], width, label='Control (31-35)', color='#b2182b')

ax.axvline(x=3.5, color='gray', linestyle=':', linewidth=2)
ax.text(3.6, max(yearly_n['Treatment'])*0.95, 'DACA\n(2012 omitted)', fontsize=9, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Sample Size by Year and Treatment Status')
ax.set_xticks(x)
ax.set_xticklabels(yearly_n.index)
ax.legend()

plt.tight_layout()
plt.savefig('figure4_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_size.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved: Sample size by year")

print("\nAll figures created successfully!")

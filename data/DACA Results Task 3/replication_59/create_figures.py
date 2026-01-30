"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Figure 1: FT Employment trends by group over time
fig, ax = plt.subplots(figsize=(10, 6))

ft_by_year_eligible = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

ax.plot(ft_by_year_eligible.index, ft_by_year_eligible[1], 'o-', label='Treated (Ages 26-30 in June 2012)', color='blue', linewidth=2, markersize=8)
ax.plot(ft_by_year_eligible.index, ft_by_year_eligible[0], 's--', label='Control (Ages 31-35 in June 2012)', color='red', linewidth=2, markersize=8)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation (2012)')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Group', fontsize=14)
ax.legend(loc='lower right')
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Pre-Post comparison (bar chart)
fig, ax = plt.subplots(figsize=(8, 6))

# Calculate means
ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()

# Create grouped bar chart
x = np.arange(2)
width = 0.35

pre_vals = [ft_means.loc[(0, 0)], ft_means.loc[(1, 0)]]
post_vals = [ft_means.loc[(0, 1)], ft_means.loc[(1, 1)]]

bars1 = ax.bar(x - width/2, pre_vals, width, label='Pre-DACA (2008-2011)', color='lightblue', edgecolor='black')
bars2 = ax.bar(x + width/2, post_vals, width, label='Post-DACA (2013-2016)', color='steelblue', edgecolor='black')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment: Pre vs Post DACA by Group', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Control\n(Ages 31-35)', 'Treated\n(Ages 26-30)'])
ax.legend()
ax.set_ylim(0.55, 0.75)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_prepost.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Event Study plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate year-specific effects relative to 2011
years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
effects = []
ses = []

# Reference year is 2011, so effect is 0
for year in years:
    if year == 2011:
        effects.append(0)
        ses.append(0)
    else:
        df_pair = df[df['YEAR'].isin([2011, year])].copy()
        df_pair['POST'] = (df_pair['YEAR'] == year).astype(int)
        df_pair['ELIGIBLE_POST'] = df_pair['ELIGIBLE'] * df_pair['POST']

        import statsmodels.formula.api as smf
        model = smf.ols('FT ~ ELIGIBLE + POST + ELIGIBLE_POST', data=df_pair).fit(cov_type='HC1')
        effects.append(model.params['ELIGIBLE_POST'])
        ses.append(model.bse['ELIGIBLE_POST'])

effects = np.array(effects)
ses = np.array(ses)

ax.errorbar(years, effects, yerr=1.96*ses, fmt='o-', capsize=5, capthick=2,
            color='blue', ecolor='blue', linewidth=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Year-by-Year Treatment Effects', fontsize=14)
ax.set_xticks(years)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure3_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Coefficient comparison plot
fig, ax = plt.subplots(figsize=(10, 6))

models = ['Basic DiD', 'With Demographics', 'With Educ.', 'State FE',
          'State+Year FE', 'Full Model', 'Clustered SE', 'WLS Weighted']
coeffs = [0.0643, 0.0553, 0.0535, 0.0639, 0.0626, 0.0520, 0.0540, 0.0617]
ses = [0.0153, 0.0142, 0.0142, 0.0153, 0.0152, 0.0141, 0.0150, 0.0142]

y_pos = np.arange(len(models))
ax.barh(y_pos, coeffs, xerr=np.array(ses)*1.96, color='steelblue',
        edgecolor='black', capsize=5, alpha=0.8)

ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (with 95% CI)', fontsize=12)
ax.set_title('Robustness of DiD Estimates Across Specifications', fontsize=14)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Heterogeneous effects
fig, ax = plt.subplots(figsize=(10, 6))

subgroups = ['Male', 'Female', 'HS Only', 'Some College', 'Married', 'Not Married']
coefs = [0.0615, 0.0452, 0.0482, 0.1075, 0.0586, 0.0758]
se_vals = [0.0170, 0.0232, 0.0180, 0.0380, 0.0214, 0.0221]

y_pos = np.arange(len(subgroups))
colors = ['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#ff7f0e', '#ff7f0e']

ax.barh(y_pos, coefs, xerr=np.array(se_vals)*1.96, color=colors,
        edgecolor='black', capsize=5, alpha=0.8)

ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.axvline(x=0.052, color='gray', linestyle=':', linewidth=2, label='Main estimate')
ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Coefficient (with 95% CI)', fontsize=12)
ax.set_title('Heterogeneous Treatment Effects by Subgroup', fontsize=14)
ax.legend()
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('figure5_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.close()

print("All figures created successfully!")
print("- figure1_trends.png")
print("- figure2_prepost.png")
print("- figure3_eventstudy.png")
print("- figure4_robustness.png")
print("- figure5_heterogeneity.png")

"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_80\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

# Figure 1: Trends in Full-Time Employment by Group
print("Creating Figure 1: Trends in FT Employment...")

years = sorted(df['YEAR'].unique())
ft_control = []
ft_treat = []

for year in years:
    control = df[(df['YEAR']==year) & (df['ELIGIBLE']==0)]
    treat = df[(df['YEAR']==year) & (df['ELIGIBLE']==1)]
    ft_control.append(np.average(control['FT'], weights=control['PERWT']))
    ft_treat.append(np.average(treat['FT'], weights=treat['PERWT']))

plt.figure(figsize=(10, 6))
plt.plot(years, ft_control, 'o-', label='Control (Ages 31-35 in 2012)', color='blue', linewidth=2, markersize=8)
plt.plot(years, ft_treat, 's-', label='Treatment (Ages 26-30 in 2012)', color='red', linewidth=2, markersize=8)
plt.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Full-Time Employment Rate', fontsize=12)
plt.title('Trends in Full-Time Employment by DACA Eligibility Group', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.ylim(0.55, 0.80)
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_80\figure1_trends.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png")

# Figure 2: Event Study Coefficients
print("Creating Figure 2: Event Study Coefficients...")

import statsmodels.formula.api as smf

# Create variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIG_X_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

event_formula = ('FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 '
                 '+ ELIG_X_2008 + ELIG_X_2009 + ELIG_X_2010 + ELIG_X_2013 + ELIG_X_2014 + ELIG_X_2015 + ELIG_X_2016 '
                 '+ FEMALE + MARRIED + AGE')

event_model = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                                        cov_kwds={'groups': df['STATEFIP']})

# Extract coefficients
event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [event_model.params[f'ELIG_X_{y}'] if y != 2011 else 0 for y in event_years]
ses = [event_model.bse[f'ELIG_X_{y}'] if y != 2011 else 0 for y in event_years]

plt.figure(figsize=(10, 6))
plt.errorbar(event_years, coefs, yerr=[1.96*se for se in ses], fmt='o', capsize=5,
             capthick=2, markersize=8, color='darkblue', ecolor='steelblue')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation')
plt.fill_between([2007.5, 2011.5], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='blue', label='Pre-treatment')
plt.fill_between([2012.5, 2016.5], [-0.2, -0.2], [0.2, 0.2], alpha=0.1, color='red', label='Post-treatment')
plt.xlabel('Year', fontsize=12)
plt.ylabel('DiD Coefficient (relative to 2011)', fontsize=12)
plt.title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks(event_years)
plt.ylim(-0.15, 0.15)
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_80\figure2_eventstudy.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure2_eventstudy.png")

# Figure 3: Distribution of FT Employment by Group and Period
print("Creating Figure 3: Distribution by Group and Period...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Pre-period
pre_data = df[df['AFTER']==0]
pre_control = pre_data[pre_data['ELIGIBLE']==0]['FT'].mean()
pre_treat = pre_data[pre_data['ELIGIBLE']==1]['FT'].mean()

# Post-period
post_data = df[df['AFTER']==1]
post_control = post_data[post_data['ELIGIBLE']==0]['FT'].mean()
post_treat = post_data[post_data['ELIGIBLE']==1]['FT'].mean()

x = np.arange(2)
width = 0.35

# Pre-period bar chart
axes[0].bar(x - width/2, [pre_control, post_control], width, label='Control (31-35)', color='blue', alpha=0.7)
axes[0].bar(x + width/2, [pre_treat, post_treat], width, label='Treatment (26-30)', color='red', alpha=0.7)
axes[0].set_ylabel('Full-Time Employment Rate', fontsize=12)
axes[0].set_title('FT Employment by Period', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Pre (2008-2011)', 'Post (2013-2016)'])
axes[0].legend()
axes[0].set_ylim(0, 0.8)
axes[0].grid(True, alpha=0.3, axis='y')

# Changes
diff_control = post_control - pre_control
diff_treat = post_treat - pre_treat
did = diff_treat - diff_control

axes[1].bar(['Control\nChange', 'Treatment\nChange', 'DiD\nEstimate'],
           [diff_control, diff_treat, did],
           color=['blue', 'red', 'green'], alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Change in FT Employment Rate', fontsize=12)
axes[1].set_title('Difference-in-Differences Decomposition', fontsize=14)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for i, v in enumerate([diff_control, diff_treat, did]):
    axes[1].text(i, v + 0.005 if v >= 0 else v - 0.015, f'{v:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_80\figure3_did_decomp.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_decomp.png")

# Figure 4: Heterogeneity by Education
print("Creating Figure 4: Heterogeneity by Education...")

educ_levels = ['Less than High School', 'High School Degree', 'Some College', 'BA+']
educ_coefs = []
educ_ses = []

for educ in educ_levels:
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        subset['ELIGIBLE_AFTER'] = subset['ELIGIBLE'] * subset['AFTER']
        model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + AGE',
                       data=subset, weights=subset['PERWT']).fit()
        educ_coefs.append(model.params['ELIGIBLE_AFTER'])
        educ_ses.append(model.bse['ELIGIBLE_AFTER'])
    else:
        educ_coefs.append(np.nan)
        educ_ses.append(np.nan)

# Filter out NaN values
valid_idx = [i for i, c in enumerate(educ_coefs) if not np.isnan(c)]
valid_educ = [educ_levels[i] for i in valid_idx]
valid_coefs = [educ_coefs[i] for i in valid_idx]
valid_ses = [educ_ses[i] for i in valid_idx]

plt.figure(figsize=(10, 6))
x = np.arange(len(valid_educ))
plt.bar(x, valid_coefs, yerr=[1.96*se for se in valid_ses], capsize=5, color='steelblue', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('DiD Coefficient', fontsize=12)
plt.title('Heterogeneity of DACA Effect by Education Level', fontsize=14)
plt.xticks(x, valid_educ, rotation=15, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(r"C:\Users\seraf\DACA Results Task 3\replication_80\figure4_heterogeneity.png", dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: figure4_heterogeneity.png")

print("\nAll figures created successfully!")

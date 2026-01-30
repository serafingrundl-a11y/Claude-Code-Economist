"""
DACA Replication Analysis - Figures Generation
Session 93
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = r"C:\Users\seraf\DACA Results Task 3\replication_93\data\prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

# Helper function
def weighted_mean(group, value_col, weight_col):
    return np.average(group[value_col], weights=group[weight_col])

# Calculate yearly FT rates by group
yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()
yearly_ft.columns = ['Control (31-35)', 'Treatment (26-30)']

# Figure 1: Parallel Trends
fig, ax = plt.subplots(figsize=(10, 6))
years = yearly_ft.index.values
ax.plot(years, yearly_ft['Control (31-35)'], 'b-o', label='Control (Ages 31-35)', linewidth=2, markersize=8)
ax.plot(years, yearly_ft['Treatment (26-30)'], 'r-s', label='Treatment (Ages 26-30)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (June 2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status\n(Weighted by Survey Weights)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim([0.55, 0.80])
ax.grid(True, alpha=0.3)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_93\figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure1_parallel_trends.png")

# Figure 2: Event Study
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
years_list = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
for year in years_list:
    df[f'ELIGIBLE_YEAR_{year}'] = ((df['ELIGIBLE'] == 1) & (df['YEAR'] == year)).astype(int)

formula = 'FT ~ ELIGIBLE + C(YEAR) + C(STATEFIP) + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016'
model_event = smf.wls(formula, data=df, weights=df['PERWT']).fit()

event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = []
event_se = []
for year in event_years:
    if year == 2011:
        event_coefs.append(0)
        event_se.append(0)
    else:
        var = f'ELIGIBLE_YEAR_{year}'
        event_coefs.append(model_event.params[var])
        event_se.append(model_event.bse[var])

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(event_years, event_coefs, yerr=[1.96*se for se in event_se],
            fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=10, color='navy')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Dynamic Treatment Effects of DACA on Full-Time Employment\n(Reference Year: 2011, with 95% Confidence Intervals)', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(event_years)
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_93\figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure2_event_study.png")

# Figure 3: DiD Visualization
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: weighted_mean(x, 'FT', 'PERWT')
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.array([0, 1])
bars1 = ax.bar(x - width/2, [ft_rates.loc[0, 0], ft_rates.loc[0, 1]], width,
               label='Control (Ages 31-35)', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, [ft_rates.loc[1, 0], ft_rates.loc[1, 1]], width,
               label='Treatment (Ages 26-30)', color='indianred', alpha=0.8)

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates: Before vs After DACA\n(Weighted by Survey Weights)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim([0, 0.85])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_93\figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure3_did_bars.png")

# Figure 4: Sample Distribution
fig, ax = plt.subplots(figsize=(10, 6))
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.columns = ['Control (31-35)', 'Treatment (26-30)']
year_counts.plot(kind='bar', ax=ax, color=['steelblue', 'indianred'], alpha=0.8, width=0.8)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Observations', fontsize=12)
ax.set_title('Sample Size by Year and Treatment Group', fontsize=14)
ax.legend(fontsize=10)
ax.set_xticklabels(year_counts.index, rotation=0)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(r'C:\Users\seraf\DACA Results Task 3\replication_93\figure4_sample_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figure4_sample_size.png")

print("\nAll figures generated successfully!")

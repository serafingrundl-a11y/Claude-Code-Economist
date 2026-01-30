"""
Generate figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Figure 1: Year-by-year FT rates by treatment/control
print("Generating Figure 1: Parallel Trends...")

yearly_rates = []
for year in sorted(df['YEAR'].unique()):
    for eligible in [0, 1]:
        subset = df[(df['YEAR']==year) & (df['ELIGIBLE']==eligible)]
        if len(subset) > 0:
            wm = np.average(subset['FT'], weights=subset['PERWT'])
            yearly_rates.append({'Year': year, 'ELIGIBLE': eligible, 'FT_Rate': wm})

yearly_df = pd.DataFrame(yearly_rates)

fig, ax = plt.subplots(figsize=(10, 6))

control = yearly_df[yearly_df['ELIGIBLE']==0]
treat = yearly_df[yearly_df['ELIGIBLE']==1]

ax.plot(control['Year'], control['FT_Rate'], 'o-', label='Control (Age 31-35)', color='blue', linewidth=2, markersize=8)
ax.plot(treat['Year'], treat['FT_Rate'], 's-', label='Treatment (Age 26-30)', color='red', linewidth=2, markersize=8)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.55, 0.80)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png")

# Figure 2: Event Study
print("Generating Figure 2: Event Study...")

import statsmodels.formula.api as smf

# Create year dummies and interactions
for year in df['YEAR'].unique():
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Event study regression (base year: 2011)
year_vars = ' + '.join([f'YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011])
interact_vars = ' + '.join([f'ELIGIBLE_YEAR_{y}' for y in sorted(df['YEAR'].unique()) if y != 2011])
formula_event = f'FT ~ ELIGIBLE + {year_vars} + {interact_vars}'

model_event = smf.wls(formula_event, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Extract coefficients
years = []
coefs = []
ses = []
for year in sorted(df['YEAR'].unique()):
    if year != 2011:
        var = f'ELIGIBLE_YEAR_{year}'
        years.append(year)
        coefs.append(model_event.params[var])
        ses.append(model_event.bse[var])

# Add 2011 as reference (0 coefficient)
years.insert(3, 2011)
coefs.insert(3, 0)
ses.insert(3, 0)

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(years, coefs, yerr=[1.96*s for s in ses], fmt='o', capsize=5,
            color='darkblue', markersize=8, linewidth=2, capthick=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png")

# Figure 3: DiD visualization (2x2 table as bar chart)
print("Generating Figure 3: DiD Visualization...")

# Weighted means for 2x2 table
cells = []
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE']==eligible) & (df['AFTER']==after)]
        wm = np.average(subset['FT'], weights=subset['PERWT'])
        cells.append({'ELIGIBLE': eligible, 'AFTER': after, 'FT_Rate': wm})

cells_df = pd.DataFrame(cells)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.array([0, 1])
width = 0.35

# Pre-DACA bars
pre_control = cells_df[(cells_df['ELIGIBLE']==0) & (cells_df['AFTER']==0)]['FT_Rate'].values[0]
pre_treat = cells_df[(cells_df['ELIGIBLE']==1) & (cells_df['AFTER']==0)]['FT_Rate'].values[0]
# Post-DACA bars
post_control = cells_df[(cells_df['ELIGIBLE']==0) & (cells_df['AFTER']==1)]['FT_Rate'].values[0]
post_treat = cells_df[(cells_df['ELIGIBLE']==1) & (cells_df['AFTER']==1)]['FT_Rate'].values[0]

bars1 = ax.bar(x - width/2, [pre_control, pre_treat], width, label='Pre-DACA (2008-2011)', color='steelblue')
bars2 = ax.bar(x + width/2, [post_control, post_treat], width, label='Post-DACA (2013-2016)', color='coral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment: Treatment vs Control, Pre vs Post DACA', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Control\n(Age 31-35)', 'Treatment\n(Age 26-30)'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0, 0.85)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1%}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Add DiD calculation annotation
did = (post_treat - pre_treat) - (post_control - pre_control)
ax.annotate(f'DiD = {did:.1%}', xy=(0.5, 0.78), fontsize=14, fontweight='bold',
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_bars.png")

# Figure 4: Coefficient comparison across models
print("Generating Figure 4: Model Comparison...")

models = ['Basic\n(Unweighted)', 'Basic\n(Weighted)', 'Clustered\nSE',
          '+ Covariates', '+ State FE', '+ Year FE', 'Full\nModel']
coefs = [0.0643, 0.0748, 0.0748, 0.0616, 0.0737, 0.0721, 0.0583]
ses = [0.0153, 0.0152, 0.0203, 0.0213, 0.0209, 0.0195, 0.0212]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
ax.bar(x, coefs, yerr=[1.96*s for s in ses], capsize=5, color='darkblue', alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient (Effect on FT Employment)', fontsize=12)
ax.set_title('DACA Effect Estimates Across Different Model Specifications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)
ax.set_ylim(-0.02, 0.15)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.annotate(f'{c:.3f}', xy=(i, c + 1.96*s + 0.005), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure4_model_comparison.png")

# Figure 5: Heterogeneity by sex
print("Generating Figure 5: Heterogeneity by Sex...")

fig, ax = plt.subplots(figsize=(8, 6))

groups = ['Male', 'Female', 'Overall']
coefs = [0.0716, 0.0527, 0.0583]
ses = [0.0195, 0.0290, 0.0212]

x = np.arange(len(groups))
colors = ['steelblue', 'coral', 'gray']

bars = ax.bar(x, coefs, yerr=[1.96*s for s in ses], capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect by Sex', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=11)
ax.set_ylim(-0.02, 0.15)
ax.grid(True, alpha=0.3, axis='y')

# Add significance stars
pvals = [0.0002, 0.0696, 0.0059]
for i, (c, p) in enumerate(zip(coefs, pvals)):
    stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.annotate(f'{c:.3f}{stars}', xy=(i, c + 1.96*ses[i] + 0.008), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_heterogeneity_sex.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure5_heterogeneity_sex.png")

# Figure 6: Sample distribution over time
print("Generating Figure 6: Sample Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# By year
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.columns = ['Control', 'Treatment']
year_counts.plot(kind='bar', ax=axes[0], color=['steelblue', 'coral'])
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Number of Observations', fontsize=11)
axes[0].set_title('Sample Size by Year and Treatment Status', fontsize=12)
axes[0].legend(title='Group')
axes[0].tick_params(axis='x', rotation=0)

# By state (top 10)
state_counts = df.groupby(['statename']).size().sort_values(ascending=False).head(10)
state_counts.plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_xlabel('Number of Observations', fontsize=11)
axes[1].set_ylabel('State', fontsize=11)
axes[1].set_title('Top 10 States by Sample Size', fontsize=12)

plt.tight_layout()
plt.savefig('figure6_sample_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: figure6_sample_distribution.png")

print("\nAll figures generated successfully!")

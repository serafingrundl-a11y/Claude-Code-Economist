"""
DACA Replication Study - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)

print("Loading data...")
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# =============================================================================
# Figure 1: Full-Time Employment Trends by Group
# =============================================================================
print("Creating Figure 1: FT Employment Trends...")

# Calculate weighted FT rates by year and group
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_by_year.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
years = ft_by_year.index
ax.plot(years, ft_by_year['Control (Ages 31-35)'] * 100, 'o-',
        color='#1f77b4', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')
ax.plot(years, ft_by_year['Treatment (Ages 26-30)'] * 100, 's-',
        color='#d62728', linewidth=2, markersize=8, label='Treatment (Ages 26-30 in 2012)')

# Add vertical line for DACA
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1] - 2, 'DACA\nImplemented\n(June 2012)',
        fontsize=10, color='gray', ha='left', va='top')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Trends: Treatment vs. Control Groups')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(55, 80)
ax.legend(loc='lower right')
ax.set_xlim(2007.5, 2016.5)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Difference-in-Differences Visualization (Bar Chart)
# =============================================================================
print("Creating Figure 2: DiD Visualization...")

# Calculate group means
ft_rates = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_rates.index = ['Control', 'Treatment']
ft_rates.columns = ['Pre', 'Post']

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(2)
width = 0.35

bars1 = ax.bar(x - width/2, ft_rates['Pre'] * 100, width, label='Pre-DACA (2008-2011)', color='#1f77b4')
bars2 = ax.bar(x + width/2, ft_rates['Post'] * 100, width, label='Post-DACA (2013-2016)', color='#d62728')

ax.set_xlabel('Group')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment: Before and After DACA')
ax.set_xticks(x)
ax.set_xticklabels(['Control\n(Ages 31-35 in 2012)', 'Treatment\n(Ages 26-30 in 2012)'])
ax.legend()
ax.set_ylim(60, 75)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Add arrows showing changes
control_change = (ft_rates.loc['Control', 'Post'] - ft_rates.loc['Control', 'Pre']) * 100
treatment_change = (ft_rates.loc['Treatment', 'Post'] - ft_rates.loc['Treatment', 'Pre']) * 100

ax.annotate(f'Change: {control_change:+.1f}pp', xy=(0, 67), fontsize=10, ha='center', color='gray')
ax.annotate(f'Change: {treatment_change:+.1f}pp', xy=(1, 67), fontsize=10, ha='center', color='gray')

plt.tight_layout()
plt.savefig('figure2_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did_bars.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_did_bars.png/pdf")

# =============================================================================
# Figure 3: Event Study Plot
# =============================================================================
print("Creating Figure 3: Event Study...")

# Run event study regression
import statsmodels.formula.api as smf

# Create year-by-treatment interactions (relative to 2011)
for year in [2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * (df['YEAR'] == year).astype(int)

for year in [2009, 2010, 2011, 2013, 2014, 2015, 2016]:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)

event_formula = 'FT ~ ELIGIBLE + ELIGIBLE_YEAR_2008 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016'

model_event = smf.wls(event_formula,
                       data=df, weights=df['PERWT']).fit(cov_type='cluster',
                                                          cov_kwds={'groups': df['STATEFIP']})

# Extract coefficients
event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coeffs = []
ses = []
for year in event_years:
    if year == 2011:
        coeffs.append(0)  # Reference year
        ses.append(0)
    else:
        coeffs.append(model_event.params[f'ELIGIBLE_YEAR_{year}'])
        ses.append(model_event.bse[f'ELIGIBLE_YEAR_{year}'])

coeffs = np.array(coeffs)
ses = np.array(ses)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(event_years, coeffs, yerr=1.96*ses, fmt='o-',
            color='#1f77b4', linewidth=2, markersize=10, capsize=5, capthick=2)

# Add horizontal line at 0
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Add vertical line for DACA
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.05, ax.get_ylim()[1] - 0.02, 'DACA', fontsize=11, color='red', ha='left')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='red', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Treatment Effect (relative to 2011)')
ax.set_title('Event Study: Year-by-Year Treatment Effects')
ax.set_xticks(event_years)
ax.legend(loc='upper left')
ax.set_xlim(2007.5, 2016.5)

plt.tight_layout()
plt.savefig('figure3_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_event_study.png/pdf")

# =============================================================================
# Figure 4: Sample Distribution by State (Top 10)
# =============================================================================
print("Creating Figure 4: Sample Distribution...")

# Map state FIPS to names
state_names = {
    1: 'Alabama', 4: 'Arizona', 6: 'California', 8: 'Colorado',
    12: 'Florida', 13: 'Georgia', 17: 'Illinois', 22: 'Louisiana',
    26: 'Michigan', 32: 'Nevada', 34: 'New Jersey', 35: 'New Mexico',
    36: 'New York', 37: 'North Carolina', 39: 'Ohio', 40: 'Oklahoma',
    41: 'Oregon', 42: 'Pennsylvania', 48: 'Texas', 53: 'Washington'
}

# Get sample by state (weighted)
state_sample = df.groupby('STATEFIP')['PERWT'].sum().sort_values(ascending=False)
top_states = state_sample.head(10)

fig, ax = plt.subplots(figsize=(10, 6))

# Get state names for top 10
state_labels = [df[df['STATEFIP']==s]['statename'].iloc[0] if len(df[df['STATEFIP']==s]) > 0 else str(s)
                for s in top_states.index]

bars = ax.barh(range(len(top_states)), top_states.values / 1000, color='#1f77b4')
ax.set_yticks(range(len(top_states)))
ax.set_yticklabels(state_labels)
ax.invert_yaxis()
ax.set_xlabel('Weighted Sample Size (thousands)')
ax.set_title('Sample Distribution by State (Top 10)')

for i, v in enumerate(top_states.values):
    ax.text(v/1000 + 5, i, f'{v/1000:.1f}k', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_state_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_state_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_state_distribution.png/pdf")

# =============================================================================
# Figure 5: Heterogeneity Analysis
# =============================================================================
print("Creating Figure 5: Heterogeneity Analysis...")

# Calculate DiD for different subgroups
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['MALE'] = (df['SEX'] == 1).astype(int)
df['MARRIED'] = (df['MARST'] == 1).astype(int)

subgroups = {
    'Overall': df,
    'Male': df[df['SEX'] == 1],
    'Female': df[df['SEX'] == 2],
    'Married': df[df['MARST'] == 1],
    'Not Married': df[df['MARST'] != 1]
}

results = []
for name, subdf in subgroups.items():
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                    data=subdf, weights=subdf['PERWT']).fit(cov_type='cluster',
                                                             cov_kwds={'groups': subdf['STATEFIP']})
    results.append({
        'Group': name,
        'DiD': model.params['ELIGIBLE_AFTER'],
        'SE': model.bse['ELIGIBLE_AFTER'],
        'pvalue': model.pvalues['ELIGIBLE_AFTER']
    })

results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results_df))
ax.errorbar(results_df['DiD'], y_pos, xerr=1.96*results_df['SE'],
            fmt='o', color='#1f77b4', markersize=10, capsize=5, capthick=2)

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(results_df['Group'])
ax.set_xlabel('DiD Estimate (with 95% CI)')
ax.set_title('Heterogeneity Analysis: DiD Estimates by Subgroup')
ax.invert_yaxis()

# Add significance stars
for i, row in results_df.iterrows():
    star = '***' if row['pvalue'] < 0.01 else ('**' if row['pvalue'] < 0.05 else ('*' if row['pvalue'] < 0.1 else ''))
    ax.text(row['DiD'] + 1.96*row['SE'] + 0.01, i, star, va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_heterogeneity.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_heterogeneity.png/pdf")

print("\nAll figures created successfully!")

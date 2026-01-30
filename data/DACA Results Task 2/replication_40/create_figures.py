"""
Create figures for the DACA Replication Study Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# Figure 1: Event Study Plot
# =============================================================================
print("Creating Figure 1: Event Study Plot...")

event_df = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_lower = event_df['CI_Lower'].values
ci_upper = event_df['CI_Upper'].values

# Plot confidence intervals
ax.fill_between(years, ci_lower, ci_upper, alpha=0.3, color='steelblue')

# Plot coefficients
ax.plot(years, coefs, 'o-', color='steelblue', linewidth=2, markersize=8)

# Add reference line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

# Add vertical line at treatment (between 2011 and 2013)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation (June 2012)')

# Labels and title
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect on Full-Time Employment', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)

ax.set_xticks(years)
ax.set_xticklabels([str(int(y)) for y in years], rotation=45)

# Add note about 2012
ax.annotate('2012 excluded\n(implementation year)', xy=(2012, -0.06),
            ha='center', fontsize=9, style='italic', color='gray')

ax.legend(loc='upper left')
ax.set_ylim(-0.12, 0.12)

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("   Figure 1 saved.")

# =============================================================================
# Figure 2: Parallel Trends Visualization
# =============================================================================
print("Creating Figure 2: Parallel Trends...")

# Load main data for this
import pandas as pd
import numpy as np

# Read summary data
summary_df = pd.read_csv('summary_statistics.csv')

# Create year-by-year trends data
# We need to recalculate this from the raw data
dtypes = {
    'YEAR': 'int16',
    'PERWT': 'float32',
    'SEX': 'int8',
    'AGE': 'int8',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
}

cols_to_use = ['YEAR', 'PERWT', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG', 'EMPSTAT', 'UHRSWORK']

print("   Loading data for trends plot...")
df = pd.read_csv('data/data.csv', usecols=cols_to_use, dtype=dtypes)

# Apply filters
df = df[df['HISPAN'] == 1]
df = df[df['BPL'] == 200]
df = df[df['CITIZEN'] == 3]
df = df[(df['YRIMMIG'] > 0) & (df['YRIMMIG'] <= 2007)]
df['age_at_immigration'] = df['YRIMMIG'] - df['BIRTHYR']
df = df[df['age_at_immigration'] < 16]

def calc_age_on_june_2012(birthyr, birthqtr):
    if birthqtr <= 2:
        return 2012 - birthyr
    else:
        return 2012 - birthyr - 1

df['age_june_2012'] = df.apply(lambda x: calc_age_on_june_2012(x['BIRTHYR'], x['BIRTHQTR']), axis=1)
df['treat'] = ((df['age_june_2012'] >= 26) & (df['age_june_2012'] <= 30)).astype(int)
df['control'] = ((df['age_june_2012'] >= 31) & (df['age_june_2012'] <= 35)).astype(int)
df = df[(df['treat'] == 1) | (df['control'] == 1)]
df = df[df['YEAR'] != 2012]
df['fulltime'] = ((df['UHRSWORK'] >= 35) & (df['EMPSTAT'] == 1)).astype(int)

# Calculate weighted means by year and group
def weighted_mean(data, values_col, weights_col):
    return np.average(data[values_col], weights=data[weights_col])

trends_data = []
for year in sorted(df['YEAR'].unique()):
    for group in [0, 1]:
        sub = df[(df['YEAR'] == year) & (df['treat'] == group)]
        if len(sub) > 0:
            ft_rate = weighted_mean(sub, 'fulltime', 'PERWT')
            trends_data.append({
                'Year': year,
                'Group': 'Treatment (26-30)' if group == 1 else 'Control (31-35)',
                'FT_Rate': ft_rate
            })

trends_df = pd.DataFrame(trends_data)

fig, ax = plt.subplots(figsize=(10, 6))

for group in ['Treatment (26-30)', 'Control (31-35)']:
    sub = trends_df[trends_df['Group'] == group]
    color = 'steelblue' if 'Treatment' in group else 'coral'
    marker = 'o' if 'Treatment' in group else 's'
    ax.plot(sub['Year'], sub['FT_Rate'], f'{marker}-', label=group,
            linewidth=2, markersize=8, color=color)

ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA\nImplementation', xy=(2012, 0.52), ha='center', fontsize=10, color='red')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends: Treatment vs. Control Groups', fontsize=14)
ax.legend(loc='lower right')
ax.set_ylim(0.45, 0.70)

years_all = list(range(2006, 2017))
years_all.remove(2012)
ax.set_xticks(years_all)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("   Figure 2 saved.")

# =============================================================================
# Figure 3: Model Comparison Forest Plot
# =============================================================================
print("Creating Figure 3: Model Comparison...")

results_df = pd.read_csv('results_summary.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results_df))
models = results_df['Model'].values
estimates = results_df['DiD_Estimate'].values
ci_lower = results_df['CI_Lower'].values
ci_upper = results_df['CI_Upper'].values

# Horizontal error bars
ax.errorbar(estimates, y_pos, xerr=[estimates - ci_lower, ci_upper - estimates],
            fmt='o', markersize=10, capsize=5, capthick=2, color='steelblue', linewidth=2)

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('Comparison of DiD Estimates Across Model Specifications', fontsize=14)

# Add estimate values as text
for i, (est, ci_l, ci_u) in enumerate(zip(estimates, ci_lower, ci_upper)):
    ax.annotate(f'{est:.3f}', xy=(est, i), xytext=(5, 0), textcoords='offset points',
                fontsize=10, va='center')

ax.set_xlim(-0.02, 0.12)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('figure3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("   Figure 3 saved.")

# =============================================================================
# Figure 4: Subgroup Analysis
# =============================================================================
print("Creating Figure 4: Subgroup Analysis...")

# Subgroup results (from the analysis output)
subgroups = ['Male', 'Female']
estimates_sub = [0.0362, 0.0426]
se_sub = [0.0139, 0.0180]
ci_lower_sub = [e - 1.96*s for e, s in zip(estimates_sub, se_sub)]
ci_upper_sub = [e + 1.96*s for e, s in zip(estimates_sub, se_sub)]

fig, ax = plt.subplots(figsize=(8, 5))

y_pos = np.arange(len(subgroups))
colors = ['steelblue', 'coral']

ax.barh(y_pos, estimates_sub, xerr=[np.array(estimates_sub) - np.array(ci_lower_sub),
                                     np.array(ci_upper_sub) - np.array(estimates_sub)],
        color=colors, capsize=5, height=0.5, alpha=0.8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(subgroups)
ax.set_xlabel('DiD Estimate (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('DACA Effect on Full-Time Employment by Sex', fontsize=14)

for i, est in enumerate(estimates_sub):
    ax.annotate(f'{est:.4f}', xy=(est + 0.005, i), va='center', fontsize=10)

ax.set_xlim(-0.02, 0.10)

plt.tight_layout()
plt.savefig('figure4_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_subgroups.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("   Figure 4 saved.")

print("\nAll figures created successfully!")

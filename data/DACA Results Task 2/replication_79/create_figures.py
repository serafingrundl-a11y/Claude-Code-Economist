"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

# Load data
print("Loading data...")
chunks = []
for chunk in pd.read_csv('data/data.csv', chunksize=500000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Apply filters
data = df.copy()
data = data[data['HISPAN'] == 1]
data = data[data['BPL'] == 200]
data = data[data['CITIZEN'] == 3]
data = data[data['YEAR'] != 2012]

# Calculate age as of June 15, 2012
data['age_june2012'] = 2012 - data['BIRTHYR']
data.loc[data['BIRTHQTR'] > 2, 'age_june2012'] = data['age_june2012'] - 1

# Age at immigration
data['age_at_immig'] = data['YRIMMIG'] - data['BIRTHYR']
data = data[data['age_at_immig'] < 16]
data = data[data['YRIMMIG'] <= 2007]

# Filter to ages 26-35
data = data[(data['age_june2012'] >= 26) & (data['age_june2012'] <= 35)]

# Create variables
data['treated'] = (data['age_june2012'] <= 30).astype(int)
data['post'] = (data['YEAR'] >= 2013).astype(int)
data['fulltime'] = (data['UHRSWORK'] >= 35).astype(int)
data['female'] = (data['SEX'] == 2).astype(int)

print(f"Sample size: {len(data):,}")

# =============================================================================
# Figure 1: Full-time Employment Trends by Treatment Status
# =============================================================================
print("Creating Figure 1: Employment trends...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate means by year and treatment status
trends = data.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()

years = sorted(data['YEAR'].unique())
treat_means = [trends.loc[y, 1] if 1 in trends.columns else np.nan for y in years]
control_means = [trends.loc[y, 0] if 0 in trends.columns else np.nan for y in years]

# Plot
ax.plot(years, treat_means, 'b-o', linewidth=2, markersize=8, label='Treatment (Age 26-30)')
ax.plot(years, control_means, 'r-s', linewidth=2, markersize=8, label='Control (Age 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='green', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

# Labels
ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Treatment Status')
ax.set_xticks(years)
ax.legend(loc='best')
ax.set_ylim([0.55, 0.70])

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("Creating Figure 2: Event study...")

# Load event study results
event_df = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

years_event = event_df['Year'].values
coefs = event_df['Coefficient'].values
ci_lower = event_df['CI_Lower'].values
ci_upper = event_df['CI_Upper'].values

# Add reference year (2011)
all_years = np.append(years_event[:5], [2011, *years_event[5:]])
all_coefs = np.append(coefs[:5], [0, *coefs[5:]])
all_ci_lower = np.append(ci_lower[:5], [0, *ci_lower[5:]])
all_ci_upper = np.append(ci_upper[:5], [0, *ci_upper[5:]])

# Plot
ax.errorbar(all_years, all_coefs,
            yerr=[all_coefs - all_ci_lower, all_ci_upper - all_coefs],
            fmt='o-', capsize=4, capthick=2, linewidth=2, markersize=8,
            color='blue', ecolor='blue', alpha=0.8)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at 2012
ax.axvline(x=2012, color='green', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray', label='Pre-period')
ax.axvspan(2012.5, 2016, alpha=0.1, color='blue', label='Post-period')

# Labels
ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(all_years)
ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png/pdf")

# =============================================================================
# Figure 3: Heterogeneity by Gender
# =============================================================================
print("Creating Figure 3: Heterogeneity by gender...")

fig, ax = plt.subplots(figsize=(8, 6))

# Calculate means by year, treatment, and gender
trends_gender = data.groupby(['YEAR', 'treated', 'female'])['fulltime'].mean().unstack([1, 2])

years = sorted(data['YEAR'].unique())

# Male treatment and control
male_treat = [trends_gender.loc[y, (1, 0)] if (1, 0) in trends_gender.columns else np.nan for y in years]
male_control = [trends_gender.loc[y, (0, 0)] if (0, 0) in trends_gender.columns else np.nan for y in years]

# Female treatment and control
female_treat = [trends_gender.loc[y, (1, 1)] if (1, 1) in trends_gender.columns else np.nan for y in years]
female_control = [trends_gender.loc[y, (0, 1)] if (0, 1) in trends_gender.columns else np.nan for y in years]

ax.plot(years, male_treat, 'b-o', linewidth=2, label='Male Treatment')
ax.plot(years, male_control, 'b--s', linewidth=2, alpha=0.7, label='Male Control')
ax.plot(years, female_treat, 'r-o', linewidth=2, label='Female Treatment')
ax.plot(years, female_control, 'r--s', linewidth=2, alpha=0.7, label='Female Control')

ax.axvline(x=2012, color='green', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Trends by Gender and Treatment Status')
ax.set_xticks(years)
ax.legend(loc='best')
ax.set_ylim([0.30, 0.90])

plt.tight_layout()
plt.savefig('figure3_gender.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_gender.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_gender.png/pdf")

# =============================================================================
# Figure 4: DiD Graphical Illustration
# =============================================================================
print("Creating Figure 4: DiD illustration...")

fig, ax = plt.subplots(figsize=(8, 6))

# Get means for 2x2
pre_treat = data[(data['treated']==1) & (data['post']==0)]['fulltime'].mean()
post_treat = data[(data['treated']==1) & (data['post']==1)]['fulltime'].mean()
pre_control = data[(data['treated']==0) & (data['post']==0)]['fulltime'].mean()
post_control = data[(data['treated']==0) & (data['post']==1)]['fulltime'].mean()

# Counterfactual
counterfactual = pre_treat + (post_control - pre_control)

# Plot
periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
x = [0, 1]

ax.plot(x, [pre_treat, post_treat], 'b-o', linewidth=3, markersize=12, label='Treatment (Observed)')
ax.plot(x, [pre_control, post_control], 'r-s', linewidth=3, markersize=12, label='Control (Observed)')
ax.plot(x, [pre_treat, counterfactual], 'b--', linewidth=2, alpha=0.5, label='Treatment (Counterfactual)')

# Add arrow for DiD effect
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.15, (post_treat + counterfactual)/2, f'DiD Effect\n= {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: Graphical Illustration')
ax.legend(loc='upper right')
ax.set_xlim([-0.2, 1.5])
ax.set_ylim([0.55, 0.70])

plt.tight_layout()
plt.savefig('figure4_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did_illustration.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_did_illustration.png/pdf")

# =============================================================================
# Figure 5: Coefficient Plot for Main Results
# =============================================================================
print("Creating Figure 5: Coefficient plot...")

# Load regression results
results_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

models = results_df['Model'].values
coefs = results_df['DiD_Estimate'].values
ses = results_df['Std_Error'].values

y_pos = np.arange(len(models))

ax.errorbar(coefs, y_pos, xerr=1.96*ses, fmt='o', capsize=5, capthick=2,
            markersize=10, color='blue', ecolor='blue', alpha=0.8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('Difference-in-Differences Estimate')
ax.set_title('DiD Estimates Across Model Specifications')

# Add coefficient values
for i, (coef, se) in enumerate(zip(coefs, ses)):
    ax.text(coef + 0.01, i, f'{coef:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure5_coef_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_coef_plot.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_coef_plot.png/pdf")

# =============================================================================
# Figure 6: Sample Distribution by Age
# =============================================================================
print("Creating Figure 6: Sample distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

age_counts = data.groupby(['age_june2012', 'treated']).size().unstack()
age_counts.plot(kind='bar', ax=ax, color=['red', 'blue'], alpha=0.7, width=0.8)

ax.set_xlabel('Age as of June 15, 2012')
ax.set_ylabel('Number of Observations')
ax.set_title('Sample Distribution by Age and Treatment Status')
ax.legend(['Control (31-35)', 'Treatment (26-30)'], loc='upper right')
ax.set_xticklabels([f'{int(x)}' for x in age_counts.index], rotation=0)

plt.tight_layout()
plt.savefig('figure6_age_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_age_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure6_age_distribution.png/pdf")

print("\nAll figures created successfully!")

"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

print("Creating figures...")

# Load and prepare data
df = pd.read_csv('data/data.csv')

# Define sample (same as analysis.py)
df['hispanic_mexican'] = (df['HISPAN'] == 1) & (df['BPL'] == 200)
df['non_citizen'] = df['CITIZEN'] == 3
df['age_in_2012'] = df['AGE'] - (df['YEAR'] - 2012)
df['treat'] = (df['age_in_2012'] >= 26) & (df['age_in_2012'] <= 30)
df['control'] = (df['age_in_2012'] >= 31) & (df['age_in_2012'] <= 35)
df['arrived_before_16'] = np.where(df['YRIMMIG'] > 0, (df['YRIMMIG'] - df['BIRTHYR']) < 16, False)
df['in_us_since_2007'] = np.where(df['YRIMMIG'] > 0, df['YRIMMIG'] <= 2007, False)

base_sample = df['hispanic_mexican'] & df['non_citizen']
eligible_sample = base_sample & df['arrived_before_16'] & df['in_us_since_2007']
analysis_sample = eligible_sample & (df['treat'] | df['control'])

dfa = df[analysis_sample].copy()
dfa['fulltime'] = (dfa['UHRSWORK'] >= 35).astype(int)
dfa['post'] = (dfa['YEAR'] >= 2013).astype(int)
dfa['treated'] = dfa['treat'].astype(int)
dfa['treat_post'] = dfa['treated'] * dfa['post']
dfa_clean = dfa[dfa['YEAR'] != 2012].copy()

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# FIGURE 1: Parallel Trends
# =============================================================================
print("  Figure 1: Parallel Trends...")

# Calculate weighted means by year and treatment
years = sorted(dfa_clean['YEAR'].unique())
treat_means = []
control_means = []

for year in years:
    # Treatment group
    treat_data = dfa_clean[(dfa_clean['YEAR'] == year) & (dfa_clean['treated'] == 1)]
    treat_mean = np.average(treat_data['fulltime'], weights=treat_data['PERWT'])
    treat_means.append(treat_mean * 100)

    # Control group
    control_data = dfa_clean[(dfa_clean['YEAR'] == year) & (dfa_clean['treated'] == 0)]
    control_mean = np.average(control_data['fulltime'], weights=control_data['PERWT'])
    control_means.append(control_mean * 100)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, treat_means, 'o-', linewidth=2, markersize=8,
        color='#2E86AB', label='Treatment (Ages 26-30 in 2012)')
ax.plot(years, control_means, 's--', linewidth=2, markersize=8,
        color='#A23B72', label='Control (Ages 31-35 in 2012)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplementation', xy=(2012, 72), fontsize=10,
            ha='center', color='gray')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(55, 75)
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 2: Event Study
# =============================================================================
print("  Figure 2: Event Study...")

# Run event study regression
dfa_clean['year_factor'] = dfa_clean['YEAR'].astype(str)
reference_year = 2011

year_interactions = []
for year in years:
    if year != reference_year:
        dfa_clean[f'treat_x_{year}'] = dfa_clean['treated'] * (dfa_clean['YEAR'] == year).astype(int)
        year_interactions.append(f'treat_x_{year}')

formula = f'fulltime ~ treated + C(year_factor) + {" + ".join(year_interactions)}'
model_es = smf.wls(formula, data=dfa_clean, weights=dfa_clean['PERWT']).fit(cov_type='HC1')

# Extract coefficients
coefs = []
ses = []
for year in years:
    if year == reference_year:
        coefs.append(0)
        ses.append(0)
    else:
        coefs.append(model_es.params[f'treat_x_{year}'] * 100)
        ses.append(model_es.bse[f'treat_x_{year}'] * 100)

coefs = np.array(coefs)
ses = np.array(ses)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot confidence intervals
ax.fill_between(years, coefs - 1.96*ses, coefs + 1.96*ses,
                alpha=0.3, color='#2E86AB')
ax.plot(years, coefs, 'o-', linewidth=2, markersize=8, color='#2E86AB')

# Add reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA implementation
ax.axvline(x=2011.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplementation', xy=(2011.5, 12), fontsize=10,
            ha='center', color='gray')

# Add reference year marker
ax.scatter([2011], [0], color='red', s=100, zorder=5, marker='*')
ax.annotate('Reference Year', xy=(2011, -2), fontsize=9, ha='center', color='red')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (Percentage Points)', fontsize=12)
ax.set_title('Event Study: Differential Effects by Year\n(Relative to 2011)',
             fontsize=14, fontweight='bold')
ax.set_xticks(years)
ax.set_ylim(-6, 14)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 3: DiD Visual
# =============================================================================
print("  Figure 3: DiD Visualization...")

# Calculate group means for pre/post
means = dfa_clean.groupby(['treated', 'post']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT']) * 100
).unstack()

fig, ax = plt.subplots(figsize=(8, 6))

# Pre-post bars for each group
x = np.array([0, 1])
width = 0.35

bars1 = ax.bar(x - width/2, [means.loc[0, 0], means.loc[0, 1]], width,
               label='Control', color='#A23B72', alpha=0.8)
bars2 = ax.bar(x + width/2, [means.loc[1, 0], means.loc[1, 1]], width,
               label='Treatment', color='#2E86AB', alpha=0.8)

ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax.set_xlabel('Period', fontsize=12)
ax.set_title('Full-Time Employment by Group and Period', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend(fontsize=10)
ax.set_ylim(0, 80)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visual.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 4: Heterogeneity by Gender
# =============================================================================
print("  Figure 4: Heterogeneity by Gender...")

fig, ax = plt.subplots(figsize=(10, 6))

# By gender
for gender, label, color in [(1, 'Male', '#2E86AB'), (2, 'Female', '#E94F37')]:
    subset = dfa_clean[dfa_clean['SEX'] == gender]

    year_means = []
    for year in years:
        year_data = subset[(subset['YEAR'] == year) & (subset['treated'] == 1)]
        if len(year_data) > 0:
            year_mean = np.average(year_data['fulltime'], weights=year_data['PERWT'])
            year_means.append(year_mean * 100)
        else:
            year_means.append(np.nan)

    linestyle = '-' if gender == 1 else '--'
    ax.plot(years, year_means, 'o' + linestyle, linewidth=2, markersize=6,
            color=color, label=f'Treatment - {label}')

ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax.set_title('Full-Time Employment Trends by Gender (Treatment Group)',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure4_gender_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_gender_heterogeneity.pdf', bbox_inches='tight')
plt.close()

# =============================================================================
# FIGURE 5: Sample Construction Flowchart Data
# =============================================================================
print("  Figure 5: Sample sizes for flowchart...")

# Create text file with sample sizes for flowchart
with open('sample_flowchart.txt', 'w') as f:
    f.write("Sample Construction\n")
    f.write("="*50 + "\n")
    f.write(f"Total ACS observations (2006-2016): {len(df):,}\n")
    f.write(f"Hispanic-Mexican, Mexican-born: {df['hispanic_mexican'].sum():,}\n")
    f.write(f"+ Non-citizen: {(df['hispanic_mexican'] & df['non_citizen']).sum():,}\n")
    f.write(f"+ DACA eligible criteria: {eligible_sample.sum():,}\n")
    f.write(f"+ Age group restriction: {analysis_sample.sum():,}\n")
    f.write(f"Final sample (excl. 2012): {len(dfa_clean):,}\n")

print("\nAll figures created successfully!")

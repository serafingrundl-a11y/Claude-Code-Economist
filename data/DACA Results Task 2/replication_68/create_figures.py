"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# Figure 1: Trends in Full-time Employment by Group
# =============================================================================
print("Creating Figure 1: Trends in Full-time Employment...")

# Manual calculation of means by year and treatment group
# We need to reload the data for this

chunks = []
for chunk in pd.read_csv('data/data.csv', chunksize=500000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Apply same sample selection
df_sample = df[(df['HISPAN'] == 1) & (df['BPL'] == 200)].copy()
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample['arrived_before_16'] = (df_sample['age_at_arrival'] < 16) & (df_sample['YRIMMIG'] > 0)
df_sample['in_us_since_2007'] = (df_sample['YRIMMIG'] <= 2007) & (df_sample['YRIMMIG'] > 0)
df_sample['not_citizen'] = df_sample['CITIZEN'].isin([3, 4, 5])
df_sample['daca_eligible_base'] = (
    df_sample['arrived_before_16'] &
    df_sample['in_us_since_2007'] &
    df_sample['not_citizen']
)
df_sample['treatment_group'] = df_sample['BIRTHYR'].between(1982, 1986)
df_sample['control_group'] = df_sample['BIRTHYR'].between(1977, 1981)
df_sample['treated'] = np.where(df_sample['treatment_group'], 1,
                                np.where(df_sample['control_group'], 0, np.nan))
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Filter to analysis sample (excluding 2012)
analysis_df = df_sample[
    (df_sample['treated'].notna()) &
    (df_sample['YEAR'] != 2012) &
    (df_sample['daca_eligible_base'])
].copy()

# Calculate means by year and treatment
yearly_means = analysis_df.groupby(['YEAR', 'treated'])['fulltime'].mean().unstack()

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_means.index.values
treatment_means = yearly_means[1.0].values
control_means = yearly_means[0.0].values

ax.plot(years, treatment_means, 'o-', color='blue', linewidth=2, markersize=8,
        label='Treatment (Ages 26-30 in 2012)')
ax.plot(years, control_means, 's--', color='red', linewidth=2, markersize=8,
        label='Control (Ages 31-35 in 2012)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]-0.01, 'DACA\nImplemented', fontsize=10, ha='left')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Proportion Full-time Employed', fontsize=12)
ax.set_title('Full-time Employment Rates by Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(years)
ax.set_ylim(0.55, 0.70)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("   Saved figure1_trends.png and figure1_trends.pdf")

# =============================================================================
# Figure 2: Event Study Coefficients
# =============================================================================
print("Creating Figure 2: Event Study...")

years_event = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []

for yr in years_event:
    if yr == 2011:
        coefs.append(0)  # Reference year
        ses.append(0)
    else:
        coefs.append(results[f'event_{yr}_coef'])
        ses.append(results[f'event_{yr}_se'])

coefs = np.array(coefs)
ses = np.array(ses)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(years_event, coefs, yerr=1.96*ses, fmt='o', color='blue',
            capsize=4, capthick=2, linewidth=2, markersize=8)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]-0.01, 'DACA', fontsize=10, ha='left')

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green', label='Post-DACA')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment', fontsize=14)
ax.set_xticks(years_event)
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("   Saved figure2_eventstudy.png and figure2_eventstudy.pdf")

# =============================================================================
# Figure 3: Comparison of Model Specifications
# =============================================================================
print("Creating Figure 3: Model Comparisons...")

models = ['Model 1:\nBasic DiD', 'Model 2:\n+ Demographics',
          'Model 3:\n+ Education', 'Model 4:\n+ State/Year FE']
coefs_models = [results['model1_coef'], results['model2_coef'],
                results['model3_coef'], results['model4_coef']]
ses_models = [results['model1_se'], results['model2_se'],
              results['model3_se'], results['model4_se']]

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.6

bars = ax.bar(x, coefs_models, width, yerr=[1.96*se for se in ses_models],
              capsize=5, color=['steelblue', 'steelblue', 'steelblue', 'darkblue'],
              edgecolor='black', linewidth=1)

# Mark preferred specification
bars[3].set_color('darkgreen')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient\n(Effect on Full-time Employment)', fontsize=12)
ax.set_title('Comparison of Model Specifications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10)

# Add significance stars
for i, (coef, pval) in enumerate(zip(coefs_models,
    [results['model1_pval'], results['model2_pval'],
     results['model3_pval'], results['model4_pval']])):
    if pval < 0.01:
        stars = '***'
    elif pval < 0.05:
        stars = '**'
    elif pval < 0.10:
        stars = '*'
    else:
        stars = ''
    ax.annotate(stars, (i, coef + 1.96*ses_models[i] + 0.005), ha='center', fontsize=12)

ax.set_ylim(-0.02, 0.10)

plt.tight_layout()
plt.savefig('figure3_models.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_models.pdf', bbox_inches='tight')
plt.close()

print("   Saved figure3_models.png and figure3_models.pdf")

# =============================================================================
# Figure 4: Heterogeneous Effects by Gender
# =============================================================================
print("Creating Figure 4: Effects by Gender...")

fig, ax = plt.subplots(figsize=(8, 6))

genders = ['Overall', 'Males', 'Females']
coefs_gender = [results['model4_coef'], results['model_male_coef'], results['model_female_coef']]
ses_gender = [results['model4_se'], results['model_male_se'], results['model_female_se']]

x = np.arange(len(genders))
bars = ax.bar(x, coefs_gender, 0.6, yerr=[1.96*se for se in ses_gender],
              capsize=5, color=['darkgreen', 'steelblue', 'coral'],
              edgecolor='black', linewidth=1)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('Effect of DACA Eligibility by Gender', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(genders, fontsize=12)

plt.tight_layout()
plt.savefig('figure4_gender.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_gender.pdf', bbox_inches='tight')
plt.close()

print("   Saved figure4_gender.png and figure4_gender.pdf")

print("\nAll figures created successfully!")

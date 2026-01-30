"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.chdir(r'C:\Users\seraf\DACA Results Task 1\replication_29')

# Load results
with open('analysis_results.json', 'r') as f:
    results = json.load(f)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# =============================================================================
# Figure 1: Event Study
# =============================================================================

event_data = pd.DataFrame(results['event_study'])
event_data = event_data.sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
ax.errorbar(event_data['year'], event_data['coef'],
            yerr=[event_data['coef'] - event_data['ci_low'],
                  event_data['ci_high'] - event_data['coef']],
            fmt='o-', capsize=4, capthick=2, linewidth=2, markersize=8,
            color='#2166ac', ecolor='#92c5de')

# Add zero line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='DACA Implementation (June 2012)')

# Labels
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-time Employment\n(Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-time Employment', fontsize=14)
ax.set_xticks(range(2006, 2017))
ax.legend(loc='lower right')

# Add shaded region for pre-treatment
ax.axvspan(2006, 2011.5, alpha=0.1, color='gray')
ax.text(2008.5, ax.get_ylim()[1]*0.9, 'Pre-DACA', fontsize=10, ha='center', color='gray')
ax.text(2014.5, ax.get_ylim()[1]*0.9, 'Post-DACA', fontsize=10, ha='center', color='gray')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: Event Study")

# =============================================================================
# Figure 2: Full-time Employment Rates Over Time by Group
# =============================================================================

# Need to reload data to create this figure
print("Loading data for trends figure...")
chunk_size = 500000
data_path = 'data/data.csv'

chunks = []
for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
    filtered = chunk[(chunk['HISPAN'] == 1) & (chunk['BPL'] == 200)]
    if len(filtered) > 0:
        chunks.append(filtered)

df = pd.concat(chunks, ignore_index=True)

# Recreate eligibility variables
df['age_at_daca'] = 2012 - df['BIRTHYR']
df['YRIMMIG_valid'] = df['YRIMMIG'].replace(0, np.nan)
df['age_at_immig'] = df['YRIMMIG_valid'] - df['BIRTHYR']
df['arrived_before_16'] = (df['age_at_immig'] < 16) & (df['age_at_immig'] >= 0)
df['under_31_at_daca'] = df['BIRTHYR'] >= 1982
df['in_us_since_2007'] = df['YRIMMIG_valid'] <= 2007
df['non_citizen'] = df['CITIZEN'] == 3
df['daca_eligible'] = (
    df['arrived_before_16'] &
    df['under_31_at_daca'] &
    df['in_us_since_2007'] &
    df['non_citizen']
)
df['fulltime'] = (df['UHRSWORK'] >= 35).astype(int)

# Working age non-citizens
df_analysis = df[(df['AGE'] >= 16) & (df['AGE'] <= 64) & df['non_citizen']].copy()

# Calculate yearly means
yearly_eligible = df_analysis[df_analysis['daca_eligible']].groupby('YEAR')['fulltime'].mean() * 100
yearly_ineligible = df_analysis[~df_analysis['daca_eligible']].groupby('YEAR')['fulltime'].mean() * 100

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(yearly_eligible.index, yearly_eligible.values, 'o-', linewidth=2, markersize=8,
        label='DACA Eligible', color='#2166ac')
ax.plot(yearly_ineligible.index, yearly_ineligible.values, 's--', linewidth=2, markersize=8,
        label='DACA Ineligible', color='#b2182b')

ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7,
           label='DACA Implementation')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate (%)', fontsize=12)
ax.set_title('Full-time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.set_xticks(range(2006, 2017))
ax.legend(loc='best')
ax.set_ylim([35, 70])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: Employment Trends")

# =============================================================================
# Figure 3: Model Comparison
# =============================================================================

models = ['Basic DiD', '+ Demographics', '+ Year FE', '+ State & Year FE']
coefs = [results['model_comparisons'][f'model{i}_basic']['coef'] if i==1
         else results['model_comparisons'][f'model{i}_controls']['coef'] if i==2
         else results['model_comparisons'][f'model{i}_year_fe']['coef'] if i==3
         else results['model_comparisons'][f'model{i}_full_fe']['coef']
         for i in [1,2,3,4]]
ses = [results['model_comparisons'][f'model{i}_basic']['se'] if i==1
       else results['model_comparisons'][f'model{i}_controls']['se'] if i==2
       else results['model_comparisons'][f'model{i}_year_fe']['se'] if i==3
       else results['model_comparisons'][f'model{i}_full_fe']['se']
       for i in [1,2,3,4]]

coefs = [results['model_comparisons']['model1_basic']['coef'],
         results['model_comparisons']['model2_controls']['coef'],
         results['model_comparisons']['model3_year_fe']['coef'],
         results['model_comparisons']['model4_full_fe']['coef']]
ses = [results['model_comparisons']['model1_basic']['se'],
       results['model_comparisons']['model2_controls']['se'],
       results['model_comparisons']['model3_year_fe']['se'],
       results['model_comparisons']['model4_full_fe']['se']]

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(models))
ax.bar(x_pos, [c*100 for c in coefs], yerr=[s*1.96*100 for s in ses],
       capsize=5, color=['#67a9cf', '#67a9cf', '#67a9cf', '#2166ac'],
       edgecolor='black', linewidth=1)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('DiD Estimate (Percentage Points)', fontsize=12)
ax.set_title('Effect of DACA Eligibility on Full-time Employment:\nModel Comparison', fontsize=14)

# Add value labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.text(i, c*100 + s*1.96*100 + 0.5, f'{c*100:.1f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_model_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: Model Comparison")

# =============================================================================
# Figure 4: Difference-in-Differences Visualization
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6))

pre_means = [results['pre_post_means']['pre_eligible']*100,
             results['pre_post_means']['pre_ineligible']*100]
post_means = [results['pre_post_means']['post_eligible']*100,
              results['pre_post_means']['post_ineligible']*100]

# Plot pre and post means
ax.plot([0, 1], [pre_means[0], post_means[0]], 'o-', linewidth=3, markersize=12,
        label='DACA Eligible', color='#2166ac')
ax.plot([0, 1], [pre_means[1], post_means[1]], 's--', linewidth=3, markersize=12,
        label='DACA Ineligible', color='#b2182b')

# Add counterfactual
pre_diff = pre_means[0] - pre_means[1]
counterfactual_post = post_means[1] + pre_diff
ax.plot([0, 1], [pre_means[0], counterfactual_post], 'o:', linewidth=2, markersize=10,
        label='Counterfactual (Eligible)', color='#2166ac', alpha=0.5)

# Arrow showing treatment effect
ax.annotate('', xy=(1.05, post_means[0]), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_means[0] + counterfactual_post)/2,
        f'DiD Effect\n{results["preferred_estimate"]["coefficient"]*100:.1f} pp',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-time Employment Rate (%)', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim([-0.2, 1.4])
ax.set_ylim([35, 70])

plt.tight_layout()
plt.savefig('figure4_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did_visual.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved: DiD Visualization")

print("\nAll figures created successfully!")

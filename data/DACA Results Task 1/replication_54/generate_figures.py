"""
Generate figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Loading data for figures...")

# Load data
dtypes = {
    'YEAR': 'int16',
    'STATEFIP': 'int8',
    'SEX': 'int8',
    'AGE': 'int16',
    'BIRTHQTR': 'int8',
    'BIRTHYR': 'int16',
    'HISPAN': 'int8',
    'BPL': 'int16',
    'CITIZEN': 'int8',
    'YRIMMIG': 'int16',
    'EMPSTAT': 'int8',
    'UHRSWORK': 'int8',
    'EDUC': 'int8',
    'MARST': 'int8',
    'PERWT': 'float64'
}

cols_needed = ['YEAR', 'STATEFIP', 'SEX', 'AGE', 'BIRTHQTR', 'BIRTHYR',
               'HISPAN', 'BPL', 'CITIZEN', 'YRIMMIG',
               'EMPSTAT', 'UHRSWORK', 'EDUC', 'MARST', 'PERWT']

df = pd.read_csv('data/data.csv', usecols=cols_needed, dtype=dtypes)

# Sample restriction
df_sample = df[(df['HISPAN'] == 1) & (df['BPL'] == 200) & (df['CITIZEN'] == 3)].copy()
df_sample = df_sample[df_sample['YRIMMIG'] > 0].copy()

# DACA eligibility
df_sample['age_at_arrival'] = df_sample['YRIMMIG'] - df_sample['BIRTHYR']
df_sample['daca_eligible'] = ((df_sample['age_at_arrival'] < 16) &
                               (df_sample['BIRTHYR'] >= 1982) &
                               (df_sample['YRIMMIG'] <= 2007))

# Full-time employment
df_sample['fulltime'] = (df_sample['UHRSWORK'] >= 35).astype(int)

# Age restriction
df_analysis = df_sample[(df_sample['AGE'] >= 18) & (df_sample['AGE'] <= 50)].copy()

print("Generating figures...")

# ============================================================================
# Figure 1: Full-Time Employment Trends by DACA Eligibility
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate yearly means by eligibility
yearly_means = df_analysis.groupby(['YEAR', 'daca_eligible']).apply(
    lambda x: np.average(x['fulltime'], weights=x['PERWT'])
).unstack()

# Plot
ax.plot(yearly_means.index, yearly_means[True], 'o-', color='#2166ac',
        linewidth=2, markersize=8, label='DACA Eligible')
ax.plot(yearly_means.index, yearly_means[False], 's-', color='#b2182b',
        linewidth=2, markersize=8, label='Non-Eligible')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 0.65, 'DACA\nImplementation', fontsize=10, color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.4, 0.75)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================================
# Figure 2: Event Study Coefficients
# ============================================================================
# Event study results from analysis
event_study_coefs = {
    2006: 0.0070,
    2007: 0.0049,
    2008: 0.0166,
    2009: 0.0206,
    2010: 0.0179,
    2011: 0.0,  # Reference year
    2013: 0.0192,
    2014: 0.0343,
    2015: 0.0505,
    2016: 0.0529
}

event_study_se = {
    2006: 0.0125,
    2007: 0.0067,
    2008: 0.0148,
    2009: 0.0112,
    2010: 0.0158,
    2011: 0.0,
    2013: 0.0096,
    2014: 0.0126,
    2015: 0.0125,
    2016: 0.0116
}

years = list(event_study_coefs.keys())
coefs = [event_study_coefs[y] for y in years]
ci_low = [event_study_coefs[y] - 1.96 * event_study_se[y] for y in years]
ci_high = [event_study_coefs[y] + 1.96 * event_study_se[y] for y in years]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(years, coefs, yerr=[np.array(coefs) - np.array(ci_low),
                                  np.array(ci_high) - np.array(coefs)],
            fmt='o', color='#2166ac', markersize=8, capsize=4, capthick=2,
            linewidth=2, elinewidth=2)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 0.07, 'DACA', fontsize=10, color='gray')

# Shade pre-treatment period
ax.axvspan(2005.5, 2012, alpha=0.1, color='gray')
ax.text(2009, -0.04, 'Pre-Treatment', fontsize=10, ha='center', color='gray')
ax.text(2014.5, -0.04, 'Post-Treatment', fontsize=10, ha='center', color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment',
             fontsize=14, fontweight='bold')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(-0.05, 0.09)
ax.set_xticks(range(2006, 2017))

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================================
# Figure 3: Sample Composition
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age distribution by eligibility
eligible_ages = df_analysis[df_analysis['daca_eligible']]['AGE']
non_eligible_ages = df_analysis[~df_analysis['daca_eligible']]['AGE']

axes[0].hist(eligible_ages, bins=range(18, 52, 2), alpha=0.6, color='#2166ac',
             label='DACA Eligible', density=True, edgecolor='white')
axes[0].hist(non_eligible_ages, bins=range(18, 52, 2), alpha=0.6, color='#b2182b',
             label='Non-Eligible', density=True, edgecolor='white')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Density', fontsize=12)
axes[0].set_title('Age Distribution by DACA Eligibility', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)

# Sample size by year and eligibility
sample_counts = df_analysis.groupby(['YEAR', 'daca_eligible']).size().unstack()
sample_counts.plot(kind='bar', ax=axes[1], color=['#b2182b', '#2166ac'], width=0.8)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_ylabel('Number of Observations', fontsize=12)
axes[1].set_title('Sample Size by Year and DACA Eligibility', fontsize=12, fontweight='bold')
axes[1].legend(['Non-Eligible', 'DACA Eligible'], fontsize=10)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('figure3_sample.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_sample.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================================
# Figure 4: DiD Visualization
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate pre and post means
pre_eligible = yearly_means.loc[2006:2011, True].mean()
post_eligible = yearly_means.loc[2013:2016, True].mean()
pre_non = yearly_means.loc[2006:2011, False].mean()
post_non = yearly_means.loc[2013:2016, False].mean()

# Plot actual trends
ax.plot([2009, 2014], [pre_eligible, post_eligible], 'o-', color='#2166ac',
        linewidth=2, markersize=10, label='DACA Eligible (Actual)')
ax.plot([2009, 2014], [pre_non, post_non], 's-', color='#b2182b',
        linewidth=2, markersize=10, label='Non-Eligible (Actual)')

# Plot counterfactual for eligible
counterfactual_post = pre_eligible + (post_non - pre_non)
ax.plot([2009, 2014], [pre_eligible, counterfactual_post], 'o--', color='#2166ac',
        linewidth=2, markersize=10, alpha=0.5, label='DACA Eligible (Counterfactual)')

# Add arrow showing treatment effect
ax.annotate('', xy=(2014.3, post_eligible), xytext=(2014.3, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(2014.5, (post_eligible + counterfactual_post)/2,
        f'DiD Effect\n{post_eligible - counterfactual_post:.3f}',
        fontsize=11, color='green', fontweight='bold')

ax.axvline(x=2011.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2011.6, 0.45, 'DACA', fontsize=10, color='gray')

ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2008, 2016)
ax.set_ylim(0.4, 0.75)
ax.set_xticks([2009, 2014])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])

plt.tight_layout()
plt.savefig('figure4_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# ============================================================================
# Figure 5: Robustness Checks Summary
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Robustness results
models = ['Main Specification\n(Age 18-50)',
          'Age 16-35',
          'Include 2012\nas Post',
          'Unweighted']
effects = [0.0279, 0.0056, 0.0189, 0.0296]
ses = [0.0036, 0.0050, 0.0030, 0.0043]

y_pos = np.arange(len(models))
ci_low = [e - 1.96*s for e, s in zip(effects, ses)]
ci_high = [e + 1.96*s for e, s in zip(effects, ses)]

ax.barh(y_pos, effects, xerr=[np.array(effects) - np.array(ci_low),
                               np.array(ci_high) - np.array(effects)],
        color='#2166ac', alpha=0.8, capsize=5)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlabel('DiD Treatment Effect', fontsize=12)
ax.set_title('Robustness of Treatment Effect Estimates', fontsize=14, fontweight='bold')

# Add effect values
for i, (eff, cil, cih) in enumerate(zip(effects, ci_low, ci_high)):
    ax.text(cih + 0.002, i, f'{eff:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_robustness.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

print("\nAll figures generated successfully!")

"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load data and results
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# ============================================================================
# Figure 1: Parallel Trends - FT Employment by Year and Group
# ============================================================================
print("Creating Figure 1: Parallel Trends...")

# Calculate weighted means by year and treatment
yearly_means = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
yearly_means.columns = ['Control (31-35)', 'Treatment (26-30)']

fig, ax = plt.subplots(figsize=(10, 6))
years = yearly_means.index.values

ax.plot(years, yearly_means['Treatment (26-30)'], 'o-', color='#2166ac',
        linewidth=2, markersize=8, label='Treatment (26-30 at DACA)')
ax.plot(years, yearly_means['Control (31-35)'], 's--', color='#b2182b',
        linewidth=2, markersize=8, label='Control (31-35 at DACA)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.75, 'DACA\nImplemented', fontsize=10, color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Full-Time Employment Trends by Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.55, 0.80)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_parallel_trends.png/pdf")

# ============================================================================
# Figure 2: Event Study Coefficients
# ============================================================================
print("Creating Figure 2: Event Study...")

import statsmodels.formula.api as smf

# Create year dummies interacted with treatment
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2011'] = (df['YEAR'] == 2011).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

df['T_2008'] = df['YEAR_2008'] * df['ELIGIBLE']
df['T_2009'] = df['YEAR_2009'] * df['ELIGIBLE']
df['T_2010'] = df['YEAR_2010'] * df['ELIGIBLE']
df['T_2013'] = df['YEAR_2013'] * df['ELIGIBLE']
df['T_2014'] = df['YEAR_2014'] * df['ELIGIBLE']
df['T_2015'] = df['YEAR_2015'] * df['ELIGIBLE']
df['T_2016'] = df['YEAR_2016'] * df['ELIGIBLE']

event_model = smf.wls('FT ~ ELIGIBLE + C(YEAR) + T_2008 + T_2009 + T_2010 + T_2013 + T_2014 + T_2015 + T_2016',
                      data=df, weights=df['PERWT']).fit(cov_type='HC1')

years_event = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [event_model.params['T_2008'], event_model.params['T_2009'],
         event_model.params['T_2010'], 0,  # 2011 is reference
         event_model.params['T_2013'], event_model.params['T_2014'],
         event_model.params['T_2015'], event_model.params['T_2016']]
ses = [event_model.bse['T_2008'], event_model.bse['T_2009'],
       event_model.bse['T_2010'], 0,  # 2011 is reference
       event_model.bse['T_2013'], event_model.bse['T_2014'],
       event_model.bse['T_2015'], event_model.bse['T_2016']]

fig, ax = plt.subplots(figsize=(10, 6))

ax.errorbar(years_event, coefs, yerr=[1.96*se for se in ses],
            fmt='o', color='#2166ac', capsize=5, capthick=2,
            markersize=8, linewidth=2, label='Point Estimate with 95% CI')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Shade pre and post periods
ax.axvspan(2007.5, 2012, alpha=0.1, color='gray')
ax.axvspan(2012, 2016.5, alpha=0.1, color='blue')

ax.text(2009.5, 0.12, 'Pre-DACA', fontsize=11, ha='center', color='gray')
ax.text(2014.5, 0.12, 'Post-DACA', fontsize=11, ha='center', color='#2166ac')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(-0.20, 0.20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_event_study.png/pdf")

# ============================================================================
# Figure 3: DiD Visualization (2x2)
# ============================================================================
print("Creating Figure 3: DiD Visualization...")

# Calculate means
means = df.groupby(['ELIGIBLE', 'AFTER']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(8, 6))

# Treatment group
ax.plot([0, 1], [means.loc[1, 0], means.loc[1, 1]], 'o-', color='#2166ac',
        linewidth=3, markersize=12, label='Treatment (26-30)')
# Control group
ax.plot([0, 1], [means.loc[0, 0], means.loc[0, 1]], 's--', color='#b2182b',
        linewidth=3, markersize=12, label='Control (31-35)')

# Counterfactual
cf = means.loc[1, 0] + (means.loc[0, 1] - means.loc[0, 0])
ax.plot([0, 1], [means.loc[1, 0], cf], 'o:', color='#2166ac',
        linewidth=2, markersize=8, alpha=0.5, label='Treatment Counterfactual')

# DiD arrow
ax.annotate('', xy=(1.05, means.loc[1, 1]), xytext=(1.05, cf),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.08, (means.loc[1, 1] + cf)/2, f'DiD\n{means.loc[1, 1] - cf:.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_xlabel('Period', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(-0.2, 1.3)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_did_visualization.png/pdf")

# ============================================================================
# Figure 4: Sample Distribution by Age and Year
# ============================================================================
print("Creating Figure 4: Sample Distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Age distribution by treatment group
ax = axes[0]
df[df['ELIGIBLE']==1]['AGE'].hist(bins=20, alpha=0.6, color='#2166ac',
                                   label='Treatment (26-30)', ax=ax, density=True)
df[df['ELIGIBLE']==0]['AGE'].hist(bins=20, alpha=0.6, color='#b2182b',
                                   label='Control (31-35)', ax=ax, density=True)
ax.set_xlabel('Age at Survey', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Age Distribution by Treatment Status', fontsize=12)
ax.legend()

# Sample size by year
ax = axes[1]
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.columns = ['Control', 'Treatment']
year_counts.plot(kind='bar', ax=ax, color=['#b2182b', '#2166ac'], alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sample Size', fontsize=12)
ax.set_title('Sample Size by Year and Treatment Status', fontsize=12)
ax.legend(title='Group')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('figure4_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_sample_distribution.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_sample_distribution.png/pdf")

# ============================================================================
# Figure 5: Heterogeneity by Gender
# ============================================================================
print("Creating Figure 5: Heterogeneity by Gender...")

fig, ax = plt.subplots(figsize=(8, 6))

# Calculate means by gender
for sex_val, sex_name, color, marker in [(1, 'Male', '#2166ac', 'o'), (2, 'Female', '#b2182b', 's')]:
    sub = df[df['SEX'] == sex_val]
    yearly = sub.groupby(['YEAR', 'ELIGIBLE']).apply(
        lambda x: np.average(x['FT'], weights=x['PERWT'])
    ).unstack()

    ax.plot(yearly.index, yearly[1], f'{marker}-', color=color,
            linewidth=2, markersize=8, label=f'{sex_name} - Treatment')
    ax.plot(yearly.index, yearly[0], f'{marker}--', color=color,
            linewidth=2, markersize=8, alpha=0.5, label=f'{sex_name} - Control')

ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Full-Time Employment by Gender and Treatment Status', fontsize=14)
ax.legend(loc='lower right', fontsize=9)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure5_heterogeneity_gender.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_heterogeneity_gender.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure5_heterogeneity_gender.png/pdf")

print("\nAll figures created successfully!")

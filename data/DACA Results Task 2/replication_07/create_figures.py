"""
Create Figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load and prepare data
print("Loading data...")
df = pd.read_csv('data/data.csv')

# Sample selection
df_sample = df[df['HISPAN'] == 1].copy()
df_sample = df_sample[df_sample['BPL'] == 200].copy()
df_sample = df_sample[df_sample['CITIZEN'] == 3].copy()

# Age calculation
df_sample['age_2012'] = 2012 - df_sample['BIRTHYR']
df_sample.loc[df_sample['BIRTHQTR'].isin([3, 4]), 'age_2012'] -= 1

# Treatment/control groups
df_sample['treat'] = ((df_sample['age_2012'] >= 26) & (df_sample['age_2012'] <= 30)).astype(int)
df_sample['control'] = ((df_sample['age_2012'] >= 31) & (df_sample['age_2012'] <= 35)).astype(int)

df_analysis = df_sample[(df_sample['treat'] == 1) | (df_sample['control'] == 1)].copy()

# DACA eligibility
df_analysis = df_analysis[df_analysis['YRIMMIG'] > 0].copy()
df_analysis['age_at_arrival'] = df_analysis['YRIMMIG'] - df_analysis['BIRTHYR']
df_analysis = df_analysis[df_analysis['age_at_arrival'] < 16].copy()
df_analysis = df_analysis[df_analysis['YRIMMIG'] <= 2007].copy()

# Exclude 2012
df_analysis = df_analysis[df_analysis['YEAR'] != 2012].copy()
df_analysis['post'] = (df_analysis['YEAR'] >= 2013).astype(int)
df_analysis['fulltime'] = (df_analysis['UHRSWORK'] >= 35).astype(int)

# Demographics
df_analysis['educ_hs'] = (df_analysis['EDUC'] >= 6).astype(int)
df_analysis['married'] = (df_analysis['MARST'].isin([1, 2])).astype(int)
df_analysis['female'] = (df_analysis['SEX'] == 2).astype(int)

print(f"Analysis sample: {len(df_analysis):,} observations")

# ============================================================================
# FIGURE 1: Parallel Trends
# ============================================================================
print("Creating Figure 1: Parallel Trends...")

trends = df_analysis.groupby(['YEAR', 'treat']).agg({
    'fulltime': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

treat_data = trends[trends['treat'] == 1]
control_data = trends[trends['treat'] == 0]

ax.plot(treat_data['YEAR'], treat_data['fulltime'] * 100,
        'o-', color='#1f77b4', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(control_data['YEAR'], control_data['fulltime'] * 100,
        's--', color='#ff7f0e', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line at 2012
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 56, 'DACA\nImplemented', fontsize=10, color='red', alpha=0.8)

# Add shaded region for post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Rates by Treatment Status')
ax.legend(loc='lower right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(55, 72)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# ============================================================================
# FIGURE 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event Study...")

# Create year dummies (2011 as reference)
for year in [2006, 2007, 2008, 2009, 2010, 2013, 2014, 2015, 2016]:
    df_analysis[f'year_{year}'] = (df_analysis['YEAR'] == year).astype(int)
    df_analysis[f'treat_year_{year}'] = df_analysis['treat'] * df_analysis[f'year_{year}']

event_formula = ('fulltime ~ treat + year_2006 + year_2007 + year_2008 + year_2009 + year_2010 + '
                 'year_2013 + year_2014 + year_2015 + year_2016 + '
                 'treat_year_2006 + treat_year_2007 + treat_year_2008 + treat_year_2009 + treat_year_2010 + '
                 'treat_year_2013 + treat_year_2014 + treat_year_2015 + treat_year_2016 + '
                 'female + educ_hs + married')

event_model = smf.ols(event_formula, data=df_analysis).fit()

# Extract coefficients
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []
for year in years:
    if year == 2011:
        coefs.append(0)
        ses.append(0)
    else:
        var = f'treat_year_{year}'
        coefs.append(event_model.params[var])
        ses.append(event_model.bse[var])

coefs = np.array(coefs)
ses = np.array(ses)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot with confidence intervals
ax.errorbar(years, coefs * 100, yerr=1.96 * np.array(ses) * 100,
            fmt='o', color='#1f77b4', capsize=5, capthick=2,
            markersize=8, linewidth=2)

# Connect points with line
ax.plot(years, coefs * 100, '-', color='#1f77b4', linewidth=1.5, alpha=0.5)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at 2012
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, -4, 'DACA', fontsize=10, color='red', alpha=0.8)

# Add shaded region for post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('DiD Coefficient (Percentage Points)')
ax.set_title('Event Study: Treatment Effect by Year\n(Reference Year: 2011)')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png/pdf")

# ============================================================================
# FIGURE 3: DiD Visualization
# ============================================================================
print("Creating Figure 3: DiD Visualization...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate means by group and period
means = df_analysis.groupby(['treat', 'post'])['fulltime'].mean() * 100

# Pre-period means
treat_pre = means[(1, 0)]
control_pre = means[(0, 0)]

# Post-period means
treat_post = means[(1, 1)]
control_post = means[(0, 1)]

# Plot actual trends
ax.plot([0, 1], [treat_pre, treat_post], 'o-', color='#1f77b4',
        linewidth=2.5, markersize=12, label='Treatment (Ages 26-30)')
ax.plot([0, 1], [control_pre, control_post], 's-', color='#ff7f0e',
        linewidth=2.5, markersize=12, label='Control (Ages 31-35)')

# Plot counterfactual for treatment group
treat_counterfactual = treat_pre + (control_post - control_pre)
ax.plot([0, 1], [treat_pre, treat_counterfactual], 'o--', color='#1f77b4',
        linewidth=1.5, markersize=8, alpha=0.5, label='Treatment Counterfactual')

# Add arrow showing the effect
ax.annotate('', xy=(1.05, treat_post), xytext=(1.05, treat_counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.12, (treat_post + treat_counterfactual)/2,
        f'DiD Effect:\n{treat_post - treat_counterfactual:.1f}pp',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-Period\n(2006-2011)', 'Post-Period\n(2013-2016)'])
ax.set_xlabel('')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Difference-in-Differences Visualization')
ax.legend(loc='lower left')
ax.set_xlim(-0.3, 1.4)
ax.set_ylim(58, 68)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_visualization.png/pdf")

# ============================================================================
# FIGURE 4: Coefficient Comparison
# ============================================================================
print("Creating Figure 4: Model Comparison...")

# Run all models
df_analysis['treat_post'] = df_analysis['treat'] * df_analysis['post']

model_names = []
model_coefs = []
model_ses = []

# Model 1: Simple
m1 = smf.ols('fulltime ~ treat + post + treat_post', data=df_analysis).fit()
model_names.append('Simple DiD')
model_coefs.append(m1.params['treat_post'])
model_ses.append(m1.bse['treat_post'])

# Model 2: Demographics
m2 = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married', data=df_analysis).fit()
model_names.append('+ Demographics')
model_coefs.append(m2.params['treat_post'])
model_ses.append(m2.bse['treat_post'])

# Model 3: Year FE
m3 = smf.ols('fulltime ~ treat + C(YEAR) + treat_post + female + educ_hs + married', data=df_analysis).fit()
model_names.append('+ Year FE')
model_coefs.append(m3.params['treat_post'])
model_ses.append(m3.bse['treat_post'])

# Model 4: Robust SE
m4 = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married', data=df_analysis).fit(cov_type='HC1')
model_names.append('Robust SE')
model_coefs.append(m4.params['treat_post'])
model_ses.append(m4.bse['treat_post'])

# Narrow age window
df_narrow = df_analysis[(df_analysis['age_2012'].isin([27,28,29])) |
                         (df_analysis['age_2012'].isin([32,33,34]))].copy()
df_narrow['treat_narrow'] = df_narrow['age_2012'].isin([27,28,29]).astype(int)
df_narrow['treat_post_narrow'] = df_narrow['treat_narrow'] * df_narrow['post']
m5 = smf.ols('fulltime ~ treat_narrow + post + treat_post_narrow + female + educ_hs + married', data=df_narrow).fit()
model_names.append('Narrow Age')
model_coefs.append(m5.params['treat_post_narrow'])
model_ses.append(m5.bse['treat_post_narrow'])

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(model_names))
coefs_pct = np.array(model_coefs) * 100
ses_pct = np.array(model_ses) * 100

# Plot coefficients with error bars
ax.errorbar(coefs_pct, y_pos, xerr=1.96*ses_pct, fmt='o', color='#1f77b4',
            capsize=5, capthick=2, markersize=10, linewidth=2)

# Add vertical line at 0
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(model_names)
ax.set_xlabel('DiD Coefficient (Percentage Points)')
ax.set_title('Comparison of DiD Estimates Across Specifications\n(95% Confidence Intervals)')
ax.set_xlim(-2, 9)

# Add coefficient values as text
for i, (c, s) in enumerate(zip(coefs_pct, ses_pct)):
    ax.text(c + 1.96*s + 0.3, i, f'{c:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_model_comparison.png/pdf")

# ============================================================================
# FIGURE 5: Heterogeneity by Gender
# ============================================================================
print("Creating Figure 5: Heterogeneity by Gender...")

df_male = df_analysis[df_analysis['female'] == 0]
df_female = df_analysis[df_analysis['female'] == 1]

m_male = smf.ols('fulltime ~ treat + post + treat_post + educ_hs + married', data=df_male).fit()
m_female = smf.ols('fulltime ~ treat + post + treat_post + educ_hs + married', data=df_female).fit()
m_all = smf.ols('fulltime ~ treat + post + treat_post + female + educ_hs + married', data=df_analysis).fit()

fig, ax = plt.subplots(figsize=(8, 5))

groups = ['Overall', 'Male', 'Female']
coefs = [m_all.params['treat_post'], m_male.params['treat_post'], m_female.params['treat_post']]
ses = [m_all.bse['treat_post'], m_male.bse['treat_post'], m_female.bse['treat_post']]
colors = ['#1f77b4', '#2ca02c', '#d62728']

y_pos = np.arange(len(groups))
coefs_pct = np.array(coefs) * 100
ses_pct = np.array(ses) * 100

for i, (c, s, col) in enumerate(zip(coefs_pct, ses_pct, colors)):
    ax.errorbar(c, i, xerr=1.96*s, fmt='o', color=col,
                capsize=5, capthick=2, markersize=12, linewidth=2)

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(groups)
ax.set_xlabel('DiD Coefficient (Percentage Points)')
ax.set_title('Heterogeneity Analysis: Effect by Gender\n(95% Confidence Intervals)')
ax.set_xlim(-2, 10)

for i, (c, s) in enumerate(zip(coefs_pct, ses_pct)):
    ax.text(c + 1.96*s + 0.3, i, f'{c:.2f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_heterogeneity.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_heterogeneity.png/pdf")

# ============================================================================
# FIGURE 6: Sample Composition by Year
# ============================================================================
print("Creating Figure 6: Sample Composition...")

sample_by_year = df_analysis.groupby(['YEAR', 'treat']).size().unstack()
sample_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

x = sample_by_year.index
width = 0.35

ax.bar(x - width/2, sample_by_year['Treatment (26-30)'], width, label='Treatment (Ages 26-30)', color='#1f77b4')
ax.bar(x + width/2, sample_by_year['Control (31-35)'], width, label='Control (Ages 31-35)', color='#ff7f0e')

ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.95, 'DACA', fontsize=10, color='red', alpha=0.8)

ax.set_xlabel('Year')
ax.set_ylabel('Sample Size')
ax.set_title('Sample Size by Year and Treatment Status')
ax.legend()
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure6_sample_composition.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_sample_composition.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure6_sample_composition.png/pdf")

print("\nAll figures created successfully!")

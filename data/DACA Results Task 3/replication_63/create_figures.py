"""
DACA Replication Study - Figure Generation Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('seaborn-v0_8-whitegrid')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Create interaction term
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)

# Calculate weighted means function
def weighted_mean(data, value_col, weight_col):
    return np.average(data[value_col], weights=data[weight_col])

# =============================================================================
# Figure 1: Full-time Employment Rates Over Time by Treatment Status
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate weighted FT rates by year and eligible status
years = sorted(df['YEAR'].unique())
treat_rates = []
control_rates = []

for year in years:
    treat_data = df[(df['YEAR'] == year) & (df['ELIGIBLE'] == 1)]
    control_data = df[(df['YEAR'] == year) & (df['ELIGIBLE'] == 0)]
    treat_rates.append(weighted_mean(treat_data, 'FT', 'PERWT'))
    control_rates.append(weighted_mean(control_data, 'FT', 'PERWT'))

# Plot
ax.plot(years, treat_rates, 'o-', color='#2171b5', linewidth=2, markersize=8,
        label='Treatment Group (Ages 26-30)')
ax.plot(years, control_rates, 's-', color='#cb181d', linewidth=2, markersize=8,
        label='Control Group (Ages 31-35)')

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, 0.76, 'DACA\nImplemented', fontsize=9, color='gray', va='top')

# Add shaded region for post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Full-time Employment Rates by Treatment Status Over Time', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(2007.5, 2016.5)
ax.set_ylim(0.55, 0.80)
ax.set_xticks(years)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Difference-in-Differences Visualization
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate pre and post means
pre_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)], 'FT', 'PERWT')
post_treat = weighted_mean(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)], 'FT', 'PERWT')
pre_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)], 'FT', 'PERWT')
post_control = weighted_mean(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)], 'FT', 'PERWT')

# Plot actual values
x = [0, 1]
ax.plot(x, [pre_treat, post_treat], 'o-', color='#2171b5', linewidth=3, markersize=12,
        label='Treatment (Ages 26-30) - Actual')
ax.plot(x, [pre_control, post_control], 's-', color='#cb181d', linewidth=3, markersize=12,
        label='Control (Ages 31-35)')

# Counterfactual for treatment group
counterfactual = pre_treat + (post_control - pre_control)
ax.plot(x, [pre_treat, counterfactual], 'o--', color='#2171b5', linewidth=2,
        markersize=10, alpha=0.5, label='Treatment - Counterfactual')

# Arrow showing treatment effect
mid_x = 1.05
ax.annotate('', xy=(mid_x, post_treat), xytext=(mid_x, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(mid_x + 0.05, (post_treat + counterfactual)/2, f'DiD = {post_treat-counterfactual:.3f}',
        fontsize=11, color='green', fontweight='bold', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Effect of DACA Eligibility on Full-time Employment',
             fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.60, 0.72)

plt.tight_layout()
plt.savefig('figure2_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_did.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: figure2_did.png/pdf")

# =============================================================================
# Figure 3: Event Study / Year-Specific Effects
# =============================================================================
import statsmodels.formula.api as smf

# Run year-specific regressions
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate year-specific DiD effects relative to 2008
years_all = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = []
ses = []

# Base year is 2008
df['year_2008'] = (df['YEAR'] == 2008).astype(int)

for year in years_all:
    # Create indicator for this year vs 2008
    subset = df[df['YEAR'].isin([2008, year])]
    if year == 2008:
        coefs.append(0)
        ses.append(0)
    else:
        subset['post'] = (subset['YEAR'] == year).astype(int)
        subset['interaction'] = subset['ELIGIBLE'] * subset['post']
        model = smf.wls('FT ~ ELIGIBLE + post + interaction',
                       data=subset, weights=subset['PERWT']).fit(
            cov_type='cluster', cov_kwds={'groups': subset['STATEFIP']})
        coefs.append(model.params['interaction'])
        ses.append(model.bse['interaction'])

coefs = np.array(coefs)
ses = np.array(ses)

# Plot
ax.errorbar(years_all, coefs, yerr=1.96*ses, fmt='o-', color='#2171b5',
            linewidth=2, markersize=8, capsize=4, capthick=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Shade pre/post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA Period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference-in-Differences Coefficient\n(Relative to 2008)', fontsize=12)
ax.set_title('Event Study: Year-Specific Treatment Effects', fontsize=14, fontweight='bold')
ax.set_xticks(years_all)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2007.5, 2016.5)

plt.tight_layout()
plt.savefig('figure3_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: figure3_event_study.png/pdf")

# =============================================================================
# Figure 4: Subgroup Analysis
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Function to calculate DiD for subgroup
def get_did_subgroup(data):
    model = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER',
                   data=data, weights=data['PERWT']).fit(
        cov_type='cluster', cov_kwds={'groups': data['STATEFIP']})
    return model.params['ELIGIBLE_AFTER'], model.bse['ELIGIBLE_AFTER']

# Panel A: By Gender
ax = axes[0]
labels = ['Male', 'Female']
did_vals = []
se_vals = []
for sex in [1, 2]:
    subset = df[df['SEX'] == sex]
    did, se = get_did_subgroup(subset)
    did_vals.append(did)
    se_vals.append(se)

bars = ax.bar(labels, did_vals, yerr=[1.96*s for s in se_vals],
              color=['#2171b5', '#cb181d'], capsize=5, alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_ylabel('DiD Coefficient', fontsize=11)
ax.set_title('(A) By Gender', fontsize=12, fontweight='bold')
for i, (v, s) in enumerate(zip(did_vals, se_vals)):
    ax.text(i, v + 1.96*s + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Panel B: By Education
ax = axes[1]
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
educ_short = ['<HS', 'HS', 'Some Col.', '2-Year', 'BA+']
did_vals = []
se_vals = []
for educ in educ_order:
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        did, se = get_did_subgroup(subset)
    else:
        did, se = 0, 0
    did_vals.append(did)
    se_vals.append(se)

colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(educ_order)))
bars = ax.bar(educ_short, did_vals, yerr=[1.96*s for s in se_vals],
              color=colors, capsize=4, alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_ylabel('DiD Coefficient', fontsize=11)
ax.set_title('(B) By Education Level', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)

# Panel C: By Marital Status
ax = axes[2]
labels = ['Married', 'Not Married']
did_vals = []
se_vals = []
for married in [1, 0]:
    subset = df[df['MARRIED'] == married]
    did, se = get_did_subgroup(subset)
    did_vals.append(did)
    se_vals.append(se)

bars = ax.bar(labels, did_vals, yerr=[1.96*s for s in se_vals],
              color=['#4daf4a', '#984ea3'], capsize=5, alpha=0.8)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_ylabel('DiD Coefficient', fontsize=11)
ax.set_title('(C) By Marital Status', fontsize=12, fontweight='bold')
for i, (v, s) in enumerate(zip(did_vals, se_vals)):
    ax.text(i, v + 1.96*s + 0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_subgroups.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_subgroups.pdf', bbox_inches='tight')
plt.close()

print("Figure 4 saved: figure4_subgroups.png/pdf")

# =============================================================================
# Figure 5: Geographic Distribution
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Get state-level DiD effects for top states
top_states = df.groupby('statename')['PERWT'].sum().nlargest(15).index.tolist()
state_did = []
state_se = []

for state in top_states:
    subset = df[df['statename'] == state]
    if len(subset) > 50:
        try:
            did, se = get_did_subgroup(subset)
        except:
            did, se = np.nan, np.nan
    else:
        did, se = np.nan, np.nan
    state_did.append(did)
    state_se.append(se)

# Sort by DiD value
sorted_idx = np.argsort(state_did)
sorted_states = [top_states[i] for i in sorted_idx if not np.isnan(state_did[i])]
sorted_did = [state_did[i] for i in sorted_idx if not np.isnan(state_did[i])]
sorted_se = [state_se[i] for i in sorted_idx if not np.isnan(state_did[i])]

colors = ['#cb181d' if d < 0 else '#2171b5' for d in sorted_did]
bars = ax.barh(sorted_states, sorted_did, xerr=[1.96*s for s in sorted_se],
               color=colors, capsize=3, alpha=0.8)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('DiD Coefficient', fontsize=12)
ax.set_title('State-Level Treatment Effects (Top 15 States by Population)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('figure5_states.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_states.pdf', bbox_inches='tight')
plt.close()

print("Figure 5 saved: figure5_states.png/pdf")

print("\nAll figures generated successfully!")

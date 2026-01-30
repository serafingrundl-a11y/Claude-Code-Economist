"""
Create figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(r"C:\Users\seraf\DACA Results Task 1\replication_15")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10, 6)

# =============================================================================
# Figure 1: Yearly trends in full-time employment
# =============================================================================
print("Creating Figure 1: Yearly trends...")

yearly_means = pd.read_csv('yearly_means_for_plot.csv', index_col=0)
print(yearly_means)

fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_means.index
control = yearly_means['Control']
treatment = yearly_means['Treatment']

ax.plot(years, control, 'b-o', linewidth=2, markersize=8, label='Control Group')
ax.plot(years, treatment, 'r-s', linewidth=2, markersize=8, label='DACA-Eligible (Treatment)')

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA Implementation (2012)')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.4, 0.75)

# Add note about 2012 exclusion
ax.annotate('Note: 2012 excluded\n(implementation year)',
            xy=(2012, 0.42), fontsize=9, ha='center', style='italic')

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', dpi=300, bbox_inches='tight')
print("  Saved figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Event study plot
# =============================================================================
print("Creating Figure 2: Event study...")

event_df = pd.read_csv('event_study_results.csv')

# Add reference year (2011) with zero effect
ref_row = pd.DataFrame({'year': [2011], 'coef': [0], 'se': [0], 'ci_low': [0], 'ci_high': [0]})
event_df = pd.concat([event_df, ref_row], ignore_index=True)
event_df = event_df.sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['year']
coefs = event_df['coef']
ci_low = event_df['ci_low']
ci_high = event_df['ci_high']

# Pre-treatment years
pre_mask = years <= 2011
post_mask = years >= 2011

# Plot confidence intervals
ax.fill_between(years[pre_mask], ci_low[pre_mask], ci_high[pre_mask], alpha=0.3, color='blue')
ax.fill_between(years[post_mask], ci_low[post_mask], ci_high[post_mask], alpha=0.3, color='red')

# Plot coefficients
ax.plot(years[pre_mask], coefs[pre_mask], 'b-o', linewidth=2, markersize=8, label='Pre-DACA')
ax.plot(years[post_mask], coefs[post_mask], 'r-o', linewidth=2, markersize=8, label='Post-DACA')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment\n(Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xlim(2005.5, 2016.5)

# Annotate DACA
ax.annotate('DACA\nImplementation', xy=(2012, ax.get_ylim()[1]*0.8),
            fontsize=10, ha='center')

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', dpi=300, bbox_inches='tight')
print("  Saved figure2_eventstudy.png/pdf")

# =============================================================================
# Figure 3: DiD visualization
# =============================================================================
print("Creating Figure 3: DiD visualization...")

# Calculate pre and post means
desc_stats = pd.read_csv('descriptive_stats.csv')
print(desc_stats)

# Manually calculate from yearly means
pre_years = [2006, 2007, 2008, 2009, 2010, 2011]
post_years = [2013, 2014, 2015, 2016]

pre_control = yearly_means.loc[yearly_means.index.isin(pre_years), 'Control'].mean()
post_control = yearly_means.loc[yearly_means.index.isin(post_years), 'Control'].mean()
pre_treat = yearly_means.loc[yearly_means.index.isin(pre_years), 'Treatment'].mean()
post_treat = yearly_means.loc[yearly_means.index.isin(post_years), 'Treatment'].mean()

# Calculate DiD
did_control = post_control - pre_control
did_treat = post_treat - pre_treat
did_effect = did_treat - did_control

fig, ax = plt.subplots(figsize=(8, 6))

# Bar positions
x = np.array([0, 1, 3, 4])
width = 0.7

# Pre and post values - plot separately to control alpha
ax.bar(x[0], pre_control, width, color='steelblue', alpha=0.5)
ax.bar(x[1], post_control, width, color='steelblue', alpha=1.0)
ax.bar(x[2], pre_treat, width, color='firebrick', alpha=0.5)
ax.bar(x[3], post_treat, width, color='firebrick', alpha=1.0)

# Add value labels
for i, (val, pos) in enumerate(zip([pre_control, post_control, pre_treat, post_treat], x)):
    ax.text(pos, val + 0.01, f'{val:.3f}', ha='center', fontsize=10)

# Add difference arrows
ax.annotate('', xy=(1, post_control), xytext=(0, pre_control),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('', xy=(4, post_treat), xytext=(3, pre_treat),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# Labels
ax.set_xticks([0.5, 3.5])
ax.set_xticklabels(['Control Group', 'DACA-Eligible (Treatment)'], fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment', fontsize=14)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='gray', alpha=0.5, label='Pre-DACA (2006-2011)'),
                   Patch(facecolor='gray', alpha=1.0, label='Post-DACA (2013-2016)')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add DiD calculation text
textstr = f'Control change: {did_control:.3f}\nTreatment change: {did_treat:.3f}\nDiD estimate: {did_effect:.3f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

ax.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', dpi=300, bbox_inches='tight')
print("  Saved figure3_did.png/pdf")

print("\nAll figures created successfully!")

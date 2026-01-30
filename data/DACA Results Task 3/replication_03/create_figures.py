"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# =============================================================================
# Figure 1: FT Employment Rates Over Time by Treatment Status
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

yearly_ft = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_ft.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

years = yearly_ft.index
control = yearly_ft['Control (Ages 31-35)']
treatment = yearly_ft['Treatment (Ages 26-30)']

ax1.plot(years, control, 'o-', color='#2166AC', linewidth=2, markersize=8, label='Control (Ages 31-35)')
ax1.plot(years, treatment, 's-', color='#B2182B', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')

# Add vertical line for DACA implementation
ax1.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax1.text(2012.1, 0.72, 'DACA\nImplementation', fontsize=10, color='gray', va='top')

# Shade pre and post periods
ax1.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-Treatment Period')
ax1.axvspan(2012.5, 2016.5, alpha=0.1, color='red', label='Post-Treatment Period')

ax1.set_xlabel('Year')
ax1.set_ylabel('Full-Time Employment Rate')
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax1.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax1.set_ylim(0.55, 0.75)
ax1.legend(loc='lower right')

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
print("Figure 1 saved: figure1_trends.png/pdf")

# =============================================================================
# Figure 2: Event Study Coefficients
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Event study coefficients from analysis (2011 is reference)
years_event = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coeffs = [-0.0451, -0.0338, -0.0694, 0, 0.0038, -0.0307, -0.0314, 0.0342]
ses = [0.0279, 0.0251, 0.0330, 0, 0.0345, 0.0205, 0.0356, 0.0329]

# Calculate 95% CIs
ci_lower = [c - 1.96*s for c, s in zip(coeffs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coeffs, ses)]

ax2.errorbar(years_event, coeffs, yerr=[np.array(coeffs)-np.array(ci_lower), np.array(ci_upper)-np.array(coeffs)],
             fmt='o', color='#2166AC', capsize=5, capthick=2, markersize=10, linewidth=2)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)

# Shade regions
ax2.axvspan(2007.5, 2011.5, alpha=0.1, color='blue')
ax2.axvspan(2012.5, 2016.5, alpha=0.1, color='red')

ax2.set_xlabel('Year')
ax2.set_ylabel('Coefficient (Relative to 2011)')
ax2.set_title('Event Study: Year-by-Year Treatment Effects')
ax2.set_xticks(years_event)
ax2.text(2012.1, 0.08, 'DACA', fontsize=10, color='gray', va='bottom')

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
print("Figure 2 saved: figure2_eventstudy.png/pdf")

# =============================================================================
# Figure 3: DID Illustration
# =============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Pre and post means
control_pre = 0.6697
control_post = 0.6449
treat_pre = 0.6263
treat_post = 0.6658

# Counterfactual
treat_cf = treat_pre + (control_post - control_pre)

# Plot actual trends
ax3.plot([0, 1], [control_pre, control_post], 'o-', color='#2166AC', linewidth=2.5, markersize=10, label='Control (Actual)')
ax3.plot([0, 1], [treat_pre, treat_post], 's-', color='#B2182B', linewidth=2.5, markersize=10, label='Treatment (Actual)')

# Plot counterfactual
ax3.plot([0, 1], [treat_pre, treat_cf], 's--', color='#B2182B', linewidth=2, markersize=10, alpha=0.5, label='Treatment (Counterfactual)')

# Arrow showing treatment effect
ax3.annotate('', xy=(1.05, treat_post), xytext=(1.05, treat_cf),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.1, (treat_post + treat_cf)/2, f'DID = {treat_post - treat_cf:.3f}', fontsize=12, color='green', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-Time Employment Rate')
ax3.set_title('Difference-in-Differences Illustration')
ax3.set_xlim(-0.2, 1.5)
ax3.set_ylim(0.55, 0.75)
ax3.legend(loc='upper left')

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
print("Figure 3 saved: figure3_did.png/pdf")

# =============================================================================
# Figure 4: Heterogeneity Analysis
# =============================================================================
fig4, ax4 = plt.subplots(figsize=(10, 6))

# Heterogeneity results
categories = ['Overall\n(Preferred)', 'Males', 'Females', 'HS or Less', 'Some College+']
effects = [0.0712, 0.0631, 0.0690, 0.0665, 0.0903]
ses_het = [0.0206, 0.0237, 0.0281, 0.0207, 0.0428]

ci_lower_het = [e - 1.96*s for e, s in zip(effects, ses_het)]
ci_upper_het = [e + 1.96*s for e, s in zip(effects, ses_het)]

colors = ['#2166AC', '#67A9CF', '#67A9CF', '#EF8A62', '#EF8A62']
x_pos = np.arange(len(categories))

ax4.bar(x_pos, effects, yerr=[np.array(effects)-np.array(ci_lower_het), np.array(ci_upper_het)-np.array(effects)],
        color=colors, capsize=5, alpha=0.8, edgecolor='black')

ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.axhline(y=0.0712, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax4.set_xticks(x_pos)
ax4.set_xticklabels(categories)
ax4.set_ylabel('DID Coefficient (Percentage Points)')
ax4.set_title('Heterogeneity in DACA Effects on Full-Time Employment')

# Add significance stars
for i, (e, p) in enumerate(zip(effects, [0.0006, 0.008, 0.014, 0.001, 0.035])):
    if p < 0.01:
        sig = '***'
    elif p < 0.05:
        sig = '**'
    elif p < 0.1:
        sig = '*'
    else:
        sig = ''
    ax4.text(i, e + 0.02, sig, ha='center', fontsize=14)

plt.tight_layout()
plt.savefig('figure4_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_heterogeneity.pdf', bbox_inches='tight')
print("Figure 4 saved: figure4_heterogeneity.png/pdf")

print("\nAll figures created successfully!")

"""
DACA Replication Study - Create Figures for Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load the analysis sample
print("Loading analysis sample...")
df = pd.read_csv('analysis_sample.csv')
print(f"Loaded {len(df):,} observations")

# Set up matplotlib
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# =============================================================================
# Figure 1: Full-time Employment Trends by Group
# =============================================================================
print("Creating Figure 1: Employment trends...")

fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate yearly means by treatment status
trends = df.groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack()
trends.columns = ['Control (Age 31-35)', 'Treatment (Age 26-30)']

# Plot
trends.plot(ax=ax1, marker='o', linewidth=2, markersize=8)

# Add vertical line at 2012 (DACA implementation)
ax1.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation (June 2012)')

# Shade the excluded year
ax1.axvspan(2011.5, 2012.5, alpha=0.2, color='gray', label='2012 excluded')

ax1.set_xlabel('Year')
ax1.set_ylabel('Full-time Employment Rate')
ax1.set_title('Full-time Employment Trends by Treatment Status')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(2005.5, 2016.5)
ax1.set_ylim(0.5, 0.75)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("Creating Figure 2: Event study...")

# Re-estimate event study model
df['treat_2006'] = df['treatment'] * (df['YEAR'] == 2006).astype(int)
df['treat_2007'] = df['treatment'] * (df['YEAR'] == 2007).astype(int)
df['treat_2008'] = df['treatment'] * (df['YEAR'] == 2008).astype(int)
df['treat_2009'] = df['treatment'] * (df['YEAR'] == 2009).astype(int)
df['treat_2010'] = df['treatment'] * (df['YEAR'] == 2010).astype(int)
df['treat_2013'] = df['treatment'] * (df['YEAR'] == 2013).astype(int)
df['treat_2014'] = df['treatment'] * (df['YEAR'] == 2014).astype(int)
df['treat_2015'] = df['treatment'] * (df['YEAR'] == 2015).astype(int)
df['treat_2016'] = df['treatment'] * (df['YEAR'] == 2016).astype(int)

event_study = smf.ols('fulltime ~ treatment + C(YEAR) + treat_2006 + treat_2007 + treat_2008 + treat_2009 + treat_2010 + treat_2013 + treat_2014 + treat_2015 + treat_2016 + female + married + educ_hs + educ_somecol + educ_ba_plus',
                      data=df).fit()

# Extract coefficients
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_vars = ['treat_2006', 'treat_2007', 'treat_2008', 'treat_2009', 'treat_2010',
              None,  # 2011 is reference
              'treat_2013', 'treat_2014', 'treat_2015', 'treat_2016']

coefs = []
ci_lows = []
ci_highs = []

for year, var in zip(years, event_vars):
    if var is None:  # Reference year
        coefs.append(0)
        ci_lows.append(0)
        ci_highs.append(0)
    else:
        coefs.append(event_study.params[var])
        ci_lows.append(event_study.conf_int().loc[var, 0])
        ci_highs.append(event_study.conf_int().loc[var, 1])

coefs = np.array(coefs)
ci_lows = np.array(ci_lows)
ci_highs = np.array(ci_highs)

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot point estimates with error bars
ax2.errorbar(years, coefs,
             yerr=[coefs - ci_lows, ci_highs - coefs],
             fmt='o', capsize=5, capthick=2, linewidth=2, markersize=8,
             color='navy', ecolor='navy')

# Connect points with line
ax2.plot(years, coefs, 'o-', color='navy', linewidth=1.5, alpha=0.7)

# Add horizontal line at zero
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at DACA implementation
ax2.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

# Shade pre and post periods
ax2.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-period')
ax2.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-period')

ax2.set_xlabel('Year')
ax2.set_ylabel('Coefficient (relative to 2011)')
ax2.set_title('Event Study: Treatment Effect on Full-time Employment\n(Reference Year: 2011)')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(2005, 2017)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: DiD visualization...")

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate pre and post means for each group
pre_treat = df[(df['treatment']==1) & (df['post']==0)]['fulltime'].mean()
post_treat = df[(df['treatment']==1) & (df['post']==1)]['fulltime'].mean()
pre_control = df[(df['treatment']==0) & (df['post']==0)]['fulltime'].mean()
post_control = df[(df['treatment']==0) & (df['post']==1)]['fulltime'].mean()

# Plot actual outcomes
ax3.plot([0, 1], [pre_treat, post_treat], 'o-', color='blue', linewidth=2.5,
         markersize=12, label='Treatment Group (Actual)')
ax3.plot([0, 1], [pre_control, post_control], 's-', color='orange', linewidth=2.5,
         markersize=12, label='Control Group (Actual)')

# Plot counterfactual for treatment group (parallel to control)
control_change = post_control - pre_control
counterfactual_post = pre_treat + control_change
ax3.plot([0, 1], [pre_treat, counterfactual_post], 'o--', color='lightblue',
         linewidth=2, markersize=8, label='Treatment (Counterfactual)')

# Annotate DiD
did_effect = (post_treat - pre_treat) - (post_control - pre_control)
ax3.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual_post),
             arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax3.text(1.12, (post_treat + counterfactual_post)/2, f'DiD = {did_effect:.3f}',
         fontsize=12, color='red', fontweight='bold', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-time Employment Rate')
ax3.set_title('Difference-in-Differences: Effect of DACA on Full-time Employment')
ax3.legend(loc='lower left')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-0.2, 1.4)
ax3.set_ylim(0.55, 0.70)

plt.tight_layout()
plt.savefig('figure3_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did.pdf', bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# =============================================================================
# Figure 4: Heterogeneity by Gender
# =============================================================================
print("Creating Figure 4: Heterogeneity by gender...")

fig4, ax4 = plt.subplots(figsize=(10, 6))

# Calculate trends by gender and treatment
trends_male = df[df['SEX']==1].groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack()
trends_female = df[df['SEX']==2].groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack()

ax4.plot(trends_male.index, trends_male[1], 'o-', color='blue', linewidth=2,
         label='Male Treatment')
ax4.plot(trends_male.index, trends_male[0], 's-', color='lightblue', linewidth=2,
         label='Male Control')
ax4.plot(trends_female.index, trends_female[1], 'o-', color='red', linewidth=2,
         label='Female Treatment')
ax4.plot(trends_female.index, trends_female[0], 's-', color='pink', linewidth=2,
         label='Female Control')

ax4.axvline(x=2012, color='gray', linestyle='--', linewidth=2, label='DACA')
ax4.axvspan(2011.5, 2012.5, alpha=0.2, color='gray')

ax4.set_xlabel('Year')
ax4.set_ylabel('Full-time Employment Rate')
ax4.set_title('Full-time Employment Trends by Gender and Treatment Status')
ax4.legend(loc='best', ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure4_heterogeneity_gender.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_heterogeneity_gender.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

# =============================================================================
# Figure 5: Sample Size by Year
# =============================================================================
print("Creating Figure 5: Sample size...")

fig5, ax5 = plt.subplots(figsize=(10, 6))

sample_sizes = df.groupby(['YEAR', 'treatment']).size().unstack()
sample_sizes.columns = ['Control', 'Treatment']

sample_sizes.plot(kind='bar', ax=ax5, width=0.8, color=['orange', 'blue'])

ax5.set_xlabel('Year')
ax5.set_ylabel('Number of Observations')
ax5.set_title('Sample Size by Year and Treatment Status')
ax5.legend(loc='upper right')
ax5.tick_params(axis='x', rotation=45)

# Add note about 2012 being excluded
ax5.text(0.5, 0.95, '(2012 excluded from analysis)', transform=ax5.transAxes,
         fontsize=10, ha='center', style='italic')

plt.tight_layout()
plt.savefig('figure5_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_sample_size.pdf', bbox_inches='tight')
plt.close()
print("Figure 5 saved.")

# =============================================================================
# Figure 6: Coefficient Comparison Across Models
# =============================================================================
print("Creating Figure 6: Model comparison...")

fig6, ax6 = plt.subplots(figsize=(10, 6))

# Model estimates (from analysis output)
models = ['Basic DiD\n(No Controls)', 'Year FE', 'Year FE +\nControls',
          'Year + State FE\n+ Controls', 'Weighted\nRegression', 'Clustered SE']
estimates = [0.0551, 0.0554, 0.0487, 0.0477, 0.0480, 0.0487]
std_errors = [0.0098, 0.0098, 0.0091, 0.0091, 0.0089, 0.0088]

ci_lows = [e - 1.96*se for e, se in zip(estimates, std_errors)]
ci_highs = [e + 1.96*se for e, se in zip(estimates, std_errors)]

x_pos = np.arange(len(models))
ax6.errorbar(x_pos, estimates, yerr=[np.array(estimates)-np.array(ci_lows),
                                       np.array(ci_highs)-np.array(estimates)],
             fmt='o', capsize=5, capthick=2, markersize=10, color='navy', ecolor='navy')

ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax6.axhline(y=estimates[2], color='red', linestyle='--', linewidth=1, alpha=0.5,
            label=f'Preferred estimate: {estimates[2]:.4f}')

ax6.set_xticks(x_pos)
ax6.set_xticklabels(models, fontsize=10)
ax6.set_ylabel('DiD Coefficient')
ax6.set_title('Comparison of DiD Estimates Across Model Specifications')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("Figure 6 saved.")

print("\nAll figures created successfully!")

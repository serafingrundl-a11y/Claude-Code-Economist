"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

output_path = r'C:\Users\seraf\DACA Results Task 3\replication_78'

# Load data
df = pd.read_csv(f'{output_path}/data/prepared_data_numeric_version.csv')

# Prepare derived variables
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# ============================================================================
# FIGURE 1: Full-time employment trends over time
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate weighted means by year and group
ft_by_year = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: np.average(x['FT'], weights=x['PERWT'])
).unstack()
ft_by_year.columns = ['Control (31-35)', 'Treatment (26-30)']

# Plot
years = ft_by_year.index.values
ax1.plot(years, ft_by_year['Treatment (26-30)'] * 100, 'o-', color='blue',
         linewidth=2, markersize=8, label='Treatment (26-30)')
ax1.plot(years, ft_by_year['Control (31-35)'] * 100, 's--', color='red',
         linewidth=2, markersize=8, label='Control (31-35)')

# Add vertical line for DACA implementation
ax1.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation (2012)')

ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax1.set_title('Full-Time Employment Rates by Treatment Status Over Time', fontsize=14)
ax1.legend(loc='lower right', fontsize=10)
ax1.set_xticks(years)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(55, 80)

plt.tight_layout()
plt.savefig(f'{output_path}/figure1_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: figure1_trends.png")

# ============================================================================
# FIGURE 2: Event study coefficients
# ============================================================================
import statsmodels.formula.api as smf

# Create year dummies and interactions
df['YEAR_2008'] = (df['YEAR'] == 2008).astype(int)
df['YEAR_2009'] = (df['YEAR'] == 2009).astype(int)
df['YEAR_2010'] = (df['YEAR'] == 2010).astype(int)
df['YEAR_2013'] = (df['YEAR'] == 2013).astype(int)
df['YEAR_2014'] = (df['YEAR'] == 2014).astype(int)
df['YEAR_2015'] = (df['YEAR'] == 2015).astype(int)
df['YEAR_2016'] = (df['YEAR'] == 2016).astype(int)

df['ELIG_2008'] = df['ELIGIBLE'] * df['YEAR_2008']
df['ELIG_2009'] = df['ELIGIBLE'] * df['YEAR_2009']
df['ELIG_2010'] = df['ELIGIBLE'] * df['YEAR_2010']
df['ELIG_2013'] = df['ELIGIBLE'] * df['YEAR_2013']
df['ELIG_2014'] = df['ELIGIBLE'] * df['YEAR_2014']
df['ELIG_2015'] = df['ELIGIBLE'] * df['YEAR_2015']
df['ELIG_2016'] = df['ELIGIBLE'] * df['YEAR_2016']

event_formula = 'FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_2008 + ELIG_2009 + ELIG_2010 + ELIG_2013 + ELIG_2014 + ELIG_2015 + ELIG_2016'
model_event = smf.wls(event_formula, data=df, weights=df['PERWT']).fit(cov_type='HC1')

# Extract coefficients
years_event = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [model_event.params.get(f'ELIG_{y}', 0) for y in years_event]
coefs[3] = 0  # 2011 is reference
ses = [model_event.bse.get(f'ELIG_{y}', 0) for y in years_event]
ses[3] = 0  # 2011 is reference

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.errorbar(years_event, [c*100 for c in coefs], yerr=[1.96*s*100 for s in ses],
             fmt='o', capsize=5, capthick=2, color='blue', markersize=8, linewidth=2)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Coefficient (Percentage Points)', fontsize=12)
ax2.set_title('Event Study: Treatment Effect on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)
ax2.set_xticks(years_event)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower right', fontsize=10)

# Add shading for pre and post periods
ax2.axvspan(2007.5, 2011.5, alpha=0.1, color='red', label='Pre-DACA')
ax2.axvspan(2012.5, 2016.5, alpha=0.1, color='blue', label='Post-DACA')

plt.tight_layout()
plt.savefig(f'{output_path}/figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: figure2_eventstudy.png")

# ============================================================================
# FIGURE 3: Difference-in-Differences Visualization
# ============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Calculate pre and post means
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean() * 100
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean() * 100
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean() * 100
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean() * 100

# Plot actual lines
ax3.plot([0, 1], [pre_treat, post_treat], 'o-', color='blue', linewidth=3,
         markersize=12, label='Treatment (26-30)')
ax3.plot([0, 1], [pre_control, post_control], 's-', color='red', linewidth=3,
         markersize=12, label='Control (31-35)')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax3.plot([0, 1], [pre_treat, counterfactual], 'o--', color='blue', linewidth=2,
         markersize=8, alpha=0.5, label='Treatment (Counterfactual)')

# Annotate the DiD
ax3.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.1, (post_treat + counterfactual)/2, f'DiD Effect:\n{post_treat - counterfactual:.2f} pp',
         fontsize=11, color='green', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax3.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
ax3.set_title('Difference-in-Differences: Effect of DACA on Full-Time Employment', fontsize=14)
ax3.legend(loc='lower left', fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xlim(-0.2, 1.4)
ax3.set_ylim(55, 75)

plt.tight_layout()
plt.savefig(f'{output_path}/figure3_did.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: figure3_did.png")

# ============================================================================
# FIGURE 4: Gender-specific effects
# ============================================================================
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

for ax, gender, title in [(ax4a, 0, 'Male'), (ax4b, 1, 'Female')]:
    df_g = df[df['FEMALE'] == gender]
    ft_gender = df_g.groupby(['YEAR', 'ELIGIBLE']).apply(
        lambda x: np.average(x['FT'], weights=x['PERWT'])
    ).unstack()
    ft_gender.columns = ['Control', 'Treatment']

    years = ft_gender.index.values
    ax.plot(years, ft_gender['Treatment'] * 100, 'o-', color='blue',
            linewidth=2, markersize=8, label='Treatment (26-30)')
    ax.plot(years, ft_gender['Control'] * 100, 's--', color='red',
            linewidth=2, markersize=8, label='Control (31-35)')
    ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Full-Time Employment Rate (%)', fontsize=12)
    ax.set_title(f'{title}', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xticks(years)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(45, 90)

plt.suptitle('Full-Time Employment Trends by Gender', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_path}/figure4_gender.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved: figure4_gender.png")

# ============================================================================
# FIGURE 5: Model comparison forest plot
# ============================================================================
import warnings
warnings.filterwarnings('ignore')

# Run all models
model1 = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df).fit()
model2 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit()
model3 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER', data=df, weights=df['PERWT']).fit(cov_type='HC1')

df['MARRIED'] = (df['MARST'] == 1).astype(int)
model4 = smf.wls('FT ~ ELIGIBLE + AFTER + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
model5 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')
model6 = smf.wls('FT ~ ELIGIBLE + ELIGIBLE_AFTER + FEMALE + MARRIED + FAMSIZE + NCHILD + C(YEAR) + C(STATEFIP)',
                 data=df, weights=df['PERWT']).fit(cov_type='HC1')

models = [
    ('Unweighted OLS', model1),
    ('Weighted OLS', model2),
    ('Weighted OLS + Robust SE', model3),
    ('+ Demographics', model4),
    ('+ Year FE', model5),
    ('+ State & Year FE', model6)
]

fig5, ax5 = plt.subplots(figsize=(10, 8))

y_positions = range(len(models))
coefs = [m[1].params['ELIGIBLE_AFTER'] * 100 for m in models]
ses = [m[1].bse['ELIGIBLE_AFTER'] * 100 for m in models]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax5.errorbar(coefs, y_positions, xerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
             fmt='o', capsize=6, capthick=2, color='blue', markersize=10, linewidth=2)
ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1)
ax5.set_yticks(y_positions)
ax5.set_yticklabels([m[0] for m in models], fontsize=11)
ax5.set_xlabel('DiD Effect (Percentage Points)', fontsize=12)
ax5.set_title('Effect of DACA Eligibility on Full-Time Employment\nAcross Model Specifications', fontsize=14)
ax5.grid(True, alpha=0.3, axis='x')

# Add coefficient labels
for i, (c, lo, hi) in enumerate(zip(coefs, ci_low, ci_high)):
    ax5.text(hi + 0.5, i, f'{c:.2f} [{lo:.2f}, {hi:.2f}]', va='center', fontsize=9)

ax5.set_xlim(-2, 16)
plt.tight_layout()
plt.savefig(f'{output_path}/figure5_forest.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved: figure5_forest.png")

print("\nAll figures created successfully!")

"""
DACA Replication Study - Create Figures for Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('data/prepared_data_numeric_version.csv')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (8, 5)

#############################################################################
# FIGURE 1: Parallel Trends Plot
#############################################################################
print("Creating Figure 1: Parallel Trends Plot...")

years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_means = []
ctrl_means = []

for year in years:
    df_year = df[df['YEAR'] == year]
    treat_ft = np.average(df_year[df_year['ELIGIBLE'] == 1]['FT'],
                          weights=df_year[df_year['ELIGIBLE'] == 1]['PERWT'])
    ctrl_ft = np.average(df_year[df_year['ELIGIBLE'] == 0]['FT'],
                         weights=df_year[df_year['ELIGIBLE'] == 0]['PERWT'])
    treat_means.append(treat_ft)
    ctrl_means.append(ctrl_ft)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years, treat_means, 'b-o', label='Treatment (Ages 26-30 in 2012)', linewidth=2, markersize=8)
ax.plot(years, ctrl_means, 'r--s', label='Control (Ages 31-35 in 2012)', linewidth=2, markersize=8)

# Add vertical line for treatment
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, label='DACA Implementation')

# Add shading for pre/post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA Period')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Full-Time Employment Trends: Treatment vs. Control Groups', fontsize=14)
ax.set_xticks(years)
ax.set_ylim(0.55, 0.80)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure1_parallel_trends.png")

#############################################################################
# FIGURE 2: Event Study Coefficients
#############################################################################
print("Creating Figure 2: Event Study Plot...")

import statsmodels.formula.api as smf

# Create necessary variables
df['ELIGIBLE_AFTER'] = df['ELIGIBLE'] * df['AFTER']
df['FEMALE'] = (df['SEX'] == 2).astype(int)
df['MARRIED'] = df['MARST'].isin([1, 2]).astype(int)
df['EDUC_SOMECOLL'] = (df['EDUC'] >= 7).fillna(0).astype(int)
df['EDUC_BA'] = (df['EDUC'] >= 10).fillna(0).astype(int)
df['HAS_CHILDREN'] = (df['NCHILD'] > 0).astype(int)

# Create year interactions
for year in years:
    if year != 2008:
        df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
        df[f'ELIGIBLE_YEAR_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Run event study regression
formula_es = 'FT ~ ELIGIBLE + YEAR_2009 + YEAR_2010 + YEAR_2011 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIGIBLE_YEAR_2009 + ELIGIBLE_YEAR_2010 + ELIGIBLE_YEAR_2011 + ELIGIBLE_YEAR_2013 + ELIGIBLE_YEAR_2014 + ELIGIBLE_YEAR_2015 + ELIGIBLE_YEAR_2016 + FEMALE + MARRIED + EDUC_SOMECOLL + EDUC_BA + HAS_CHILDREN + AGE'
model_es = smf.wls(formula_es, data=df, weights=df['PERWT']).fit(cov_type='cluster', cov_kwds={'groups': df['STATEFIP']})

# Extract coefficients and CIs
event_years = [2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs = [0]  # 2008 is reference (0)
ci_low = [0]
ci_high = [0]
plot_years = [2008]

for year in event_years:
    coef = model_es.params[f'ELIGIBLE_YEAR_{year}']
    se = model_es.bse[f'ELIGIBLE_YEAR_{year}']
    coefs.append(coef)
    ci_low.append(coef - 1.96 * se)
    ci_high.append(coef + 1.96 * se)
    plot_years.append(year)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot confidence intervals as error bars
yerr = [np.array(coefs) - np.array(ci_low), np.array(ci_high) - np.array(coefs)]
ax.errorbar(plot_years, coefs, yerr=yerr, fmt='ko-', capsize=4, capthick=2, linewidth=2, markersize=8)

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Add vertical line for treatment
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

# Add shading
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='blue')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('DiD Coefficient (Relative to 2008)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.set_xticks(plot_years)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure2_event_study.png")

#############################################################################
# FIGURE 3: DiD Visualization (2x2)
#############################################################################
print("Creating Figure 3: DiD 2x2 Visualization...")

# Calculate weighted means for pre/post
pre_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                       weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT'])
post_treat = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                        weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])
pre_ctrl = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                      weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
post_ctrl = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                       weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])

fig, ax = plt.subplots(figsize=(8, 6))

# Plot actual trends
x_vals = [0, 1]
ax.plot(x_vals, [pre_treat, post_treat], 'b-o', label='Treatment (Actual)', linewidth=3, markersize=12)
ax.plot(x_vals, [pre_ctrl, post_ctrl], 'r--s', label='Control (Actual)', linewidth=3, markersize=12)

# Plot counterfactual
counterfactual = pre_treat + (post_ctrl - pre_ctrl)
ax.plot([0, 1], [pre_treat, counterfactual], 'b:', label='Treatment (Counterfactual)', linewidth=2, alpha=0.7)

# Add DiD arrow
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=3))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD\n= {post_treat - counterfactual:.3f}',
        fontsize=12, color='green', fontweight='bold', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate (Weighted)', fontsize=12)
ax.set_title('Difference-in-Differences Estimate of DACA Effect', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.58, 0.74)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure3_did_visualization.png")

#############################################################################
# FIGURE 4: Sample Distribution by Age
#############################################################################
print("Creating Figure 4: Age Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

# Age distribution by group
ages_treat = df[df['ELIGIBLE']==1]['AGE']
ages_ctrl = df[df['ELIGIBLE']==0]['AGE']

ax.hist(ages_treat, bins=range(20, 42), alpha=0.5, label='Treatment (ELIGIBLE=1)', color='blue', edgecolor='black')
ax.hist(ages_ctrl, bins=range(20, 42), alpha=0.5, label='Control (ELIGIBLE=0)', color='red', edgecolor='black')

ax.axvline(x=25.5, color='blue', linestyle='--', linewidth=2, label='Treatment age range (26-30 in 2012)')
ax.axvline(x=30.5, color='blue', linestyle='--', linewidth=2)
ax.axvline(x=30.5, color='red', linestyle=':', linewidth=2, label='Control age range (31-35 in 2012)')
ax.axvline(x=35.5, color='red', linestyle=':', linewidth=2)

ax.set_xlabel('Age (in survey year)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Age Distribution by Treatment Status', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_age_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure4_age_distribution.png")

#############################################################################
# FIGURE 5: Heterogeneity Results
#############################################################################
print("Creating Figure 5: Heterogeneity Analysis...")

# Store heterogeneity results (from analysis_robustness.py output)
categories = ['Overall', 'Males', 'Females', 'Married', 'Not Married']
estimates = [0.0621, 0.0600, 0.0535, 0.0095, 0.1005]
ses = [0.0212, 0.0196, 0.0280, 0.0127, 0.0402]

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(categories))
colors = ['darkblue', 'steelblue', 'lightsteelblue', 'coral', 'orangered']

# Plot bars
bars = ax.barh(y_pos, estimates, xerr=[1.96*s for s in ses], align='center',
               color=colors, capsize=5, alpha=0.8)

# Add vertical line at zero
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=11)
ax.set_xlabel('DiD Estimate (Effect on FT Employment Rate)', fontsize=12)
ax.set_title('Heterogeneity in DACA Effect by Subgroup', fontsize=14)
ax.set_xlim(-0.05, 0.20)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (est, se) in enumerate(zip(estimates, ses)):
    ax.text(est + 1.96*se + 0.01, i, f'{est:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure5_heterogeneity.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure5_heterogeneity.png")

#############################################################################
# FIGURE 6: Coefficient Comparison Across Models
#############################################################################
print("Creating Figure 6: Model Comparison...")

models = ['Basic OLS', 'Weighted', 'Covariates', 'Year FE', 'Full', 'State FE']
estimates_all = [0.0643, 0.0748, 0.0621, 0.0721, 0.0593, 0.0588]
ses_all = [0.0153, 0.0152, 0.0142, 0.0151, 0.0142, 0.0142]

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = np.arange(len(models))

# Plot error bars
ax.errorbar(x_pos, estimates_all, yerr=[1.96*s for s in ses_all], fmt='ko', capsize=6,
            capthick=2, linewidth=2, markersize=10)

# Add horizontal line at zero
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# Add horizontal band for main result
ax.axhspan(0.0621 - 1.96*0.0212, 0.0621 + 1.96*0.0212, alpha=0.2, color='blue',
           label='Preferred Estimate 95% CI')

ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=10, rotation=15)
ax.set_ylabel('DiD Estimate', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.12)

plt.tight_layout()
plt.savefig('figure6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("  Saved: figure6_model_comparison.png")

print("\nAll figures created successfully!")

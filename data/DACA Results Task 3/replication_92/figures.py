"""
Generate figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.figsize'] = (10, 6)

# Load data
data_path = "C:/Users/seraf/DACA Results Task 3/replication_92/data/prepared_data_numeric_version.csv"
df = pd.read_csv(data_path)

# Create interaction term
df['ELIGIBLE_x_AFTER'] = df['ELIGIBLE'] * df['AFTER']

# Figure 1: Year-by-year full-time employment rates
print("Creating Figure 1: Year-by-year FT rates...")
fig1, ax1 = plt.subplots(figsize=(10, 6))

years = sorted(df['YEAR'].unique())
treat_means = []
control_means = []
treat_ses = []
control_ses = []

for year in years:
    treat_data = df[(df['YEAR']==year) & (df['ELIGIBLE']==1)]['FT']
    control_data = df[(df['YEAR']==year) & (df['ELIGIBLE']==0)]['FT']
    treat_means.append(treat_data.mean())
    control_means.append(control_data.mean())
    treat_ses.append(treat_data.std() / np.sqrt(len(treat_data)))
    control_ses.append(control_data.std() / np.sqrt(len(control_data)))

ax1.errorbar(years, treat_means, yerr=[1.96*se for se in treat_ses],
             marker='o', markersize=8, capsize=4, linewidth=2, label='Treatment (Eligible, ages 26-30)')
ax1.errorbar(years, control_means, yerr=[1.96*se for se in control_ses],
             marker='s', markersize=8, capsize=4, linewidth=2, label='Control (Ineligible, ages 31-35)')

# Add vertical line for DACA implementation
ax1.axvline(x=2012, color='red', linestyle='--', linewidth=2, label='DACA Implementation (June 2012)')
ax1.axvspan(2012.5, 2017, alpha=0.1, color='green', label='Post-treatment period')

ax1.set_xlabel('Year')
ax1.set_ylabel('Full-Time Employment Rate')
ax1.set_title('Full-Time Employment Rates by DACA Eligibility Status Over Time')
ax1.legend(loc='lower right')
ax1.set_xticks(years)
ax1.set_xlim(2007.5, 2016.5)
ax1.set_ylim(0.55, 0.8)

plt.tight_layout()
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure1_trends.png', dpi=150, bbox_inches='tight')
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure1_trends.pdf', bbox_inches='tight')
print("  Saved figure1_trends.png and .pdf")

# Figure 2: Event Study Coefficients
print("Creating Figure 2: Event study coefficients...")

# Create year-specific interactions
for year in years:
    df[f'YEAR_{year}'] = (df['YEAR'] == year).astype(int)
    if year != 2011:  # Reference year
        df[f'ELIG_x_{year}'] = df['ELIGIBLE'] * df[f'YEAR_{year}']

# Run event study regression
event_formula = 'FT ~ ELIGIBLE + YEAR_2008 + YEAR_2009 + YEAR_2010 + YEAR_2013 + YEAR_2014 + YEAR_2015 + YEAR_2016 + ELIG_x_2008 + ELIG_x_2009 + ELIG_x_2010 + ELIG_x_2013 + ELIG_x_2014 + ELIG_x_2015 + ELIG_x_2016'
event_model = smf.ols(event_formula, data=df).fit()

# Extract coefficients
event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []

for year in event_years:
    if year == 2011:
        event_coefs.append(0)  # Reference year
        event_ses.append(0)
    else:
        coef = event_model.params[f'ELIG_x_{year}']
        se = event_model.bse[f'ELIG_x_{year}']
        event_coefs.append(coef)
        event_ses.append(se)

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot point estimates with confidence intervals
ax2.errorbar(event_years, event_coefs, yerr=[1.96*se for se in event_ses],
             fmt='o', markersize=10, capsize=5, linewidth=2, color='navy')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=2011.5, color='red', linestyle='--', linewidth=2, label='DACA Implementation')

# Shade post-treatment period
ax2.axvspan(2011.5, 2016.5, alpha=0.1, color='green')

ax2.set_xlabel('Year')
ax2.set_ylabel('DiD Coefficient (relative to 2011)')
ax2.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax2.set_xticks(event_years)
ax2.legend(loc='lower right')
ax2.set_xlim(2007.5, 2016.5)

# Add annotation
ax2.annotate('Reference\nYear', xy=(2011, 0), xytext=(2011, 0.08),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure2_event_study.png', dpi=150, bbox_inches='tight')
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure2_event_study.pdf', bbox_inches='tight')
print("  Saved figure2_event_study.png and .pdf")

# Figure 3: 2x2 DiD illustration
print("Creating Figure 3: DiD illustration...")

fig3, ax3 = plt.subplots(figsize=(9, 6))

# Calculate group means
pre_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'].mean()
post_treat = df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'].mean()
pre_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'].mean()
post_control = df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'].mean()

# Plot lines
ax3.plot([0, 1], [pre_treat, post_treat], 'o-', markersize=12, linewidth=3,
         label=f'Treatment (Eligible)', color='blue')
ax3.plot([0, 1], [pre_control, post_control], 's-', markersize=12, linewidth=3,
         label=f'Control (Ineligible)', color='orange')

# Counterfactual line
counterfactual = pre_treat + (post_control - pre_control)
ax3.plot([0, 1], [pre_treat, counterfactual], 'o--', markersize=8, linewidth=2,
         label='Counterfactual', color='lightblue', alpha=0.7)

# DiD arrow
ax3.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax3.text(1.1, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
        fontsize=11, color='green', va='center')

ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax3.set_ylabel('Full-Time Employment Rate')
ax3.set_title('Difference-in-Differences Design')
ax3.legend(loc='lower left')
ax3.set_xlim(-0.2, 1.4)
ax3.set_ylim(0.55, 0.75)

# Add text annotations for values
ax3.annotate(f'{pre_treat:.3f}', xy=(0, pre_treat), xytext=(-0.12, pre_treat),
            fontsize=10, ha='right', va='center')
ax3.annotate(f'{post_treat:.3f}', xy=(1, post_treat), xytext=(0.88, post_treat+0.015),
            fontsize=10, ha='right', va='center')
ax3.annotate(f'{pre_control:.3f}', xy=(0, pre_control), xytext=(-0.12, pre_control),
            fontsize=10, ha='right', va='center')
ax3.annotate(f'{post_control:.3f}', xy=(1, post_control), xytext=(0.88, post_control-0.015),
            fontsize=10, ha='right', va='center')

plt.tight_layout()
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure3_did_illustration.png', dpi=150, bbox_inches='tight')
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure3_did_illustration.pdf', bbox_inches='tight')
print("  Saved figure3_did_illustration.png and .pdf")

# Figure 4: Subgroup analysis
print("Creating Figure 4: Subgroup analysis...")

fig4, ax4 = plt.subplots(figsize=(10, 6))

subgroups = []
coefs = []
ses = []
labels = []

# Overall
model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=df).fit()
subgroups.append(0)
coefs.append(model.params['ELIGIBLE_x_AFTER'])
ses.append(model.bse['ELIGIBLE_x_AFTER'])
labels.append('Overall')

# By sex
for i, (sex, name) in enumerate([(1, 'Male'), (2, 'Female')]):
    subset = df[df['SEX'] == sex]
    model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=subset).fit()
    subgroups.append(i + 1)
    coefs.append(model.params['ELIGIBLE_x_AFTER'])
    ses.append(model.bse['ELIGIBLE_x_AFTER'])
    labels.append(name)

# By education (selected groups)
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'BA+']
for i, educ in enumerate(educ_order):
    subset = df[df['EDUC_RECODE'] == educ]
    if len(subset) > 100:
        model = smf.ols('FT ~ ELIGIBLE + AFTER + ELIGIBLE_x_AFTER', data=subset).fit()
        subgroups.append(i + 3)
        coefs.append(model.params['ELIGIBLE_x_AFTER'])
        ses.append(model.bse['ELIGIBLE_x_AFTER'])
        labels.append(educ)

ax4.errorbar(coefs, range(len(coefs)), xerr=[1.96*se for se in ses],
            fmt='o', markersize=10, capsize=5, linewidth=2, color='navy')
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1)

ax4.set_yticks(range(len(coefs)))
ax4.set_yticklabels(labels)
ax4.set_xlabel('DiD Coefficient (Effect on P(Full-Time Employment))')
ax4.set_title('Heterogeneous Effects of DACA Eligibility by Subgroup')
ax4.invert_yaxis()

# Add vertical line at overall estimate
ax4.axvline(x=coefs[0], color='red', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure4_subgroups.png', dpi=150, bbox_inches='tight')
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure4_subgroups.pdf', bbox_inches='tight')
print("  Saved figure4_subgroups.png and .pdf")

# Figure 5: Distribution of FT by group
print("Creating Figure 5: Sample composition...")

fig5, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: Year distribution
ax = axes[0, 0]
year_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()
year_counts.plot(kind='bar', ax=ax, color=['orange', 'blue'])
ax.set_xlabel('Year')
ax.set_ylabel('Number of Observations')
ax.set_title('A. Sample Size by Year and Eligibility')
ax.legend(['Control (Ineligible)', 'Treatment (Eligible)'])
ax.set_xticklabels([str(y) for y in year_counts.index], rotation=45)

# Panel B: Age distribution
ax = axes[0, 1]
df[df['ELIGIBLE']==1]['AGE'].hist(ax=ax, bins=20, alpha=0.7, label='Treatment', color='blue')
df[df['ELIGIBLE']==0]['AGE'].hist(ax=ax, bins=20, alpha=0.7, label='Control', color='orange')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.set_title('B. Age Distribution by Eligibility')
ax.legend()

# Panel C: Education distribution
ax = axes[1, 0]
educ_treat = df[df['ELIGIBLE']==1]['EDUC_RECODE'].value_counts(normalize=True)
educ_control = df[df['ELIGIBLE']==0]['EDUC_RECODE'].value_counts(normalize=True)
educ_order = ['Less than High School', 'High School Degree', 'Some College', 'Two-Year Degree', 'BA+']
x = np.arange(len(educ_order))
width = 0.35
treat_vals = [educ_treat.get(e, 0) for e in educ_order]
control_vals = [educ_control.get(e, 0) for e in educ_order]
ax.bar(x - width/2, control_vals, width, label='Control', color='orange')
ax.bar(x + width/2, treat_vals, width, label='Treatment', color='blue')
ax.set_xticks(x)
ax.set_xticklabels(['< HS', 'HS', 'Some Coll', '2-Yr', 'BA+'], rotation=45)
ax.set_ylabel('Proportion')
ax.set_title('C. Education Distribution by Eligibility')
ax.legend()

# Panel D: Sex distribution
ax = axes[1, 1]
sex_treat = df[df['ELIGIBLE']==1]['SEX'].value_counts(normalize=True)
sex_control = df[df['ELIGIBLE']==0]['SEX'].value_counts(normalize=True)
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [sex_control.get(1, 0), sex_control.get(2, 0)], width, label='Control', color='orange')
ax.bar(x + width/2, [sex_treat.get(1, 0), sex_treat.get(2, 0)], width, label='Treatment', color='blue')
ax.set_xticks(x)
ax.set_xticklabels(['Male', 'Female'])
ax.set_ylabel('Proportion')
ax.set_title('D. Sex Distribution by Eligibility')
ax.legend()

plt.tight_layout()
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure5_sample.png', dpi=150, bbox_inches='tight')
plt.savefig('C:/Users/seraf/DACA Results Task 3/replication_92/figure5_sample.pdf', bbox_inches='tight')
print("  Saved figure5_sample.png and .pdf")

print("\nAll figures created successfully!")

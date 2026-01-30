"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_labelled_version.csv', low_memory=False)

# =============================================================================
# Figure 1: Parallel Trends
# =============================================================================
print("Creating Figure 1: Parallel Trends...")

yearly_ft = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()
yearly_ft.columns = ['Control (Ages 31-35)', 'Treatment (Ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_ft.index
ax.plot(years, yearly_ft['Control (Ages 31-35)'], 'o-', color='blue',
        label='Control (Ages 31-35 in June 2012)', linewidth=2, markersize=8)
ax.plot(years, yearly_ft['Treatment (Ages 26-30)'], 's-', color='red',
        label='Treatment (Ages 26-30 in June 2012)', linewidth=2, markersize=8)

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2012.1, ax.get_ylim()[1]*0.95, 'DACA\nImplementation',
        fontsize=10, verticalalignment='top')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim([0.55, 0.75])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_parallel_trends.png")

# =============================================================================
# Figure 2: Event Study
# =============================================================================
print("Creating Figure 2: Event Study...")

# Event study coefficients (from analysis)
event_coefs = {
    2008: -0.0527,
    2009: -0.0408,
    2010: -0.0593,
    2011: 0.0,  # reference
    2013: 0.0219,
    2014: -0.0139,
    2015: 0.0247,
    2016: 0.0414
}

event_se = {
    2008: 0.0270,
    2009: 0.0278,
    2010: 0.0276,
    2011: 0.0,
    2013: 0.0283,
    2014: 0.0286,
    2015: 0.0290,
    2016: 0.0293
}

years = list(event_coefs.keys())
coefs = list(event_coefs.values())
ses = list(event_se.values())

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate 95% CI
ci_lower = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_upper = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.errorbar(years, coefs, yerr=[np.array(coefs)-np.array(ci_lower),
            np.array(ci_upper)-np.array(coefs)], fmt='o', color='darkblue',
            capsize=4, capthick=2, markersize=8, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=2011.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2011.6, ax.get_ylim()[1]*0.9, 'DACA', fontsize=10, color='red')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment\n(Reference Year: 2011)', fontsize=14)
ax.set_xticks(years)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_event_study.png")

# =============================================================================
# Figure 3: DiD Visualization
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

# Calculate pre and post means
pre_control = df[(df['AFTER']==0) & (df['ELIGIBLE']==0)]['FT'].mean()
pre_treat = df[(df['AFTER']==0) & (df['ELIGIBLE']==1)]['FT'].mean()
post_control = df[(df['AFTER']==1) & (df['ELIGIBLE']==0)]['FT'].mean()
post_treat = df[(df['AFTER']==1) & (df['ELIGIBLE']==1)]['FT'].mean()

fig, ax = plt.subplots(figsize=(8, 6))

# Plot actual lines
ax.plot([0, 1], [pre_control, post_control], 'o-', color='blue',
        linewidth=2, markersize=10, label='Control (Ages 31-35)')
ax.plot([0, 1], [pre_treat, post_treat], 's-', color='red',
        linewidth=2, markersize=10, label='Treatment (Ages 26-30)')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'r--', linewidth=2, alpha=0.5,
        label='Treatment (Counterfactual)')

# Mark DiD
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD = {(post_treat-counterfactual)*100:.1f}pp',
        fontsize=11, color='green', verticalalignment='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Design', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([-0.2, 1.4])
ax.set_ylim([0.58, 0.72])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figure3_did_design.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_did_design.png")

# =============================================================================
# Figure 4: Distribution of Key Covariates
# =============================================================================
print("Creating Figure 4: Covariate Distributions...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution by group
ax = axes[0, 0]
df[df['ELIGIBLE']==0]['AGE_IN_JUNE_2012'].hist(ax=ax, bins=20, alpha=0.5,
                                                label='Control', color='blue')
df[df['ELIGIBLE']==1]['AGE_IN_JUNE_2012'].hist(ax=ax, bins=20, alpha=0.5,
                                                label='Treatment', color='red')
ax.set_xlabel('Age in June 2012')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Age in June 2012')
ax.legend()

# Education distribution
ax = axes[0, 1]
educ_order = ['Less than High School', 'High School Degree', 'Some College',
              'Two-Year Degree', 'BA+']
educ_control = df[df['ELIGIBLE']==0]['EDUC_RECODE'].value_counts()
educ_treat = df[df['ELIGIBLE']==1]['EDUC_RECODE'].value_counts()

x = np.arange(len(educ_order))
width = 0.35
bars1 = ax.bar(x - width/2, [educ_control.get(e, 0)/len(df[df['ELIGIBLE']==0]) for e in educ_order],
               width, label='Control', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, [educ_treat.get(e, 0)/len(df[df['ELIGIBLE']==1]) for e in educ_order],
               width, label='Treatment', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(['<HS', 'HS', 'Some Col', '2-Yr', 'BA+'], fontsize=9)
ax.set_ylabel('Proportion')
ax.set_title('Education Distribution by Group')
ax.legend()

# Gender composition
ax = axes[1, 0]
df['FEMALE_label'] = df['SEX'].apply(lambda x: 'Female' if x == 'Female' else 'Male')
gender_control = df[df['ELIGIBLE']==0]['FEMALE_label'].value_counts(normalize=True)
gender_treat = df[df['ELIGIBLE']==1]['FEMALE_label'].value_counts(normalize=True)

x = np.arange(2)
width = 0.35
ax.bar(x - width/2, [gender_control.get('Male', 0), gender_control.get('Female', 0)],
       width, label='Control', color='blue', alpha=0.7)
ax.bar(x + width/2, [gender_treat.get('Male', 0), gender_treat.get('Female', 0)],
       width, label='Treatment', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(['Male', 'Female'])
ax.set_ylabel('Proportion')
ax.set_title('Gender Distribution by Group')
ax.legend()

# FT by group and period
ax = axes[1, 1]
groups = ['Control\nPre', 'Control\nPost', 'Treatment\nPre', 'Treatment\nPost']
ft_rates = [pre_control, post_control, pre_treat, post_treat]
colors = ['lightblue', 'darkblue', 'lightcoral', 'darkred']
ax.bar(groups, ft_rates, color=colors)
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment by Group and Period')
ax.set_ylim([0.55, 0.75])

for i, v in enumerate(ft_rates):
    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure4_covariates.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_covariates.png")

print("\nAll figures created successfully!")

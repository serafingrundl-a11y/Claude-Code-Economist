"""
DACA Replication Study - Create Figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
results = pd.read_csv('analysis_results.csv')

# Create weights function
def weighted_mean(group):
    return np.average(group['FT'], weights=group['PERWT'])

# =============================================================================
# Figure 1: Parallel Trends - Full-time Employment Over Time
# =============================================================================
print("Creating Figure 1: Parallel Trends...")

yearly_ft = df.groupby(['YEAR', 'ELIGIBLE']).apply(weighted_mean).unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years = yearly_ft.index.values
treated = yearly_ft[1].values
control = yearly_ft[0].values

ax.plot(years, treated, 'o-', color='#2166ac', linewidth=2, markersize=8, label='Treatment (Ages 26-30)')
ax.plot(years, control, 's--', color='#b2182b', linewidth=2, markersize=8, label='Control (Ages 31-35)')

# Add vertical line at treatment
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.75), fontsize=10, ha='center', color='gray')

# Shade treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status')
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.55, 0.80)
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure1_parallel_trends.png/pdf")

# =============================================================================
# Figure 2: Event Study Coefficients
# =============================================================================
print("Creating Figure 2: Event Study...")

event_years = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
event_coefs = []
event_ses = []

for year in event_years:
    if year == 2011:
        event_coefs.append(0)  # Reference year
        event_ses.append(0)
    else:
        event_coefs.append(results[f'event_y{year}_coef'].values[0])
        event_ses.append(results[f'event_y{year}_se'].values[0])

event_coefs = np.array(event_coefs)
event_ses = np.array(event_ses)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with confidence intervals
ax.errorbar(event_years, event_coefs, yerr=1.96*event_ses,
            fmt='o', color='#2166ac', markersize=10, capsize=5, capthick=2, linewidth=2)

# Connect points with lines
ax.plot(event_years, event_coefs, '-', color='#2166ac', alpha=0.5, linewidth=1)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at treatment
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)

# Shade treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='blue')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Treatment Effects by Year\n(Reference Year: 2011)')
ax.set_xticks(event_years)
ax.set_ylim(-0.15, 0.15)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure2_event_study.png/pdf")

# =============================================================================
# Figure 3: DiD Visual Representation
# =============================================================================
print("Creating Figure 3: DiD Visual...")

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate averages for pre and post periods
ft_treated_pre = df[(df['ELIGIBLE']==1) & (df['AFTER']==0)].apply(
    lambda x: np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['FT'],
                         weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==0)]['PERWT']), axis=1).iloc[0]
ft_treated_post = np.average(df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['FT'],
                              weights=df[(df['ELIGIBLE']==1) & (df['AFTER']==1)]['PERWT'])
ft_control_pre = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['FT'],
                             weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==0)]['PERWT'])
ft_control_post = np.average(df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['FT'],
                              weights=df[(df['ELIGIBLE']==0) & (df['AFTER']==1)]['PERWT'])

# Calculate counterfactual for treatment group
counterfactual = ft_treated_pre + (ft_control_post - ft_control_pre)

# Plot
x_pre = 0.3
x_post = 0.7

# Treatment group
ax.plot([x_pre, x_post], [ft_treated_pre, ft_treated_post], 'o-',
        color='#2166ac', linewidth=3, markersize=12, label='Treatment (Observed)')

# Control group
ax.plot([x_pre, x_post], [ft_control_pre, ft_control_post], 's--',
        color='#b2182b', linewidth=3, markersize=12, label='Control (Observed)')

# Counterfactual
ax.plot([x_pre, x_post], [ft_treated_pre, counterfactual], 'o:',
        color='#2166ac', linewidth=2, markersize=8, alpha=0.5, label='Treatment (Counterfactual)')

# Draw the treatment effect arrow
ax.annotate('', xy=(x_post + 0.03, ft_treated_post), xytext=(x_post + 0.03, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(x_post + 0.06, (ft_treated_post + counterfactual)/2,
        f'DiD Effect\n= {(ft_treated_post - counterfactual)*100:.1f} pp',
        fontsize=10, va='center', color='green', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([x_pre, x_post])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Difference-in-Differences: DACA Effect on Full-Time Employment')
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visual.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure3_did_visual.png/pdf")

# =============================================================================
# Figure 4: Coefficient Comparison Across Models
# =============================================================================
print("Creating Figure 4: Model Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))

models = ['Model 1\n(Basic)', 'Model 2\n(+ Demographics)', 'Model 3\n(+ State/Year FE)']
coefs = [results['model1_coef'].values[0],
         results['model2_coef'].values[0],
         results['model3_coef'].values[0]]
ses = [results['model1_se'].values[0],
       results['model2_se'].values[0],
       results['model3_se'].values[0]]

x = np.arange(len(models))
colors = ['#4393c3', '#2166ac', '#053061']

ax.bar(x, coefs, yerr=1.96*np.array(ses), capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

ax.set_ylabel('DiD Coefficient')
ax.set_title('DACA Effect on Full-Time Employment Across Model Specifications')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(-0.02, 0.15)

# Add coefficient labels
for i, (c, s) in enumerate(zip(coefs, ses)):
    ax.text(i, c + 1.96*s + 0.01, f'{c:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure4_model_comparison.png/pdf")

# =============================================================================
# Figure 5: Sample Distribution by Year and Group
# =============================================================================
print("Creating Figure 5: Sample Distribution...")

fig, ax = plt.subplots(figsize=(10, 6))

sample_counts = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()

x = np.arange(len(sample_counts.index))
width = 0.35

bars1 = ax.bar(x - width/2, sample_counts[1], width, label='Treatment (Ages 26-30)', color='#2166ac', alpha=0.8)
bars2 = ax.bar(x + width/2, sample_counts[0], width, label='Control (Ages 31-35)', color='#b2182b', alpha=0.8)

ax.set_xlabel('Year')
ax.set_ylabel('Number of Observations')
ax.set_title('Sample Size by Year and DACA Eligibility Status')
ax.set_xticks(x)
ax.set_xticklabels(sample_counts.index)
ax.legend()

# Add vertical line at treatment
treatment_idx = 3.5  # Between 2011 and 2013
ax.axvline(x=treatment_idx, color='gray', linestyle=':', linewidth=2)

plt.tight_layout()
plt.savefig('figure5_sample_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_sample_distribution.pdf', bbox_inches='tight')
plt.close()
print("  Saved: figure5_sample_distribution.png/pdf")

print("\nAll figures created successfully!")

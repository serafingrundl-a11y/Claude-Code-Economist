"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Load analysis data
df = pd.read_csv('analysis_data.csv')

# ============================================================================
# FIGURE 1: Parallel Trends - Full-time Employment by Year
# ============================================================================
print("Creating Figure 1: Parallel Trends...")

# Calculate mean full-time employment by year and treatment status
trends = df.groupby(['YEAR', 'treatment'])['fulltime'].mean().unstack()
trends.columns = ['Control (ages 31-35)', 'Treatment (ages 26-30)']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(trends.index, trends['Treatment (ages 26-30)'], 'o-', color='#1f77b4',
        linewidth=2, markersize=8, label='Treatment (ages 26-30)')
ax.plot(trends.index, trends['Control (ages 31-35)'], 's--', color='#ff7f0e',
        linewidth=2, markersize=8, label='Control (ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, label='DACA (June 2012)')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Full-time Employment Trends by Treatment Status')
ax.legend(loc='lower left')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.55, 0.75)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure1_parallel_trends.png/pdf")

# ============================================================================
# FIGURE 2: Event Study Plot
# ============================================================================
print("Creating Figure 2: Event Study...")

# Event study coefficients from analysis
event_data = {
    'year': [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016],
    'coef': [-0.0223, -0.0381, -0.0079, -0.0202, -0.0285, 0, 0.0237, 0.0296, 0.0265, 0.0489],
    'se': [0.0200, 0.0202, 0.0206, 0.0210, 0.0209, 0, 0.0217, 0.0219, 0.0222, 0.0223]
}
event_df = pd.DataFrame(event_data)
event_df['ci_lower'] = event_df['coef'] - 1.96 * event_df['se']
event_df['ci_upper'] = event_df['coef'] + 1.96 * event_df['se']

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
ax.errorbar(event_df['year'], event_df['coef'],
            yerr=1.96*event_df['se'],
            fmt='o', capsize=4, capthick=2,
            color='#1f77b4', markersize=8, linewidth=2)

# Connect with line
ax.plot(event_df['year'], event_df['coef'], '-', color='#1f77b4', alpha=0.5)

# Reference lines
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, label='DACA (June 2012)')

# Shade post-period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (relative to 2011)')
ax.set_title('Event Study: Dynamic Treatment Effects on Full-time Employment')
ax.legend(loc='upper left')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

# Add annotation
ax.annotate('Reference\nyear', xy=(2011, 0), xytext=(2011.5, 0.04),
            fontsize=10, ha='left',
            arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure2_event_study.png/pdf")

# ============================================================================
# FIGURE 3: DiD Visualization
# ============================================================================
print("Creating Figure 3: DiD Visualization...")

# Mean values for each group-period cell
treatment_pre = df[(df['treatment']==1) & (df['post']==0)]['fulltime'].mean()
treatment_post = df[(df['treatment']==1) & (df['post']==1)]['fulltime'].mean()
control_pre = df[(df['treatment']==0) & (df['post']==0)]['fulltime'].mean()
control_post = df[(df['treatment']==0) & (df['post']==1)]['fulltime'].mean()

# Counterfactual for treatment group
counterfactual_post = treatment_pre + (control_post - control_pre)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot actual trends
ax.plot([0, 1], [treatment_pre, treatment_post], 'o-', color='#1f77b4',
        linewidth=3, markersize=12, label='Treatment (actual)')
ax.plot([0, 1], [control_pre, control_post], 's-', color='#ff7f0e',
        linewidth=3, markersize=12, label='Control')

# Plot counterfactual
ax.plot([0, 1], [treatment_pre, counterfactual_post], 'o--', color='#1f77b4',
        linewidth=2, markersize=8, alpha=0.5, label='Treatment (counterfactual)')

# Annotate the DiD effect
ax.annotate('', xy=(1.05, treatment_post), xytext=(1.05, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (treatment_post + counterfactual_post)/2, f'DiD = {treatment_post - counterfactual_post:.3f}',
        fontsize=12, color='green', fontweight='bold', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Difference-in-Differences: DACA Effect on Full-time Employment')
ax.legend(loc='lower left')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(0.55, 0.70)

# Add data labels
ax.annotate(f'{treatment_pre:.3f}', xy=(0, treatment_pre), xytext=(-0.08, treatment_pre),
            fontsize=10, ha='right', va='center')
ax.annotate(f'{treatment_post:.3f}', xy=(1, treatment_post), xytext=(0.92, treatment_post+0.01),
            fontsize=10, ha='right', va='bottom')
ax.annotate(f'{control_pre:.3f}', xy=(0, control_pre), xytext=(-0.08, control_pre),
            fontsize=10, ha='right', va='center')
ax.annotate(f'{control_post:.3f}', xy=(1, control_post), xytext=(0.92, control_post-0.01),
            fontsize=10, ha='right', va='top')

plt.tight_layout()
plt.savefig('figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure3_did_visualization.png/pdf")

# ============================================================================
# FIGURE 4: Sample Distribution by Age
# ============================================================================
print("Creating Figure 4: Age Distribution...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Age at DACA distribution
ax1.hist(df[df['treatment']==1]['age_at_daca'], bins=5, alpha=0.7, color='#1f77b4',
         label='Treatment', edgecolor='black')
ax1.hist(df[df['treatment']==0]['age_at_daca'], bins=5, alpha=0.7, color='#ff7f0e',
         label='Control', edgecolor='black')
ax1.set_xlabel('Age at DACA Implementation (June 2012)')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Age at DACA Implementation')
ax1.legend()

# Age at immigration distribution
ax2.hist(df[df['treatment']==1]['age_at_immigration'], bins=16, alpha=0.7, color='#1f77b4',
         label='Treatment', edgecolor='black')
ax2.hist(df[df['treatment']==0]['age_at_immigration'], bins=16, alpha=0.7, color='#ff7f0e',
         label='Control', edgecolor='black')
ax2.set_xlabel('Age at Immigration')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Age at Immigration')
ax2.legend()

plt.tight_layout()
plt.savefig('figure4_age_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_age_distribution.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure4_age_distribution.png/pdf")

# ============================================================================
# FIGURE 5: Full-time Employment by Education
# ============================================================================
print("Creating Figure 5: Employment by Education...")

# Employment by education and treatment status
educ_emp = df.groupby(['educ_cat', 'treatment', 'post'])['fulltime'].mean().unstack(level=[1,2])

fig, ax = plt.subplots(figsize=(10, 6))

# Get education categories in order
educ_order = ['less_hs', 'hs', 'some_college', 'college_plus']
educ_labels = ['Less than HS', 'High School', 'Some College', 'College+']

x = np.arange(len(educ_order))
width = 0.2

# Plot bars for each group
bars1 = ax.bar(x - 1.5*width, [educ_emp.loc[e, (0, 0)] if e in educ_emp.index else 0 for e in educ_order],
               width, label='Control Pre', color='#ff7f0e', alpha=0.6)
bars2 = ax.bar(x - 0.5*width, [educ_emp.loc[e, (0, 1)] if e in educ_emp.index else 0 for e in educ_order],
               width, label='Control Post', color='#ff7f0e')
bars3 = ax.bar(x + 0.5*width, [educ_emp.loc[e, (1, 0)] if e in educ_emp.index else 0 for e in educ_order],
               width, label='Treatment Pre', color='#1f77b4', alpha=0.6)
bars4 = ax.bar(x + 1.5*width, [educ_emp.loc[e, (1, 1)] if e in educ_emp.index else 0 for e in educ_order],
               width, label='Treatment Post', color='#1f77b4')

ax.set_xlabel('Education Level')
ax.set_ylabel('Full-time Employment Rate')
ax.set_title('Full-time Employment by Education and Treatment Status')
ax.set_xticks(x)
ax.set_xticklabels(educ_labels)
ax.legend()
ax.set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig('figure5_employment_education.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_employment_education.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure5_employment_education.png/pdf")

# ============================================================================
# FIGURE 6: Coefficient Comparison Across Models
# ============================================================================
print("Creating Figure 6: Model Comparison...")

# Load regression results
results = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(results))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot coefficients with error bars
ax.errorbar(results['Coefficient'], y_pos,
            xerr=1.96*results['Std_Error'],
            fmt='o', capsize=5, capthick=2,
            color='#1f77b4', markersize=10, linewidth=2)

ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(results['Model'])
ax.set_xlabel('DiD Coefficient (Effect on Full-time Employment)')
ax.set_title('DACA Effect Estimates Across Model Specifications')

# Add sample size annotation
for i, (_, row) in enumerate(results.iterrows()):
    ax.annotate(f"N={row['N']:,.0f}",
                xy=(row['Coefficient'] + 1.96*row['Std_Error'] + 0.005, i),
                fontsize=9, va='center')

plt.tight_layout()
plt.savefig('figure6_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_model_comparison.pdf', bbox_inches='tight')
plt.close()

print("   Saved: figure6_model_comparison.png/pdf")

print("\nAll figures created successfully!")

"""
Create figures for DACA Replication Report
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Figure 1: Full-time employment trends by group
print("Creating Figure 1: Employment trends...")

# Data from analysis
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treatment = [0.628355, 0.645705, 0.642438, 0.592482, 0.581720, 0.570341, 0.609847, 0.618587, 0.642824, 0.672046]
control = [0.680582, 0.689277, 0.662887, 0.612973, 0.609038, 0.592631, 0.600121, 0.598866, 0.623626, 0.627131]

fig, ax = plt.subplots()
ax.plot(years, treatment, 'o-', linewidth=2, markersize=8, label='Treatment (Ages 26-30 in 2012)', color='#1f77b4')
ax.plot(years, control, 's--', linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)', color='#ff7f0e')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7, label='DACA Implementation (2012)')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Full-Time Employment Rates by Treatment Status', fontsize=16)
ax.legend(loc='lower left', fontsize=11)
ax.set_ylim(0.5, 0.75)
ax.set_xticks(years)
ax.set_xticklabels([str(y) for y in years], rotation=45)

plt.tight_layout()
plt.savefig('figure1_employment_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_employment_trends.pdf', bbox_inches='tight')
plt.close()
print("Saved figure1_employment_trends.png")

# Figure 2: Event study plot
print("Creating Figure 2: Event study...")
event_df = pd.read_csv('event_study_results.csv')

# Add reference year (2011 = 0)
event_df = pd.concat([event_df, pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'Std_Error': [0], 'CI_lower': [0], 'CI_upper': [0]})], ignore_index=True)
event_df = event_df.sort_values('Year')

fig, ax = plt.subplots()

# Plot coefficients with confidence intervals
ax.errorbar(event_df['Year'], event_df['Coefficient'],
            yerr=[event_df['Coefficient'] - event_df['CI_lower'], event_df['CI_upper'] - event_df['Coefficient']],
            fmt='o', markersize=8, capsize=5, capthick=2, linewidth=2, color='#1f77b4')

# Connect points
ax.plot(event_df['Year'], event_df['Coefficient'], '-', linewidth=1, alpha=0.5, color='#1f77b4')

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at 2012
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=14)
ax.set_title('Event Study: Treatment Effects Over Time', fontsize=16)
ax.legend(loc='upper left', fontsize=11)
ax.set_xticks(event_df['Year'].tolist())
ax.set_xticklabels([str(int(y)) for y in event_df['Year']], rotation=45)

plt.tight_layout()
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.pdf', bbox_inches='tight')
plt.close()
print("Saved figure2_event_study.png")

# Figure 3: DiD visualization
print("Creating Figure 3: DiD visualization...")

fig, ax = plt.subplots()

# Pre and post means
pre_treat = 0.6112
post_treat = 0.6345
pre_control = 0.6425
post_control = 0.6116

# Plot actual trends
ax.plot([0, 1], [pre_treat, post_treat], 'o-', linewidth=3, markersize=12,
        label='Treatment Group (Actual)', color='#1f77b4')
ax.plot([0, 1], [pre_control, post_control], 's-', linewidth=3, markersize=12,
        label='Control Group (Actual)', color='#ff7f0e')

# Plot counterfactual
counterfactual = pre_treat + (post_control - pre_control)
ax.plot([0, 1], [pre_treat, counterfactual], 'o--', linewidth=2, markersize=8,
        label='Treatment Counterfactual', color='#1f77b4', alpha=0.5)

# Arrow for treatment effect
ax.annotate('', xy=(1, post_treat), xytext=(1, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.05, (post_treat + counterfactual)/2, f'DiD = {post_treat - counterfactual:.3f}',
        fontsize=12, color='green', fontweight='bold')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Full-Time Employment Rate', fontsize=14)
ax.set_title('Difference-in-Differences Visualization', fontsize=16)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend(loc='upper right', fontsize=11)
ax.set_ylim(0.55, 0.70)

plt.tight_layout()
plt.savefig('figure3_did_visual.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_visual.pdf', bbox_inches='tight')
plt.close()
print("Saved figure3_did_visual.png")

# Figure 4: Regression coefficients comparison
print("Creating Figure 4: Coefficient comparison...")

reg_df = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(reg_df))
ax.barh(y_pos, reg_df['Coefficient'], xerr=[reg_df['Coefficient'] - reg_df['CI_lower'],
        reg_df['CI_upper'] - reg_df['Coefficient']], align='center',
        capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

ax.set_yticks(y_pos)
ax.set_yticklabels(reg_df['Model'])
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=14)
ax.set_title('Comparison of DiD Estimates Across Specifications', fontsize=16)
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add coefficient values on bars
for i, (coef, ci_l, ci_u) in enumerate(zip(reg_df['Coefficient'], reg_df['CI_lower'], reg_df['CI_upper'])):
    ax.text(coef + 0.005, i, f'{coef:.3f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('figure4_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficients.pdf', bbox_inches='tight')
plt.close()
print("Saved figure4_coefficients.png")

print("\nAll figures created successfully!")

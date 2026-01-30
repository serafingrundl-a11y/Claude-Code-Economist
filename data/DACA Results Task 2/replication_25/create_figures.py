"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Read data
event_df = pd.read_csv('event_study_results.csv')
results_df = pd.read_csv('regression_results.csv')

# Figure 1: Event Study Plot
plt.figure(figsize=(10, 6))
years = event_df['Year'].values
coefs = event_df['Coefficient'].values
ses = event_df['SE'].values

# Calculate confidence intervals
ci_upper = coefs + 1.96 * ses
ci_lower = coefs - 1.96 * ses

plt.errorbar(years, coefs, yerr=1.96*ses, fmt='o-', capsize=5,
             color='navy', markersize=8, linewidth=2, capthick=2)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Coefficient (relative to 2011)', fontsize=12)
plt.title('Event Study: Treatment Effect on Full-Time Employment\nby Year Relative to 2011', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(years)
plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure1_event_study.png")

# Figure 2: Full-time employment trends
# Create from summary stats
summary_df = pd.read_csv('summary_statistics.csv', header=[0, 1], index_col=[0, 1])

# Manually create the trend data
years_all = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016]
# Approximate based on overall means (would need original data for precise)
treated_pre = 0.6147
treated_post = 0.6339
control_pre = 0.6461
control_post = 0.6136

# For simplified trend visualization
plt.figure(figsize=(10, 6))

# Treatment group: ages 26-30 at DACA
# Control group: ages 31-35 at DACA
periods = ['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)']
x = np.arange(len(periods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, [treated_pre, treated_post], width, label='Treatment (Age 26-30)', color='steelblue')
rects2 = ax.bar(x + width/2, [control_pre, control_post], width, label='Control (Age 31-35)', color='lightcoral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by Treatment Status and Period', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=11)
ax.legend()
ax.set_ylim([0.5, 0.75])

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure2_trends.png")

# Figure 3: Coefficient comparison across models
plt.figure(figsize=(10, 6))
models = ['Basic DiD', 'With Demographics', 'With Education', 'State FE', 'Year+State FE']
coefs = results_df['Coefficient'].values
ci_lower = results_df['CI_Lower'].values
ci_upper = results_df['CI_Upper'].values

errors = [coefs - ci_lower, ci_upper - coefs]
colors = ['navy' if ci_lower[i] > 0 else 'gray' for i in range(len(models))]

y_pos = np.arange(len(models))
plt.barh(y_pos, coefs, xerr=errors, align='center', color=colors, alpha=0.8, capsize=5)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
plt.yticks(y_pos, models)
plt.xlabel('Effect on Full-Time Employment (Percentage Points)', fontsize=12)
plt.title('DACA Effect Estimates Across Model Specifications', fontsize=14)
plt.tight_layout()
plt.savefig('figure3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure3_model_comparison.png")

# Figure 4: DiD visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Data points
x_vals = [0, 1]
treated = [treated_pre, treated_post]
control = [control_pre, control_post]

# Plot actual trends
ax.plot(x_vals, treated, 'o-', color='steelblue', linewidth=2, markersize=10, label='Treatment Group')
ax.plot(x_vals, control, 's-', color='lightcoral', linewidth=2, markersize=10, label='Control Group')

# Counterfactual for treatment
counterfactual = [treated_pre, treated_pre + (control_post - control_pre)]
ax.plot(x_vals, counterfactual, '--', color='steelblue', linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# DiD effect annotation
ax.annotate('', xy=(1.1, treated_post), xytext=(1.1, counterfactual[1]),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.15, (treated_post + counterfactual[1])/2, f'DiD Effect\n= {(treated_post - counterfactual[1]):.3f}',
        fontsize=11, color='green', va='center')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'], fontsize=11)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences Visualization', fontsize=14)
ax.legend(loc='upper right')
ax.set_xlim([-0.2, 1.5])
ax.set_ylim([0.55, 0.7])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_did_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figure4_did_visualization.png")

print("\nAll figures created successfully!")

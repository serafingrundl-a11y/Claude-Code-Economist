"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Set working directory
os.chdir(r'C:\Users\seraf\DACA Results Task 3\replication_31')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (8, 5)

# Load results
with open('tables/key_results.pkl', 'rb') as f:
    results = pickle.load(f)

# =============================================================================
# Figure 1: Full-time Employment Rates by Group and Period
# =============================================================================
print("Creating Figure 1: FT Rates by Group and Period...")

fig, ax = plt.subplots(figsize=(8, 5))

groups = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
treatment_rates = [results['ft_pre_treat'] * 100, results['ft_post_treat'] * 100]
control_rates = [results['ft_pre_ctrl'] * 100, results['ft_post_ctrl'] * 100]

x = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x - width/2, treatment_rates, width, label='Treatment (Ages 26-30)', color='#2166ac', alpha=0.8)
bars2 = ax.bar(x + width/2, control_rates, width, label='Control (Ages 31-35)', color='#b2182b', alpha=0.8)

ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_xlabel('Period')
ax.set_title('Full-Time Employment Rates Before and After DACA')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(loc='upper right')
ax.set_ylim(50, 80)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/ft_rates_by_group.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/ft_rates_by_group.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/ft_rates_by_group.png")

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("Creating Figure 2: Event Study...")

event_df = results['event_study']

fig, ax = plt.subplots(figsize=(10, 6))

years = event_df['Year'].tolist()
# Add reference year (2011) with coefficient 0
years_full = [2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
coefs_full = []
ci_low_full = []
ci_high_full = []

for y in years_full:
    if y == 2011:
        coefs_full.append(0)
        ci_low_full.append(0)
        ci_high_full.append(0)
    else:
        row = event_df[event_df['Year'] == y].iloc[0]
        coefs_full.append(row['Coefficient'])
        ci_low_full.append(row['CI_low'])
        ci_high_full.append(row['CI_high'])

# Plot
ax.errorbar(years_full, coefs_full,
            yerr=[np.array(coefs_full) - np.array(ci_low_full),
                  np.array(ci_high_full) - np.array(coefs_full)],
            fmt='o-', color='#2166ac', capsize=4, capthick=1.5, linewidth=2, markersize=8)

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)

# Vertical line at policy implementation
ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation (2012)')

# Shade pre and post periods
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(years_full)
ax.set_xlim(2007.5, 2016.5)
ax.legend(loc='upper left')

# Add note about reference year
ax.annotate('Reference Year', xy=(2011, 0), xytext=(2011.3, 0.04),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('figures/event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/event_study.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/event_study.png")

# =============================================================================
# Figure 3: DiD Estimate Comparison Across Models
# =============================================================================
print("Creating Figure 3: Model Comparison...")

results_summary = results['results_summary']

fig, ax = plt.subplots(figsize=(10, 6))

models = ['Basic DiD', '+ Demographics', '+ State/Year FE\n(Preferred)',
          '+ State Policies', 'Males Only', 'Females Only']
estimates = results_summary['Estimate'].values * 100  # Convert to percentage points
ci_low = results_summary['CI_low'].values * 100
ci_high = results_summary['CI_high'].values * 100

y_pos = np.arange(len(models))

# Plot
colors = ['#2166ac', '#4393c3', '#d6604d', '#b2182b', '#762a83', '#9970ab']
ax.barh(y_pos, estimates, xerr=[estimates - ci_low, ci_high - estimates],
        color=colors, alpha=0.8, capsize=4, ecolor='black')

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(models)
ax.set_xlabel('DiD Estimate (Percentage Points)')
ax.set_title('DACA Effect on Full-Time Employment: Comparison Across Models')

# Add value labels
for i, (est, lo, hi) in enumerate(zip(estimates, ci_low, ci_high)):
    label = f'{est:.2f} pp'
    ax.annotate(label, xy=(est + 0.5, i), va='center', fontsize=9)

plt.tight_layout()
plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/model_comparison.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/model_comparison.png")

# =============================================================================
# Figure 4: Trends in FT Employment Over Time
# =============================================================================
print("Creating Figure 4: Trends Over Time...")

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Calculate weighted FT rates by year and group
def weighted_mean(x, w):
    return np.average(x, weights=w)

trend_data = []
for year in sorted(df['YEAR'].unique()):
    for elig in [0, 1]:
        sub = df[(df['YEAR'] == year) & (df['ELIGIBLE'] == elig)]
        ft_rate = weighted_mean(sub['FT'], sub['PERWT'])
        trend_data.append({
            'Year': year,
            'Group': 'Treatment' if elig == 1 else 'Control',
            'FT_Rate': ft_rate * 100
        })

trend_df = pd.DataFrame(trend_data)

fig, ax = plt.subplots(figsize=(10, 6))

for group, color, marker in [('Treatment', '#2166ac', 'o'), ('Control', '#b2182b', 's')]:
    data = trend_df[trend_df['Group'] == group]
    ax.plot(data['Year'], data['FT_Rate'], marker=marker, color=color,
            linewidth=2, markersize=8, label=f'{group} (Ages {"26-30" if group == "Treatment" else "31-35"})')

ax.axvline(x=2012, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')
ax.axvspan(2007.5, 2011.5, alpha=0.1, color='gray')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Trends in Full-Time Employment by Treatment Status')
ax.legend(loc='lower right')
ax.set_xticks(sorted(df['YEAR'].unique()))
ax.set_ylim(55, 75)

plt.tight_layout()
plt.savefig('figures/ft_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/ft_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/ft_trends.png")

# =============================================================================
# Figure 5: DiD Visualization
# =============================================================================
print("Creating Figure 5: DiD Visualization...")

fig, ax = plt.subplots(figsize=(8, 6))

# Pre-post means
pre_treat = results['ft_pre_treat'] * 100
post_treat = results['ft_post_treat'] * 100
pre_ctrl = results['ft_pre_ctrl'] * 100
post_ctrl = results['ft_post_ctrl'] * 100

# Plot actual trends
ax.plot([0, 1], [pre_treat, post_treat], 'o-', color='#2166ac', linewidth=2,
        markersize=10, label='Treatment (Actual)')
ax.plot([0, 1], [pre_ctrl, post_ctrl], 's-', color='#b2182b', linewidth=2,
        markersize=10, label='Control (Actual)')

# Counterfactual (parallel trend)
counterfactual = pre_treat + (post_ctrl - pre_ctrl)
ax.plot([0, 1], [pre_treat, counterfactual], 'o--', color='#2166ac', linewidth=2,
        markersize=10, alpha=0.5, label='Treatment (Counterfactual)')

# DiD effect arrow
ax.annotate('', xy=(1.05, post_treat), xytext=(1.05, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(1.1, (post_treat + counterfactual)/2, f'DiD\n{(post_treat - counterfactual):.1f} pp',
        va='center', fontsize=10, color='green', fontweight='bold')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)'])
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Difference-in-Differences Visualization')
ax.legend(loc='upper left')
ax.set_xlim(-0.2, 1.4)
ax.set_ylim(55, 75)

plt.tight_layout()
plt.savefig('figures/did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/did_visualization.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/did_visualization.png")

print("\nAll figures created successfully!")

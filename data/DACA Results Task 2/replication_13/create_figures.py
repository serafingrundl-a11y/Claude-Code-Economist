"""
Create figures for DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.figsize'] = (8, 5)

# Figure 1: Trends in Full-Time Employment
print("Creating Figure 1: Trends...")
trends = pd.read_csv('trends_data.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines
ax.plot(trends['YEAR'], trends['Control']*100, 'o-', color='blue',
        linewidth=2, markersize=8, label='Control (ages 31-35 in 2012)')
ax.plot(trends['YEAR'], trends['Treatment']*100, 's-', color='red',
        linewidth=2, markersize=8, label='Treatment (ages 26-30 in 2012)')

# Add vertical line for DACA
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, 72, 'DACA\nImplemented', fontsize=10, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate (%)')
ax.set_title('Full-Time Employment Trends by Treatment Status')
ax.legend(loc='lower left')
ax.set_xticks(trends['YEAR'])
ax.set_ylim([55, 75])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
print("   Saved: figure1_trends.png and figure1_trends.pdf")
plt.close()

# Figure 2: Event Study
print("Creating Figure 2: Event Study...")
event = pd.read_csv('event_study_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Calculate confidence intervals (95%)
event['CI_lower'] = event['Coefficient'] - 1.96 * event['Std_Error']
event['CI_upper'] = event['Coefficient'] + 1.96 * event['Std_Error']

# Colors for pre and post
colors = ['blue' if year < 2012 else 'red' for year in event['Year']]
colors[5] = 'gray'  # 2011 reference year

# Plot points with error bars
ax.errorbar(event['Year'], event['Coefficient'],
            yerr=1.96*event['Std_Error'],
            fmt='o', capsize=5, capthick=2, markersize=8,
            color='black', ecolor='black', linewidth=2)

# Fill circles
for i, (year, coef) in enumerate(zip(event['Year'], event['Coefficient'])):
    color = 'blue' if year < 2012 else ('gray' if year == 2011 else 'red')
    ax.plot(year, coef, 'o', color=color, markersize=10, zorder=5)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA
ax.axvline(x=2011.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2011.6, 0.10, 'DACA', fontsize=10, color='gray')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Relative to 2011)')
ax.set_title('Event Study: Effect of DACA Eligibility on Full-Time Employment')
ax.set_xticks(event['Year'])
ax.set_ylim([-0.06, 0.12])
ax.grid(True, alpha=0.3)

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                          markersize=10, label='Pre-Treatment'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                          markersize=10, label='Post-Treatment'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                          markersize=10, label='Reference Year (2011)')]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
print("   Saved: figure2_eventstudy.png and figure2_eventstudy.pdf")
plt.close()

# Figure 3: Coefficient Comparison
print("Creating Figure 3: Coefficient Comparison...")
results = pd.read_csv('regression_results.csv')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
y_pos = np.arange(len(results))
ax.barh(y_pos, results['DiD_Coefficient'], xerr=1.96*results['Std_Error'],
        color=['steelblue', 'steelblue', 'steelblue', 'steelblue', 'darkred', 'steelblue'],
        capsize=5, alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(results['Model'])
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('DiD Coefficient (Effect on P(Full-Time Employment))')
ax.set_title('Comparison of Model Specifications')
ax.grid(True, alpha=0.3, axis='x')

# Highlight preferred model
ax.barh(4, results.iloc[4]['DiD_Coefficient'], color='darkred', alpha=0.8)

plt.tight_layout()
plt.savefig('figure3_coefficients.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_coefficients.pdf', bbox_inches='tight')
print("   Saved: figure3_coefficients.png and figure3_coefficients.pdf")
plt.close()

print("\nAll figures created successfully!")

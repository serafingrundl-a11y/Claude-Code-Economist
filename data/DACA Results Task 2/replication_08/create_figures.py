"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_DIR = r"C:\Users\seraf\DACA Results Task 2\replication_08"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# FIGURE 1: Full-time Employment Trends
# ============================================================================
print("Creating Figure 1: Full-time Employment Trends...")

fig1_data = pd.read_csv(f"{OUTPUT_DIR}/figure1_data.csv")

fig, ax = plt.subplots(figsize=(10, 6))

# Treatment group
treat_data = fig1_data[fig1_data['treat'] == 1].sort_values('year')
control_data = fig1_data[fig1_data['treat'] == 0].sort_values('year')

ax.plot(treat_data['year'], treat_data['fulltime_rate'] * 100, 'b-o',
        linewidth=2, markersize=8, label='Treatment (Ages 26-30 in 2012)')
ax.plot(control_data['year'], control_data['fulltime_rate'] * 100, 'r--s',
        linewidth=2, markersize=8, label='Control (Ages 31-35 in 2012)')

# Add vertical line for DACA implementation
ax.axvline(x=2012.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.6, 68, 'DACA\nImplemented', fontsize=9, color='gray', verticalalignment='center')

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='blue', label='Pre-period')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green', label='Post-period')

ax.set_xlabel('Year')
ax.set_ylabel('Full-time Employment Rate (%)')
ax.set_title('Full-time Employment Trends by Treatment Status')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(55, 75)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure1_trends.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/figure1_trends.pdf", bbox_inches='tight')
plt.close()
print("Figure 1 saved.")

# ============================================================================
# FIGURE 2: Event Study
# ============================================================================
print("Creating Figure 2: Event Study...")

event_data = pd.read_csv(f"{OUTPUT_DIR}/figure2_event_study.csv").sort_values('year')

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients
ax.errorbar(event_data['year'], event_data['coef'] * 100,
            yerr=1.96 * event_data['se'] * 100,
            fmt='o-', color='darkblue', linewidth=2, markersize=8,
            capsize=5, capthick=2, elinewidth=2)

# Add horizontal line at zero
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='gray', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 9, 'DACA', fontsize=9, color='gray', verticalalignment='center')

# Shade pre and post periods
ax.axvspan(2006, 2011.5, alpha=0.1, color='blue')
ax.axvspan(2012.5, 2016, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Coefficient (Percentage Points)')
ax.set_title('Event Study: Treatment Effect by Year (Reference: 2011)')
ax.set_xlim(2005.5, 2016.5)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

# Add note
ax.text(0.02, 0.02, 'Note: Error bars represent 95% confidence intervals',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure2_event_study.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/figure2_event_study.pdf", bbox_inches='tight')
plt.close()
print("Figure 2 saved.")

# ============================================================================
# FIGURE 3: Simple DID Visualization
# ============================================================================
print("Creating Figure 3: Simple DID Visualization...")

import json
with open(f"{OUTPUT_DIR}/analysis_results.json", 'r') as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(8, 6))

# Data points
pre_period = 0
post_period = 1
width = 0.35

# Treatment group
treat_pre = results['pre_treat_rate'] * 100
treat_post = results['post_treat_rate'] * 100

# Control group
control_pre = results['pre_control_rate'] * 100
control_post = results['post_control_rate'] * 100

# Plot lines
ax.plot([pre_period, post_period], [treat_pre, treat_post], 'b-o',
        linewidth=2.5, markersize=12, label='Treatment (26-30)')
ax.plot([pre_period, post_period], [control_pre, control_post], 'r--s',
        linewidth=2.5, markersize=12, label='Control (31-35)')

# Add counterfactual
counterfactual = treat_pre + (control_post - control_pre)
ax.plot([pre_period, post_period], [treat_pre, counterfactual], 'b:',
        linewidth=2, alpha=0.5, label='Treatment Counterfactual')

# Annotate DID effect
did_effect = results['simple_did'] * 100
mid_x = 1.05
ax.annotate('', xy=(mid_x, treat_post), xytext=(mid_x, counterfactual),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(mid_x + 0.08, (treat_post + counterfactual) / 2,
        f'DID = {did_effect:.1f} pp', fontsize=11, color='green',
        verticalalignment='center')

ax.set_xlabel('Period')
ax.set_ylabel('Full-time Employment Rate (%)')
ax.set_title('Difference-in-Differences: Effect of DACA on Full-time Employment')
ax.set_xlim(-0.3, 1.5)
ax.set_ylim(58, 72)
ax.set_xticks([pre_period, post_period])
ax.set_xticklabels(['Pre-DACA\n(2006-2011)', 'Post-DACA\n(2013-2016)'])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure3_did_visual.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/figure3_did_visual.pdf", bbox_inches='tight')
plt.close()
print("Figure 3 saved.")

# ============================================================================
# FIGURE 4: Robustness Check - By Gender
# ============================================================================
print("Creating Figure 4: Robustness by Gender...")

fig, ax = plt.subplots(figsize=(8, 5))

effects = ['Overall', 'Male', 'Female']
values = [results['model4_coef'] * 100, results['male_effect'] * 100, results['female_effect'] * 100]
colors = ['darkblue', 'steelblue', 'coral']

bars = ax.bar(effects, values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
            f'{val:.2f} pp', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Effect Size (Percentage Points)')
ax.set_title('Effect of DACA on Full-time Employment by Gender')
ax.set_ylim(0, 7)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure4_by_gender.png", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/figure4_by_gender.pdf", bbox_inches='tight')
plt.close()
print("Figure 4 saved.")

print("\nAll figures created successfully!")

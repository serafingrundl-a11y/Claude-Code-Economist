"""
Create figures for the DACA replication report
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Load event study data
event_df = pd.read_csv('event_study_data.csv')

# Figure 1: Event Study Plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot treatment and control group trends
ax.plot(event_df['Year'], event_df['FT_Treatment'], 'o-', color='#2166AC',
        linewidth=2, markersize=8, label='Treatment (ages 26-30 in 2012)')
ax.plot(event_df['Year'], event_df['FT_Control'], 's-', color='#B2182B',
        linewidth=2, markersize=8, label='Control (ages 31-35 in 2012)')

# Add vertical line at policy implementation
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.annotate('DACA\nImplemented', xy=(2012, 0.72), fontsize=10, ha='center',
            color='gray')

# Shade post-treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year')
ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment Rates by Treatment Status Over Time')
ax.legend(loc='lower right')
ax.set_xlim(2005.5, 2016.5)
ax.set_ylim(0.48, 0.75)
ax.set_xticks([2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("Figure 1 saved: Event study plot")

# Figure 2: Gap Plot (Treatment - Control)
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(event_df['Year'], event_df['Gap'], color=['#4393C3' if y < 2012 else '#D6604D' for y in event_df['Year']],
       edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Calculate pre and post mean gaps
pre_mean = event_df[event_df['Year'] <= 2011]['Gap'].mean()
post_mean = event_df[event_df['Year'] >= 2013]['Gap'].mean()

ax.axhline(y=pre_mean, color='#4393C3', linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(y=post_mean, color='#D6604D', linestyle=':', linewidth=2, alpha=0.7)

ax.annotate(f'Pre-mean: {pre_mean:.3f}', xy=(2006.5, pre_mean + 0.005), fontsize=10, color='#4393C3')
ax.annotate(f'Post-mean: {post_mean:.3f}', xy=(2014.5, post_mean + 0.008), fontsize=10, color='#D6604D')

ax.set_xlabel('Year')
ax.set_ylabel('Gap (Treatment - Control)')
ax.set_title('Difference in Full-Time Employment Rates: Treatment vs Control')
ax.set_xlim(2005.5, 2016.5)

plt.tight_layout()
plt.savefig('figure2_gap_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_gap_plot.pdf', bbox_inches='tight')
plt.close()

print("Figure 2 saved: Gap plot")

# Figure 3: DiD Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Summary stats
summary_df = pd.read_csv('summary_stats.csv')

# Pre and post means
pre_treat = summary_df[(summary_df['Group'] == 'Treatment') & (summary_df['Period'] == 'Pre')]['Full-Time Rate'].values[0]
post_treat = summary_df[(summary_df['Group'] == 'Treatment') & (summary_df['Period'] == 'Post')]['Full-Time Rate'].values[0]
pre_ctrl = summary_df[(summary_df['Group'] == 'Control') & (summary_df['Period'] == 'Pre')]['Full-Time Rate'].values[0]
post_ctrl = summary_df[(summary_df['Group'] == 'Control') & (summary_df['Period'] == 'Post')]['Full-Time Rate'].values[0]

# Create bar positions
x = np.array([0, 1])
width = 0.35

# Treatment bars
bars1 = ax.bar(x - width/2, [pre_treat, post_treat], width, label='Treatment (26-30)',
               color='#2166AC', edgecolor='black')
# Control bars
bars2 = ax.bar(x + width/2, [pre_ctrl, post_ctrl], width, label='Control (31-35)',
               color='#B2182B', edgecolor='black')

ax.set_ylabel('Full-Time Employment Rate')
ax.set_title('Full-Time Employment by Group and Period')
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA (2006-2011)', 'Post-DACA (2013-2016)'])
ax.legend()
ax.set_ylim(0.5, 0.7)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

# Add counterfactual line
counterfactual = pre_treat + (post_ctrl - pre_ctrl)
ax.plot([0 - width/2, 1 - width/2], [pre_treat, counterfactual], 'k--', linewidth=2, alpha=0.5)
ax.annotate(f'Counterfactual: {counterfactual:.3f}', xy=(1 - width/2, counterfactual - 0.015),
            fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.pdf', bbox_inches='tight')
plt.close()

print("Figure 3 saved: DiD bar chart")

print("\nAll figures created successfully!")

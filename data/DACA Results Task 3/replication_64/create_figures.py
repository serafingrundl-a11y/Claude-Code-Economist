"""
Create figures for the DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Read trends data
trends = pd.read_csv('trends_data.csv', index_col=0)
weighted_trends = pd.read_csv('weighted_trends_data.csv', index_col=0)

# Figure 1: Parallel Trends - FT Employment by Year and Eligibility
fig, ax = plt.subplots(figsize=(10, 6))
years = trends.index.astype(int)
ax.plot(years, trends['0'], 'b-o', label='Control (ages 31-35 in June 2012)', linewidth=2, markersize=8)
ax.plot(years, trends['1'], 'r-s', label='Treatment (ages 26-30 in June 2012)', linewidth=2, markersize=8)
ax.axvline(x=2012, color='gray', linestyle='--', linewidth=1.5, label='DACA Implementation (2012)')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks(years)
ax.set_ylim(0.55, 0.75)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure1_parallel_trends.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved: parallel trends")

# Figure 2: Event Study Coefficients
event_coefs = {
    '2008': -0.0591,
    '2009': -0.0388,
    '2010': -0.0663,
    '2011': 0.0,  # Reference year
    '2013': 0.0188,
    '2014': -0.0088,
    '2015': 0.0303,
    '2016': 0.0491
}
event_ses = {
    '2008': 0.0289,
    '2009': 0.0297,
    '2010': 0.0294,
    '2011': 0.0,
    '2013': 0.0306,
    '2014': 0.0308,
    '2015': 0.0316,
    '2016': 0.0314
}

years_event = list(event_coefs.keys())
coefs = list(event_coefs.values())
ses = list(event_ses.values())

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(years_event))
ax.errorbar(x, coefs, yerr=[1.96*s for s in ses], fmt='ko', capsize=5, capthick=2, markersize=8)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.axvline(x=3.5, color='red', linestyle='--', linewidth=1.5, label='DACA Implementation')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (relative to 2011)', fontsize=12)
ax.set_title('Event Study: ELIGIBLE × Year Interaction Coefficients', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(years_event)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figure2_event_study.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved: event study")

# Figure 3: DiD Visualization (2x2 table style)
fig, ax = plt.subplots(figsize=(8, 6))
# Data
treat_before = 0.6263
treat_after = 0.6658
control_before = 0.6697
control_after = 0.6449

periods = ['Before DACA\n(2008-2011)', 'After DACA\n(2013-2016)']
x = np.arange(len(periods))
width = 0.35

bars1 = ax.bar(x - width/2, [control_before, control_after], width, label='Control (ages 31-35)', color='steelblue')
bars2 = ax.bar(x + width/2, [treat_before, treat_after], width, label='Treatment (ages 26-30)', color='coral')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(periods)
ax.legend(fontsize=10)
ax.set_ylim(0.55, 0.75)

# Add value labels on bars
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_did_bars.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved: DiD bars")

# Figure 4: Coefficient comparison across models
models = ['Basic DiD', 'Year FE', 'Year FE +\nCovariates', 'Year + State FE\n+ Covariates', 'Weighted\nBasic DiD', 'Weighted\n+ FE + Cov']
coefs_model = [0.0643, 0.0629, 0.0545, 0.0546, 0.0748, 0.0614]
ses_model = [0.0153, 0.0152, 0.0141, 0.0142, 0.0181, 0.0166]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
ax.errorbar(x, coefs_model, yerr=[1.96*s for s in ses_model], fmt='ko', capsize=6, capthick=2, markersize=10)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax.set_xlabel('Model Specification', fontsize=12)
ax.set_ylabel('DiD Coefficient', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylim(-0.02, 0.12)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figure4_coefficient_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure4_coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 4 saved: coefficient comparison")

print("\nAll figures created successfully!")

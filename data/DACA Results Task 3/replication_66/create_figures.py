"""
Create figures for the DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv', low_memory=False)

# Create figure directory
import os
os.makedirs('figures', exist_ok=True)

# =============================================================================
# Figure 1: Parallel Trends - Full-Time Employment by Year and Eligibility
# =============================================================================
print("Creating Figure 1: Parallel Trends...")

ft_by_year = df.groupby(['YEAR', 'ELIGIBLE'])['FT'].mean().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years = ft_by_year.index.tolist()
treatment = ft_by_year[1].values
control = ft_by_year[0].values

ax.plot(years, treatment, 'o-', label='Treatment (Ages 26-30 in 2012)', color='blue', linewidth=2, markersize=8)
ax.plot(years, control, 's--', label='Control (Ages 31-35 in 2012)', color='red', linewidth=2, markersize=8)

# Add vertical line at 2012
ax.axvline(x=2012, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.62, 'DACA\nImplemented', fontsize=10, ha='left')

# Add shaded region for post-treatment
ax.axvspan(2012, 2017, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Figure 1: Full-Time Employment Trends by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.55, 0.75)
ax.set_xlim(2007.5, 2016.5)
ax.grid(True, alpha=0.3)

# Set x-ticks
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])

plt.tight_layout()
plt.savefig('figures/figure1_parallel_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure1_parallel_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure1_parallel_trends.png")

# =============================================================================
# Figure 2: Event Study Plot
# =============================================================================
print("Creating Figure 2: Event Study...")

event_study = pd.read_csv('event_study_results.csv')

# Add 2011 as reference year with coefficient 0
event_study_full = pd.concat([
    event_study,
    pd.DataFrame({'Year': [2011], 'Coefficient': [0], 'SE': [0], 'CI_low': [0], 'CI_high': [0]})
]).sort_values('Year')

fig, ax = plt.subplots(figsize=(10, 6))

years = event_study_full['Year'].values
coefs = event_study_full['Coefficient'].values
ci_low = event_study_full['CI_low'].values
ci_high = event_study_full['CI_high'].values

# Plot confidence intervals
ax.fill_between(years, ci_low, ci_high, alpha=0.3, color='blue')
ax.plot(years, coefs, 'o-', color='blue', linewidth=2, markersize=10)

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Add vertical line at 2012
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(2012.1, 0.08, 'DACA\nImplemented', fontsize=10, ha='left', color='red')

# Shade pre-treatment period
ax.axvspan(2007.5, 2012, alpha=0.1, color='gray')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Treatment Effect (Relative to 2011)', fontsize=12)
ax.set_title('Figure 2: Event Study - Dynamic Treatment Effects\n(Reference Year: 2011)', fontsize=14)
ax.set_xlim(2007.5, 2016.5)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/figure2_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure2_event_study.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure2_event_study.png")

# =============================================================================
# Figure 3: DiD Visualization (2x2 table)
# =============================================================================
print("Creating Figure 3: DiD Visualization...")

ft_means = df.groupby(['ELIGIBLE', 'AFTER'])['FT'].mean()

fig, ax = plt.subplots(figsize=(10, 6))

# Pre-treatment
pre_control = ft_means.loc[(0, 0)]
pre_treat = ft_means.loc[(1, 0)]
post_control = ft_means.loc[(0, 1)]
post_treat = ft_means.loc[(1, 1)]

# Plot bars
x = [0, 1]
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], [pre_control, post_control], width, label='Control (Ages 31-35)', color='red', alpha=0.7)
bars2 = ax.bar([i + width/2 for i in x], [pre_treat, post_treat], width, label='Treatment (Ages 26-30)', color='blue', alpha=0.7)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Figure 3: Difference-in-Differences Visualization', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(['Pre-DACA (2008-2011)', 'Post-DACA (2013-2016)'], fontsize=11)
ax.legend(fontsize=10)
ax.set_ylim(0.5, 0.75)

# Add DiD calculation annotation
did = (post_treat - pre_treat) - (post_control - pre_control)
ax.text(0.5, 0.52, f'DiD = ({post_treat:.3f} - {pre_treat:.3f}) - ({post_control:.3f} - {pre_control:.3f}) = {did:.3f}',
        ha='center', fontsize=11, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figures/figure3_did_visualization.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3_did_visualization.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure3_did_visualization.png")

# =============================================================================
# Figure 4: Coefficient Plot for Model Comparison
# =============================================================================
print("Creating Figure 4: Model Comparison...")

models = [
    ('Basic DiD', 0.0643, 0.0153),
    ('+ Demographics', 0.0558, 0.0142),
    ('+ State FE', 0.0559, 0.0142),
    ('+ Year FE', 0.0543, 0.0141),
    ('State + Year FE', 0.0544, 0.0142),
    ('Weighted Basic', 0.0748, 0.0181),
    ('Weighted + Controls', 0.0648, 0.0167),
    ('Clustered SE', 0.0643, 0.0141),
]

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(models))
names = [m[0] for m in models]
coefs = [m[1] for m in models]
ses = [m[2] for m in models]

# Calculate 95% CI
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

ax.errorbar(coefs, y_pos, xerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o', color='blue', ecolor='blue', capsize=4, capthick=2, markersize=8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Treatment Effect (Percentage Points)', fontsize=12)
ax.set_title('Figure 4: Treatment Effect Estimates Across Model Specifications\n(95% Confidence Intervals)', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Highlight preferred model
ax.axhspan(5.5, 6.5, alpha=0.2, color='yellow')
ax.text(0.09, 6, 'Preferred', fontsize=10, ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure4_model_comparison.png")

# =============================================================================
# Figure 5: Heterogeneous Effects
# =============================================================================
print("Creating Figure 5: Heterogeneous Effects...")

hetero = [
    ('Male', 0.0615, 0.0170),
    ('Female', 0.0452, 0.0232),
    ('Not Married', 0.0758, 0.0221),
    ('Married', 0.0586, 0.0214),
    ('High School', 0.0482, 0.0180),
    ('Some College', 0.1075, 0.0380),
]

fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(hetero))
names = [h[0] for h in hetero]
coefs = [h[1] for h in hetero]
ses = [h[2] for h in hetero]

ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

colors = ['blue', 'lightblue', 'green', 'lightgreen', 'red', 'salmon']

ax.barh(y_pos, coefs, xerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
        align='center', color=colors, capsize=4, alpha=0.7)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.axvline(x=0.0643, color='gray', linestyle='--', linewidth=1, label='Overall DiD')

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel('Treatment Effect (Percentage Points)', fontsize=12)
ax.set_title('Figure 5: Heterogeneous Treatment Effects by Subgroup\n(95% Confidence Intervals)', fontsize=14)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('figures/figure5_heterogeneous_effects.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure5_heterogeneous_effects.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure5_heterogeneous_effects.png")

# =============================================================================
# Figure 6: Sample Size by Year
# =============================================================================
print("Creating Figure 6: Sample Size by Year...")

sample_by_year = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years = sample_by_year.index.tolist()
treatment = sample_by_year[1].values
control = sample_by_year[0].values

x = np.arange(len(years))
width = 0.35

bars1 = ax.bar(x - width/2, control, width, label='Control (Ages 31-35)', color='red', alpha=0.7)
bars2 = ax.bar(x + width/2, treatment, width, label='Treatment (Ages 26-30)', color='blue', alpha=0.7)

ax.axvline(x=3.5, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(3.6, 1700, 'DACA', fontsize=10, ha='left')

ax.set_ylabel('Sample Size', fontsize=12)
ax.set_xlabel('Year', fontsize=12)
ax.set_title('Figure 6: Sample Size by Year and Eligibility Status', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/figure6_sample_size.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure6_sample_size.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figures/figure6_sample_size.png")

print("\nAll figures created successfully!")

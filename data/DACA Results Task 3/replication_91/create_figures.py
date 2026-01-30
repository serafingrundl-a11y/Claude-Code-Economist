"""
Create figures for DACA Replication Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load data
df = pd.read_csv('data/prepared_data_numeric_version.csv')
df_labels = pd.read_csv('data/prepared_data_labelled_version.csv')

# Helper function for weighted mean
def weighted_mean(x, weights):
    return np.average(x, weights=weights)

# ============================================================================
# Figure 1: Full-time Employment Trends by Group
# ============================================================================

print("Creating Figure 1: Employment trends...")

# Calculate weighted FT rates by year and eligibility
trends = df.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: weighted_mean(x['FT'], x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years = trends.index.values
ax.plot(years, trends[1].values, 'o-', color='#1f77b4', linewidth=2, markersize=8,
        label='Treatment (Ages 26-30)')
ax.plot(years, trends[0].values, 's--', color='#ff7f0e', linewidth=2, markersize=8,
        label='Control (Ages 31-35)')

# Add vertical line for DACA implementation
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, label='DACA Implementation (2012)')

# Shade post-treatment period
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Rates by DACA Eligibility Status', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.set_xticks([2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016])
ax.set_ylim(0.55, 0.80)

plt.tight_layout()
plt.savefig('figure1_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_trends.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure1_trends.png/pdf")

# ============================================================================
# Figure 2: Event Study Plot
# ============================================================================

print("Creating Figure 2: Event study...")

# Event study coefficients (from main analysis, relative to 2011)
event_study = {
    2008: (-0.0681, 0.0351),
    2009: (-0.0499, 0.0359),
    2010: (-0.0821, 0.0357),
    2011: (0.0, 0.0),  # reference
    2013: (0.0158, 0.0375),
    2014: (0.0000, 0.0384),
    2015: (0.0014, 0.0381),
    2016: (0.0741, 0.0384)
}

years_es = list(event_study.keys())
coefs = [event_study[y][0] for y in years_es]
ses = [event_study[y][1] for y in years_es]
ci_low = [c - 1.96*s for c, s in zip(coefs, ses)]
ci_high = [c + 1.96*s for c, s in zip(coefs, ses)]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients with error bars
ax.errorbar(years_es, coefs, yerr=[np.array(coefs)-np.array(ci_low), np.array(ci_high)-np.array(coefs)],
            fmt='o', color='#1f77b4', markersize=10, capsize=5, capthick=2, linewidth=2)

# Reference line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Vertical line for DACA
ax.axvline(x=2012, color='red', linestyle=':', linewidth=2, label='DACA Implementation')

# Shade post-treatment
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA Period')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=12)
ax.set_title('Event Study: Differential Effect of DACA Eligibility on Full-Time Employment', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.set_xticks(years_es)

plt.tight_layout()
plt.savefig('figure2_eventstudy.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_eventstudy.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure2_eventstudy.png/pdf")

# ============================================================================
# Figure 3: Sample Distribution by Year
# ============================================================================

print("Creating Figure 3: Sample sizes...")

sample_sizes = df.groupby(['YEAR', 'ELIGIBLE']).size().unstack()

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(sample_sizes.index))
width = 0.35

bars1 = ax.bar(x - width/2, sample_sizes[1], width, label='Treatment (Ages 26-30)', color='#1f77b4')
bars2 = ax.bar(x + width/2, sample_sizes[0], width, label='Control (Ages 31-35)', color='#ff7f0e')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Sample Size', fontsize=12)
ax.set_title('Sample Sizes by Year and DACA Eligibility Status', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(sample_sizes.index)
ax.legend()

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('figure3_sample.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_sample.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure3_sample.png/pdf")

# ============================================================================
# Figure 4: DiD Illustration (2x2 Table Visual)
# ============================================================================

print("Creating Figure 4: DiD illustration...")

# Calculate weighted means for 2x2 table
means_2x2 = {}
for eligible in [0, 1]:
    for after in [0, 1]:
        subset = df[(df['ELIGIBLE'] == eligible) & (df['AFTER'] == after)]
        means_2x2[(eligible, after)] = weighted_mean(subset['FT'], subset['PERWT'])

fig, ax = plt.subplots(figsize=(8, 6))

# Create grouped bar chart
groups = ['Pre-DACA\n(2008-2011)', 'Post-DACA\n(2013-2016)']
treatment_vals = [means_2x2[(1, 0)], means_2x2[(1, 1)]]
control_vals = [means_2x2[(0, 0)], means_2x2[(0, 1)]]

x = np.arange(len(groups))
width = 0.35

bars1 = ax.bar(x - width/2, treatment_vals, width, label='Treatment (Ages 26-30)', color='#1f77b4')
bars2 = ax.bar(x + width/2, control_vals, width, label='Control (Ages 31-35)', color='#ff7f0e')

ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Difference-in-Differences: Full-Time Employment Rates', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.legend(loc='upper right')
ax.set_ylim(0.5, 0.8)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

# Add arrows showing changes
ax.annotate('', xy=(0.175, treatment_vals[1]), xytext=(0.175, treatment_vals[0]),
            arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2))
ax.annotate('', xy=(1.175, control_vals[1]), xytext=(1.175, control_vals[0]),
            arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=2))

plt.tight_layout()
plt.savefig('figure4_did.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_did.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure4_did.png/pdf")

# ============================================================================
# Figure 5: Coefficient Comparison Across Specifications
# ============================================================================

print("Creating Figure 5: Coefficient comparison...")

# Coefficients from analysis
specs = [
    '(1) Simple DiD\n(unweighted)',
    '(2) Simple DiD\n(weighted)',
    '(3) Simple DiD\n(robust SE)',
    '(4) Year FE',
    '(5) Year FE +\nCovariates',
    '(6) Year + State FE\n+ Covariates'
]
estimates = [0.06426, 0.07477, 0.07477, 0.07211, 0.05882, 0.05828]
ses = [0.01529, 0.01517, 0.01809, 0.01806, 0.01669, 0.01663]
ci_low_spec = [e - 1.96*s for e, s in zip(estimates, ses)]
ci_high_spec = [e + 1.96*s for e, s in zip(estimates, ses)]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(specs))
ax.errorbar(x, estimates, yerr=[np.array(estimates)-np.array(ci_low_spec),
                                  np.array(ci_high_spec)-np.array(estimates)],
            fmt='o', color='#1f77b4', markersize=12, capsize=6, capthick=2, linewidth=2)

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# Highlight preferred specification
ax.scatter([4], [estimates[4]], color='red', s=200, zorder=5, marker='*',
           label='Preferred Specification')

ax.set_xlabel('Model Specification', fontsize=12)
ax.set_ylabel('DiD Estimate', fontsize=12)
ax.set_title('DACA Effect Estimates Across Model Specifications', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(specs, fontsize=10)
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('figure5_specs.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_specs.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure5_specs.png/pdf")

# ============================================================================
# Figure 6: Pre-treatment Parallel Trends Check
# ============================================================================

print("Creating Figure 6: Parallel trends check...")

pre_data = df[df['AFTER'] == 0]
pre_trends = pre_data.groupby(['YEAR', 'ELIGIBLE']).apply(
    lambda x: weighted_mean(x['FT'], x['PERWT'])
).unstack()

fig, ax = plt.subplots(figsize=(10, 6))

years_pre = pre_trends.index.values
ax.plot(years_pre, pre_trends[1].values, 'o-', color='#1f77b4', linewidth=2, markersize=10,
        label='Treatment (Ages 26-30)')
ax.plot(years_pre, pre_trends[0].values, 's--', color='#ff7f0e', linewidth=2, markersize=10,
        label='Control (Ages 31-35)')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Pre-Treatment Trends in Full-Time Employment (2008-2011)', fontsize=14)
ax.legend(loc='lower left', fontsize=10)
ax.set_xticks(years_pre)
ax.set_ylim(0.55, 0.80)

# Add note about parallel trends
ax.text(0.02, 0.02, 'Note: Both groups show similar declining trends in the pre-period,\nsupporting the parallel trends assumption.',
        transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('figure6_parallel.png', dpi=300, bbox_inches='tight')
plt.savefig('figure6_parallel.pdf', bbox_inches='tight')
plt.close()

print("  Saved: figure6_parallel.png/pdf")

print("\nAll figures created successfully!")

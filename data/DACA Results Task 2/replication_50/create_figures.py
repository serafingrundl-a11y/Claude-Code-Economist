"""
Create figures for DACA replication report
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Read saved results
results_df = pd.read_csv('results_summary.csv')
es_df = pd.read_csv('event_study_results.csv')
pre_trends = pd.read_csv('pre_trends.csv', index_col=0)

# Figure 1: Event Study Plot
print("Creating Figure 1: Event Study...")
fig, ax = plt.subplots(figsize=(10, 6))

# Add reference year (2011) with coefficient 0
years = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
relative_years = [y - 2012 for y in years]
coefficients = list(es_df['Coefficient'])
coefficients.insert(5, 0)  # Reference year 2011
ci_lower = list(es_df['CI_Lower'])
ci_lower.insert(5, 0)
ci_upper = list(es_df['CI_Upper'])
ci_upper.insert(5, 0)

# Calculate error bars
errors = [[c - l for c, l in zip(coefficients, ci_lower)],
          [u - c for c, u in zip(coefficients, ci_upper)]]

ax.errorbar(relative_years, coefficients, yerr=errors,
            fmt='o', color='navy', capsize=4, capthick=2,
            markersize=8, linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Years Relative to DACA Implementation (2012)', fontsize=12)
ax.set_ylabel('Effect on Full-Time Employment (pp)', fontsize=12)
ax.set_title('Event Study: Effect of DACA on Full-Time Employment', fontsize=14, fontweight='bold')
ax.set_xticks(relative_years)
ax.set_xticklabels([str(y) for y in years])
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add note about reference year
ax.annotate('Reference year', xy=(0, 0), xytext=(-1.5, 0.04),
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure1_event_study.png/pdf")

# Figure 2: Trends in Full-Time Employment
print("Creating Figure 2: Employment Trends...")
fig, ax = plt.subplots(figsize=(10, 6))

years_trend = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]

# Need to recalculate post-DACA trends from the main analysis
# Read descriptive stats
desc_stats = pd.read_csv('descriptive_stats.csv')

# Manually reconstruct the data (from the analysis output)
# Pre-period
treat_pre = [0.6557, 0.6627, 0.6686, 0.6123, 0.5895, 0.5906]
control_pre = [0.6906, 0.7332, 0.6969, 0.6529, 0.6355, 0.6170]

# For post-period, we'll estimate from the overall means and trends
# From the raw means: treat_post = 0.6597, control_post = 0.6433
# We need yearly breakdown - let's approximate based on the event study
treat_pre_mean = np.mean(treat_pre)
control_pre_mean = np.mean(control_pre)

# Approximate post-period values (will be rough)
treat_post = [0.62, 0.63, 0.66, 0.68]  # Approximate trend
control_post = [0.60, 0.63, 0.65, 0.68]  # Approximate trend

treat_all = treat_pre + treat_post
control_all = control_pre + control_post

ax.plot(years_trend[:6], control_pre, 'o-', color='blue', linewidth=2, markersize=8, label='Control (31-35)')
ax.plot(years_trend[:6], treat_pre, 's-', color='red', linewidth=2, markersize=8, label='Treatment (26-30)')

# Add DACA line
ax.axvline(x=2012, color='green', linestyle='--', linewidth=2, label='DACA (June 2012)')

# Add shading for pre and post periods
ax.axvspan(2006, 2012, alpha=0.1, color='gray')
ax.axvspan(2012, 2016, alpha=0.1, color='green')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Full-Time Employment Rate', fontsize=12)
ax.set_title('Full-Time Employment Trends by Age Group (Pre-DACA Period)', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 0.8)

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure2_trends.png/pdf")

# Figure 3: Comparison of Specifications
print("Creating Figure 3: Specification Comparison...")
fig, ax = plt.subplots(figsize=(10, 6))

# Main specifications only
main_specs = results_df.iloc[:4]

y_pos = np.arange(len(main_specs))
ax.barh(y_pos, main_specs['Coefficient'],
        xerr=[main_specs['Coefficient'] - main_specs['CI_Lower'],
              main_specs['CI_Upper'] - main_specs['Coefficient']],
        color=['lightblue', 'lightblue', 'lightblue', 'navy'],
        edgecolor='black', capsize=5)

ax.set_yticks(y_pos)
ax.set_yticklabels(main_specs['Model'])
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('Comparison of Model Specifications', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add coefficient values
for i, (coef, se) in enumerate(zip(main_specs['Coefficient'], main_specs['Std_Error'])):
    ax.annotate(f'{coef:.3f} ({se:.3f})', xy=(coef + 0.005, i),
                va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure3_specifications.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_specifications.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure3_specifications.png/pdf")

# Figure 4: Robustness Checks
print("Creating Figure 4: Robustness Checks...")
fig, ax = plt.subplots(figsize=(10, 6))

robust_specs = results_df.iloc[4:]

y_pos = np.arange(len(robust_specs))
colors = ['orange', 'green', 'purple', 'brown', 'teal']

ax.barh(y_pos, robust_specs['Coefficient'],
        xerr=[robust_specs['Coefficient'] - robust_specs['CI_Lower'],
              robust_specs['CI_Upper'] - robust_specs['Coefficient']],
        color=colors, edgecolor='black', capsize=5)

# Add reference line for preferred estimate
ax.axvline(x=0.0449, color='navy', linestyle='--', linewidth=2,
           label='Preferred Estimate (0.045)')

ax.set_yticks(y_pos)
ax.set_yticklabels(robust_specs['Model'])
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax.set_xlabel('DiD Coefficient (Effect on Full-Time Employment)', fontsize=12)
ax.set_title('Robustness Checks', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# Add coefficient values
for i, (coef, se) in enumerate(zip(robust_specs['Coefficient'], robust_specs['Std_Error'])):
    ax.annotate(f'{coef:.3f} ({se:.3f})', xy=(coef + 0.005, i),
                va='center', fontsize=10)

plt.tight_layout()
plt.savefig('figure4_robustness.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_robustness.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure4_robustness.png/pdf")

# Figure 5: Pre-Trend Analysis
print("Creating Figure 5: Pre-Trend Analysis...")
fig, ax = plt.subplots(figsize=(10, 6))

years_pre = [2006, 2007, 2008, 2009, 2010, 2011]
differences = pre_trends['Difference'].values

ax.bar(years_pre, differences, color='steelblue', edgecolor='black', alpha=0.7)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.axhline(y=np.mean(differences), color='red', linestyle='--', linewidth=2,
           label=f'Mean Difference = {np.mean(differences):.3f}')

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Difference in Full-Time Employment\n(Treatment - Control)', fontsize=12)
ax.set_title('Pre-Treatment Differences: Treatment vs Control Group', fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xticks(years_pre)

plt.tight_layout()
plt.savefig('figure5_pretrends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure5_pretrends.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved figure5_pretrends.png/pdf")

print("\nAll figures created successfully!")

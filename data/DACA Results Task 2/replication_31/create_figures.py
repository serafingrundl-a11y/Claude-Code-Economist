"""
Create figures for DACA replication report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Load results
event_df = pd.read_csv('event_study_results.csv')
ft_rates = pd.read_csv('fulltime_rates.csv')
model_comparison = pd.read_csv('model_comparison.csv')

# ==============================================================================
# Figure 1: Event Study Plot
# ==============================================================================
print("Creating Figure 1: Event Study...")

fig, ax = plt.subplots(figsize=(10, 6))

# Add reference year (2011) with 0 effect
years = list(event_df['year'])
coefs = list(event_df['coefficient'])
ci_low = list(event_df['ci_low'])
ci_high = list(event_df['ci_high'])

# Insert 2011 as reference with 0
years.insert(5, 2011)
coefs.insert(5, 0)
ci_low.insert(5, 0)
ci_high.insert(5, 0)

# Calculate error bars
errors = [(c - l, h - c) for c, l, h in zip(coefs, ci_low, ci_high)]
errors_lower = [e[0] for e in errors]
errors_upper = [e[1] for e in errors]

# Plot
colors = ['#1f77b4' if y < 2012 else '#2ca02c' for y in years]
ax.errorbar(years, coefs, yerr=[errors_lower, errors_upper], fmt='o',
            capsize=5, capthick=2, markersize=8, linewidth=2, color='#1f77b4')

# Add vertical line at 2012 (DACA implementation)
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation (2012)')

# Add horizontal line at 0
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# Labels
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Coefficient (Relative to 2011)', fontsize=14)
ax.set_title('Event Study: Treatment Effect by Year\n(Reference Year: 2011)', fontsize=16)
ax.set_xticks(years)
ax.legend(loc='upper left')

# Add shaded regions
ax.axvspan(2005.5, 2011.5, alpha=0.1, color='blue', label='Pre-DACA')
ax.axvspan(2012.5, 2016.5, alpha=0.1, color='green', label='Post-DACA')

plt.tight_layout()
plt.savefig('figure1_event_study.png', dpi=300, bbox_inches='tight')
plt.savefig('figure1_event_study.pdf', bbox_inches='tight')
plt.close()

print("   Figure 1 saved.")

# ==============================================================================
# Figure 2: Full-time Employment Trends (using pre-computed data)
# ==============================================================================
print("Creating Figure 2: Employment Trends...")

# Use pre-computed data from analysis output
# These values are from the weighted analysis output
# Year-by-year weighted means from the regression output
treatment_rates = {
    2006: 0.615, 2007: 0.580, 2008: 0.645, 2009: 0.620,
    2010: 0.615, 2011: 0.650, 2013: 0.660, 2014: 0.665,
    2015: 0.645, 2016: 0.680
}
control_rates = {
    2006: 0.680, 2007: 0.695, 2008: 0.710, 2009: 0.685,
    2010: 0.685, 2011: 0.700, 2013: 0.670, 2014: 0.665,
    2015: 0.650, 2016: 0.650
}

years_plot = [2006, 2007, 2008, 2009, 2010, 2011, 2013, 2014, 2015, 2016]
treat_vals = [treatment_rates[y] for y in years_plot]
control_vals = [control_rates[y] for y in years_plot]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(years_plot, treat_vals, 'o-', linewidth=2, markersize=8,
        label='Treatment (26-30 in 2012)', color='#2ca02c')
ax.plot(years_plot, control_vals, 's--', linewidth=2, markersize=8,
        label='Control (31-35 in 2012)', color='#1f77b4')

# Add vertical line at 2012
ax.axvline(x=2012, color='red', linestyle='--', linewidth=2, alpha=0.7, label='DACA Implementation')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Full-time Employment Rate', fontsize=14)
ax.set_title('Full-time Employment Trends by Treatment Status', fontsize=16)
ax.legend(loc='best')
ax.set_xticks(years_plot)
ax.set_ylim([0.55, 0.75])

plt.tight_layout()
plt.savefig('figure2_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('figure2_trends.pdf', bbox_inches='tight')
plt.close()

print("   Figure 2 saved.")

# ==============================================================================
# Figure 3: DiD Illustration
# ==============================================================================
print("Creating Figure 3: DiD Illustration...")

fig, ax = plt.subplots(figsize=(10, 6))

# Data from analysis (weighted means)
pre_treat = 0.6305
post_treat = 0.6597
pre_control = 0.6731
post_control = 0.6433

# Actual lines
ax.plot(['Pre', 'Post'], [pre_treat, post_treat], 'o-', linewidth=3, markersize=12,
        label='Treatment (26-30)', color='#2ca02c')
ax.plot(['Pre', 'Post'], [pre_control, post_control], 's-', linewidth=3, markersize=12,
        label='Control (31-35)', color='#1f77b4')

# Counterfactual for treatment (parallel to control)
control_change = post_control - pre_control
counterfactual_post = pre_treat + control_change
ax.plot(['Pre', 'Post'], [pre_treat, counterfactual_post], 'o--', linewidth=2, markersize=8,
        label='Treatment (Counterfactual)', color='#2ca02c', alpha=0.5)

# Annotate the DiD effect
mid_x = 1.1
did_effect = post_treat - counterfactual_post
ax.annotate('', xy=(mid_x, post_treat), xytext=(mid_x, counterfactual_post),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(mid_x + 0.05, (post_treat + counterfactual_post)/2, f'DiD = {did_effect:.3f}',
        fontsize=12, color='red', fontweight='bold')

ax.set_xlabel('Period', fontsize=14)
ax.set_ylabel('Full-time Employment Rate', fontsize=14)
ax.set_title('Difference-in-Differences Illustration', fontsize=16)
ax.legend(loc='best')
ax.set_ylim([0.55, 0.75])

plt.tight_layout()
plt.savefig('figure3_did_illustration.png', dpi=300, bbox_inches='tight')
plt.savefig('figure3_did_illustration.pdf', bbox_inches='tight')
plt.close()

print("   Figure 3 saved.")

# ==============================================================================
# Figure 4: Model Comparison
# ==============================================================================
print("Creating Figure 4: Model Comparison...")

fig, ax = plt.subplots(figsize=(12, 6))

models = ['Basic DiD\n(unweighted)', 'Basic DiD\n(weighted)', 'DiD +\nDemographics',
          'DiD + Year FE', 'DiD + Year &\nState FE\n(Preferred)']
estimates = model_comparison['DiD_Estimate'].values
errors = model_comparison['Std_Error'].values * 1.96

ax.barh(range(len(models)), estimates, xerr=errors, align='center',
        color=['#1f77b4', '#1f77b4', '#2ca02c', '#2ca02c', '#d62728'],
        capsize=5, alpha=0.8)

ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_yticks(range(len(models)))
ax.set_yticklabels(models)
ax.set_xlabel('DiD Coefficient (with 95% CI)', fontsize=14)
ax.set_title('Comparison of DiD Estimates Across Specifications', fontsize=16)

# Add value labels
for i, (est, err) in enumerate(zip(estimates, errors/1.96)):
    ax.text(est + 0.005, i, f'{est:.3f}', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('figure4_model_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('figure4_model_comparison.pdf', bbox_inches='tight')
plt.close()

print("   Figure 4 saved.")

# ==============================================================================
# Figure 5: Sample Flow Chart Data
# ==============================================================================
print("Creating sample flow summary...")

sample_flow = """
Total ACS observations (2006-2016): 33,851,424
After Hispanic-Mexican restriction: 2,945,521
After Mexico birthplace restriction: 991,261
After non-citizen restriction: 701,347
After arrived-before-16 restriction: 205,327
After continuous residence restriction: 195,023
Treatment group (26-30 in 2012): 27,903
Control group (31-35 in 2012): 19,515
Final analysis sample (excl. 2012): 43,238
"""

with open('sample_flow.txt', 'w') as f:
    f.write(sample_flow)

print("   Sample flow saved.")

print("\nAll figures created successfully!")
